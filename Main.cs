using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

namespace BitmapSIMDGrayScare
{
    public class Program
    {
        public static void Main(string[] args)
        {
            BenchmarkRunner.Run<GrayscaleBenchmark>();
        }
    }

    [SimpleJob(RuntimeMoniker.HostProcess, launchCount: 1, warmupCount: 5, iterationCount: 15)]
    [MemoryDiagnoser]
    public unsafe class GrayscaleBenchmark : IDisposable
    {
        private const int Width = 1280;
        private const int Height = 720;
        private const int Iterations = 100;
        private const int Alignment = 64;

        private byte* _source;
        private byte* _destination;
        private int _bufferLength;

        private readonly Random _random = new(42);

        private static readonly int ProcessorCount = Environment.ProcessorCount;

        private delegate void Converter(byte* source, byte* destination, int width, int height, int stride);

        private static readonly Converter ScalarPath = ScalarGrayscale;
        private static readonly Converter Avx2Path = Avx2.IsSupported ? Avx2Grayscale : ScalarGrayscale;
        private static readonly Converter Avx512Path = Avx512F.IsSupported && Avx512BW.IsSupported ? Avx512Grayscale : Avx2Path;
        private static readonly Converter FmaPath = Fma.IsSupported && Avx.IsSupported ? FmaGrayscale : Avx2Path;
        private static readonly Converter ArmPath = AdvSimd.Arm64.IsSupported ? AdvSimdGrayscale : ScalarGrayscale;

        [GlobalSetup]
        public void Setup()
        {
            _bufferLength = Width * Height * 4;
            _source = (byte*)NativeMemory.AlignedAlloc((nuint)_bufferLength, (nuint)Alignment);
            _destination = (byte*)NativeMemory.AlignedAlloc((nuint)_bufferLength, (nuint)Alignment);

            FillRandom(_source, _bufferLength);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            NativeMemory.AlignedFree(_source);
            NativeMemory.AlignedFree(_destination);
        }

        public void Dispose()
        {
            Cleanup();
        }

        [Benchmark(Baseline = true)]
        public void BaselineScalar()
        {
            RunIterations(ScalarPath);
        }

        [Benchmark]
        public void OptimizedAvx2()
        {
            RunIterations(Avx2Path);
        }

        [Benchmark]
        public void OptimizedAvx512()
        {
            RunIterations(Avx512Path);
        }

        [Benchmark]
        public void OptimizedFma()
        {
            RunIterations(FmaPath);
        }

        [Benchmark]
        public void OptimizedArm()
        {
            RunIterations(ArmPath);
        }

        [Benchmark]
        public void ParallelAvx2()
        {
            if (!Avx2.IsSupported)
            {
                RunIterations(ScalarPath);
                return;
            }

            RunIterations(ParallelAvx2Grayscale);
        }

        private void RunIterations(Converter converter)
        {
            for (int i = 0; i < Iterations; i++)
            {
                converter(_source, _destination, Width, Height, Width * 4);
            }
        }

        private void FillRandom(byte* buffer, int length)
        {
            Span<byte> span = new(buffer, length);
            _random.NextBytes(span);
        }
        private static void ScalarGrayscale(byte* source, byte* destination, int width, int height, int stride)
        {
            const int rWeight = 77;
            const int gWeight = 150;
            const int bWeight = 29;

            for (int y = 0; y < height; y++)
            {
                byte* srcRow = source + y * stride;
                byte* dstRow = destination + y * stride;

                for (int x = 0; x < width; x++)
                {
                    int pixelIndex = x * 4;
                    byte b = srcRow[pixelIndex + 0];
                    byte g = srcRow[pixelIndex + 1];
                    byte r = srcRow[pixelIndex + 2];

                    int gray = (rWeight * r + gWeight * g + bWeight * b + 128) >> 8;
                    if (gray > 255)
                    {
                        gray = 255;
                    }

                    dstRow[pixelIndex + 0] = (byte)gray;
                    dstRow[pixelIndex + 1] = (byte)gray;
                    dstRow[pixelIndex + 2] = (byte)gray;
                    dstRow[pixelIndex + 3] = 255;
                }
            }
        }

        private static void Avx2Grayscale(byte* source, byte* destination, int width, int height, int stride)
        {
            int simdWidth = width & ~15;

            for (int y = 0; y < height; y++)
            {
                byte* srcRow = source + y * stride;
                byte* dstRow = destination + y * stride;

                int x = 0;
                for (; x < simdWidth; x += 16)
                {
                    int offset = x * 4;
                    Vector256<byte> first = Avx.LoadVector256(srcRow + offset);
                    Vector256<byte> second = Avx.LoadVector256(srcRow + offset + 32);

                    if (Sse3.IsSupported)
                    {
                        Sse.Prefetch0(srcRow + offset + 64);
                        Sse.Prefetch0(dstRow + offset + 64);
                    }

                    Vector256<byte> lowerGray = ConvertBlock256(first);
                    Vector256<byte> upperGray = ConvertBlock256(second);

                    Avx.Store(dstRow + offset, lowerGray);
                    Avx.Store(dstRow + offset + 32, upperGray);
                }

                for (; x < width; x++)
                {
                    int pixelIndex = x * 4;
                    byte b = srcRow[pixelIndex + 0];
                    byte g = srcRow[pixelIndex + 1];
                    byte r = srcRow[pixelIndex + 2];

                    int gray = (77 * r + 150 * g + 29 * b + 128) >> 8;
                    if (gray > 255)
                    {
                        gray = 255;
                    }

                    dstRow[pixelIndex + 0] = (byte)gray;
                    dstRow[pixelIndex + 1] = (byte)gray;
                    dstRow[pixelIndex + 2] = (byte)gray;
                    dstRow[pixelIndex + 3] = 255;
                }
            }
        }

        private static void Avx512Grayscale(byte* source, byte* destination, int width, int height, int stride)
        {
            int simdWidth = width & ~31;

            for (int y = 0; y < height; y++)
            {
                byte* srcRow = source + y * stride;
                byte* dstRow = destination + y * stride;

                int x = 0;
                for (; x < simdWidth; x += 32)
                {
                    int offset = x * 4;
                    Vector512<byte> block = Avx512F.LoadVector512(srcRow + offset);
                    Vector256<byte> lower = block.GetLower();
                    Vector256<byte> upper = block.GetUpper();

                    Vector256<byte> lowerGray = ConvertBlock256(lower);
                    Vector256<byte> upperGray = ConvertBlock256(upper);

                    Avx.Store(dstRow + offset, lowerGray);
                    Avx.Store(dstRow + offset + 32, upperGray);
                }

                for (; x < width; x++)
                {
                    int pixelIndex = x * 4;
                    byte b = srcRow[pixelIndex + 0];
                    byte g = srcRow[pixelIndex + 1];
                    byte r = srcRow[pixelIndex + 2];

                    int gray = (77 * r + 150 * g + 29 * b + 128) >> 8;
                    if (gray > 255)
                    {
                        gray = 255;
                    }

                    dstRow[pixelIndex + 0] = (byte)gray;
                    dstRow[pixelIndex + 1] = (byte)gray;
                    dstRow[pixelIndex + 2] = (byte)gray;
                    dstRow[pixelIndex + 3] = 255;
                }
            }
        }
        private static void FmaGrayscale(byte* source, byte* destination, int width, int height, int stride)
        {
            Span<float> weights = stackalloc float[] { 0.114f, 0.587f, 0.299f };
            Vector256<float> bWeight = Vector256.Create(weights[0]);
            Vector256<float> gWeight = Vector256.Create(weights[1]);
            Vector256<float> rWeight = Vector256.Create(weights[2]);
            Vector256<float> half = Vector256.Create(0.5f);

            Span<float> bufferB = stackalloc float[8];
            Span<float> bufferG = stackalloc float[8];
            Span<float> bufferR = stackalloc float[8];
            Span<float> grayTemp = stackalloc float[8];

            for (int y = 0; y < height; y++)
            {
                byte* srcRow = source + y * stride;
                byte* dstRow = destination + y * stride;
                int x = 0;
                int simdWidth = width & ~7;

                for (; x < simdWidth; x += 8)
                {
                    int offset = x * 4;
                    for (int i = 0; i < 8; i++)
                    {
                        int pixelIndex = offset + i * 4;
                        bufferB[i] = srcRow[pixelIndex + 0];
                        bufferG[i] = srcRow[pixelIndex + 1];
                        bufferR[i] = srcRow[pixelIndex + 2];
                    }

                    Vector256<float> bVec = Unsafe.ReadUnaligned<Vector256<float>>(ref MemoryMarshal.GetReference(bufferB));
                    Vector256<float> gVec = Unsafe.ReadUnaligned<Vector256<float>>(ref MemoryMarshal.GetReference(bufferG));
                    Vector256<float> rVec = Unsafe.ReadUnaligned<Vector256<float>>(ref MemoryMarshal.GetReference(bufferR));

                    Vector256<float> gray = Fma.MultiplyAdd(rVec, rWeight, Fma.MultiplyAdd(gVec, gWeight, Avx.Multiply(bVec, bWeight)));
                    gray = Avx.Add(gray, half);

                    Unsafe.WriteUnaligned(ref MemoryMarshal.GetReference(grayTemp), gray);

                    for (int i = 0; i < 8; i++)
                    {
                        int pixelIndex = offset + i * 4;
                        int gValue = (int)grayTemp[i];
                        if (gValue > 255)
                        {
                            gValue = 255;
                        }
                        else if (gValue < 0)
                        {
                            gValue = 0;
                        }

                        byte grayByte = (byte)gValue;
                        dstRow[pixelIndex + 0] = grayByte;
                        dstRow[pixelIndex + 1] = grayByte;
                        dstRow[pixelIndex + 2] = grayByte;
                        dstRow[pixelIndex + 3] = 255;
                    }
                }

                for (; x < width; x++)
                {
                    int pixelIndex = x * 4;
                    float gray = srcRow[pixelIndex + 2] * weights[2] + srcRow[pixelIndex + 1] * weights[1] + srcRow[pixelIndex + 0] * weights[0] + 0.5f;
                    byte grayByte = (byte)Math.Clamp((int)gray, 0, 255);
                    dstRow[pixelIndex + 0] = grayByte;
                    dstRow[pixelIndex + 1] = grayByte;
                    dstRow[pixelIndex + 2] = grayByte;
                    dstRow[pixelIndex + 3] = 255;
                }
            }
        }

        private static void ParallelAvx2Grayscale(byte* source, byte* destination, int width, int height, int stride)
        {
            Parallel.For(0, height, new ParallelOptions { MaxDegreeOfParallelism = ProcessorCount }, y =>
            {
                byte* srcRow = source + y * stride;
                byte* dstRow = destination + y * stride;
                Avx2Row(srcRow, dstRow, width);
            });
        }

        private static void Avx2Row(byte* srcRow, byte* dstRow, int width)
        {
            int simdWidth = width & ~15;
            int x = 0;
            for (; x < simdWidth; x += 16)
            {
                int offset = x * 4;
                Vector256<byte> first = Avx.LoadVector256(srcRow + offset);
                Vector256<byte> second = Avx.LoadVector256(srcRow + offset + 32);
                Vector256<byte> lowerGray = ConvertBlock256(first);
                Vector256<byte> upperGray = ConvertBlock256(second);
                Avx.Store(dstRow + offset, lowerGray);
                Avx.Store(dstRow + offset + 32, upperGray);
            }

            for (; x < width; x++)
            {
                int pixelIndex = x * 4;
                byte b = srcRow[pixelIndex + 0];
                byte g = srcRow[pixelIndex + 1];
                byte r = srcRow[pixelIndex + 2];

                int gray = (77 * r + 150 * g + 29 * b + 128) >> 8;
                if (gray > 255)
                {
                    gray = 255;
                }

                dstRow[pixelIndex + 0] = (byte)gray;
                dstRow[pixelIndex + 1] = (byte)gray;
                dstRow[pixelIndex + 2] = (byte)gray;
                dstRow[pixelIndex + 3] = 255;
            }
        }
        private static void AdvSimdGrayscale(byte* source, byte* destination, int width, int height, int stride)
        {
            if (!AdvSimd.Arm64.IsSupported)
            {
                ScalarGrayscale(source, destination, width, height, stride);
                return;
            }

            Span<byte> lane = stackalloc byte[16];

            for (int y = 0; y < height; y++)
            {
                byte* srcRow = source + y * stride;
                byte* dstRow = destination + y * stride;

                int x = 0;
                int simdWidth = width & ~3;

                for (; x < simdWidth; x += 4)
                {
                    int offset = x * 4;
                    Vector128<byte> data = AdvSimd.LoadVector128(srcRow + offset);
                    Unsafe.WriteUnaligned(ref lane[0], data);

                    for (int i = 0; i < 4; i++)
                    {
                        int pixelIndex = i * 4;
                        byte b = lane[pixelIndex + 0];
                        byte g = lane[pixelIndex + 1];
                        byte r = lane[pixelIndex + 2];

                        int gray = (77 * r + 150 * g + 29 * b + 128) >> 8;
                        if (gray > 255)
                        {
                            gray = 255;
                        }

                        byte grayByte = (byte)gray;
                        dstRow[offset + pixelIndex + 0] = grayByte;
                        dstRow[offset + pixelIndex + 1] = grayByte;
                        dstRow[offset + pixelIndex + 2] = grayByte;
                        dstRow[offset + pixelIndex + 3] = 255;
                    }
                }

                for (; x < width; x++)
                {
                    int pixelIndex = x * 4;
                    byte b = srcRow[pixelIndex + 0];
                    byte g = srcRow[pixelIndex + 1];
                    byte r = srcRow[pixelIndex + 2];
                    int gray = (77 * r + 150 * g + 29 * b + 128) >> 8;
                    if (gray > 255)
                    {
                        gray = 255;
                    }

                    dstRow[pixelIndex + 0] = (byte)gray;
                    dstRow[pixelIndex + 1] = (byte)gray;
                    dstRow[pixelIndex + 2] = (byte)gray;
                    dstRow[pixelIndex + 3] = 255;
                }
            }
        }
        private static Vector256<byte> ConvertBlock256(Vector256<byte> bgra)
        {
            Vector256<byte> bgPairs = Avx2.Shuffle(bgra, ShuffleBgMask256);
            Vector256<short> bgWeighted = Avx2.MultiplyAddAdjacent(bgPairs, BgWeights256);

            Vector256<byte> gPairs = Avx2.Shuffle(bgra, ShuffleGMask256);
            Vector256<short> gWeighted = Avx2.MultiplyAddAdjacent(gPairs, GWeights256);

            Vector256<byte> rPairs = Avx2.Shuffle(bgra, ShuffleRMask256);
            Vector256<short> rWeighted = Avx2.MultiplyAddAdjacent(rPairs, RWeights256);

            Vector256<short> sum = Avx2.Add(bgWeighted, gWeighted);
            sum = Avx2.Add(sum, rWeighted);
            sum = Avx2.Add(sum, Rounding256);

            Vector256<short> gray16 = Avx2.ShiftRightLogical(sum.AsUInt16(), 8).AsInt16();
            Vector256<byte> grayBytes = Avx2.PackUnsignedSaturate(gray16, gray16);

            Vector256<byte> replicated = Avx2.Shuffle(grayBytes, DuplicateGrayShuffle256);
            return Avx2.Or(replicated, AlphaMask256);
        }

        private static readonly Vector256<byte> AlphaMask256 = Vector256.Create(
            (byte)0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
            0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255);

        private static readonly Vector256<short> Rounding256 = Vector256.Create(
            (short)128, 128, 128, 128, 128, 128, 128, 128,
            0, 0, 0, 0, 0, 0, 0, 0);

        private static readonly Vector256<sbyte> ShuffleBgMask256 = Vector256.Create(
            (sbyte)0, (sbyte)1, (sbyte)4, (sbyte)5, (sbyte)8, (sbyte)9, (sbyte)12, (sbyte)13,
            (sbyte)16, (sbyte)17, (sbyte)20, (sbyte)21, (sbyte)24, (sbyte)25, (sbyte)28, (sbyte)29,
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80));

        private static readonly Vector256<sbyte> ShuffleGMask256 = Vector256.Create(
            (sbyte)1, unchecked((sbyte)0x80), (sbyte)5, unchecked((sbyte)0x80), (sbyte)9, unchecked((sbyte)0x80), (sbyte)13, unchecked((sbyte)0x80),
            (sbyte)17, unchecked((sbyte)0x80), (sbyte)21, unchecked((sbyte)0x80), (sbyte)25, unchecked((sbyte)0x80), (sbyte)29, unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80));

        private static readonly Vector256<sbyte> ShuffleRMask256 = Vector256.Create(
            (sbyte)2, unchecked((sbyte)0x80), (sbyte)6, unchecked((sbyte)0x80), (sbyte)10, unchecked((sbyte)0x80), (sbyte)14, unchecked((sbyte)0x80),
            (sbyte)18, unchecked((sbyte)0x80), (sbyte)22, unchecked((sbyte)0x80), (sbyte)26, unchecked((sbyte)0x80), (sbyte)30, unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80),
            unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80), unchecked((sbyte)0x80));

        private static readonly Vector256<sbyte> DuplicateGrayShuffle256 = Vector256.Create(
            (sbyte)0, (sbyte)0, (sbyte)0, unchecked((sbyte)0x80), (sbyte)1, (sbyte)1, (sbyte)1, unchecked((sbyte)0x80),
            (sbyte)2, (sbyte)2, (sbyte)2, unchecked((sbyte)0x80), (sbyte)3, (sbyte)3, (sbyte)3, unchecked((sbyte)0x80),
            (sbyte)4, (sbyte)4, (sbyte)4, unchecked((sbyte)0x80), (sbyte)5, (sbyte)5, (sbyte)5, unchecked((sbyte)0x80),
            (sbyte)6, (sbyte)6, (sbyte)6, unchecked((sbyte)0x80), (sbyte)7, (sbyte)7, (sbyte)7, unchecked((sbyte)0x80));

        private static readonly Vector256<sbyte> BgWeights256 = Vector256.Create(
            (sbyte)29, (sbyte)22, (sbyte)29, (sbyte)22, (sbyte)29, (sbyte)22, (sbyte)29, (sbyte)22,
            (sbyte)29, (sbyte)22, (sbyte)29, (sbyte)22, (sbyte)29, (sbyte)22, (sbyte)29, (sbyte)22,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

        private static readonly Vector256<sbyte> GWeights256 = Vector256.Create(
            (sbyte)150, 0, (sbyte)150, 0, (sbyte)150, 0, (sbyte)150, 0,
            (sbyte)150, 0, (sbyte)150, 0, (sbyte)150, 0, (sbyte)150, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

        private static readonly Vector256<sbyte> RWeights256 = Vector256.Create(
            (sbyte)77, 0, (sbyte)77, 0, (sbyte)77, 0, (sbyte)77, 0,
            (sbyte)77, 0, (sbyte)77, 0, (sbyte)77, 0, (sbyte)77, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    }
}
