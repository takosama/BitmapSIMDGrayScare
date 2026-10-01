using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace ConsoleApp67
{
    [SimpleJob(RuntimeMoniker.HostProcess, launchCount: 1, warmupCount: 5, iterationCount: 15)]
    [MemoryDiagnoser]
    public unsafe partial class test : IDisposable
    {
        private const int Iterations = 100;
        private const nuint Alignment = 64;
        private const int RedWeight = 77;
        private const int GreenWeight = 150;
        private const int BlueWeight = 29;
        private const float RedWeightFloat = RedWeight / 256f;
        private const float GreenWeightFloat = GreenWeight / 256f;
        private const float BlueWeightFloat = BlueWeight / 256f;
        private const byte AlphaOpaque = 255;

        [ParamsSource(nameof(GetTestImages))]
        public ImageSpec Spec { get; set; } = ImageSpec.Hd;

        private static readonly ParallelOptions RowParallelOptions = new()
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount
        };

        private byte* _source;
        private byte* _destination;
        private readonly Random _random = new(42);

        private delegate void Converter(byte* source, byte* destination, in ImageSpec spec);

        private static readonly Converter ScalarPath = ScalarGrayscale;
        private static readonly Converter Avx2Path = Avx2.IsSupported ? Avx2Grayscale : ScalarGrayscale;
        private static readonly Converter Avx512Path = Avx512F.IsSupported && Avx512BW.IsSupported && Avx2.IsSupported ? Avx512Grayscale : Avx2Path;
        private static readonly Converter FmaPath = Fma.IsSupported && Avx.IsSupported ? FmaGrayscale : Avx2Path;
        private static readonly Converter AdvSimdPath = AdvSimd.Arm64.IsSupported ? AdvSimdGrayscale : ScalarGrayscale;

        public static IEnumerable<ImageSpec> GetTestImages() => new[] { ImageSpec.Hd, ImageSpec.FullHd };

        [GlobalSetup]
        public void Setup()
        {
            Cleanup();

            int byteLength = Spec.BufferLength;
            _source = (byte*)NativeMemory.AlignedAlloc((nuint)byteLength, Alignment);
            _destination = (byte*)NativeMemory.AlignedAlloc((nuint)byteLength, Alignment);
            FillRandom(_source, byteLength);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            if (_source != null)
            {
                NativeMemory.AlignedFree(_source);
                _source = null;
            }

            if (_destination != null)
            {
                NativeMemory.AlignedFree(_destination);
                _destination = null;
            }
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

        [Benchmark(Description = "Legacy Run() entry point")]
        public void Run()
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
        public void OptimizedAdvSimd()
        {
            RunIterations(AdvSimdPath);
        }

        [Benchmark]
        public void ParallelAvx2()
        {
            RunIterations(Avx2.IsSupported ? ParallelAvx2Grayscale : ScalarPath);
        }

        private void RunIterations(Converter converter)
        {
            for (int i = 0; i < Iterations; i++)
            {
                converter(_source, _destination, Spec);
            }
        }

        private void FillRandom(byte* buffer, int length)
        {
            Span<byte> span = new(buffer, length);
            _random.NextBytes(span);
        }

        private static void ScalarGrayscale(byte* source, byte* destination, in ImageSpec spec)
        {
            for (int y = 0; y < spec.Height; y++)
            {
                byte* srcRow = source + y * spec.Stride;
                byte* dstRow = destination + y * spec.Stride;
                ConvertScalarRange(srcRow, dstRow, 0, spec.Width);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ConvertScalarRange(byte* srcRow, byte* dstRow, int startX, int width)
        {
            for (int x = startX; x < width; x++)
            {
                int pixelIndex = x * 4;
                ConvertScalarPixel(srcRow + pixelIndex, dstRow + pixelIndex);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ConvertScalarPixel(byte* srcPixel, byte* dstPixel)
        {
            int gray = (RedWeight * srcPixel[2] + GreenWeight * srcPixel[1] + BlueWeight * srcPixel[0] + 128) >> 8;
            if (gray > 255)
            {
                gray = 255;
            }

            WriteGray(dstPixel, (byte)gray);
        }

        private static void Avx2Grayscale(byte* source, byte* destination, in ImageSpec spec)
        {
            int simdWidth = spec.Width & ~15;

            for (int y = 0; y < spec.Height; y++)
            {
                byte* srcRow = source + y * spec.Stride;
                byte* dstRow = destination + y * spec.Stride;

                int x = 0;
                for (; x < simdWidth; x += 16)
                {
                    int offset = x * 4;
                    Vector256<byte> first = Unsafe.ReadUnaligned<Vector256<byte>>(srcRow + offset);
                    Vector256<byte> second = Unsafe.ReadUnaligned<Vector256<byte>>(srcRow + offset + 32);

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

                ConvertScalarRange(srcRow, dstRow, x, spec.Width);
            }
        }

        private static void Avx512Grayscale(byte* source, byte* destination, in ImageSpec spec)
        {
            int simdWidth = spec.Width & ~15;

            for (int y = 0; y < spec.Height; y++)
            {
                byte* srcRow = source + y * spec.Stride;
                byte* dstRow = destination + y * spec.Stride;

                int x = 0;
                for (; x < simdWidth; x += 16)
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

                ConvertScalarRange(srcRow, dstRow, x, spec.Width);
            }
        }

        private static void FmaGrayscale(byte* source, byte* destination, in ImageSpec spec)
        {
            if (!Fma.IsSupported || !Avx.IsSupported)
            {
                ScalarGrayscale(source, destination, spec);
                return;
            }

            Vector256<float> bWeight = Vector256.Create(BlueWeightFloat);
            Vector256<float> gWeight = Vector256.Create(GreenWeightFloat);
            Vector256<float> rWeight = Vector256.Create(RedWeightFloat);
            Vector256<float> half = Vector256.Create(0.5f);

            Span<float> bufferB = stackalloc float[8];
            Span<float> bufferG = stackalloc float[8];
            Span<float> bufferR = stackalloc float[8];
            Span<float> grayTemp = stackalloc float[8];

            for (int y = 0; y < spec.Height; y++)
            {
                byte* srcRow = source + y * spec.Stride;
                byte* dstRow = destination + y * spec.Stride;
                int x = 0;
                int simdWidth = spec.Width & ~7;

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

                    Vector256<float> bVec = Unsafe.ReadUnaligned<Vector256<float>>(ref Unsafe.As<float, byte>(ref MemoryMarshal.GetReference(bufferB)));
                    Vector256<float> gVec = Unsafe.ReadUnaligned<Vector256<float>>(ref Unsafe.As<float, byte>(ref MemoryMarshal.GetReference(bufferG)));
                    Vector256<float> rVec = Unsafe.ReadUnaligned<Vector256<float>>(ref Unsafe.As<float, byte>(ref MemoryMarshal.GetReference(bufferR)));

                    Vector256<float> gray = Fma.MultiplyAdd(rVec, rWeight, Fma.MultiplyAdd(gVec, gWeight, Avx.Multiply(bVec, bWeight)));
                    gray = Avx.Add(gray, half);

                    Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref MemoryMarshal.GetReference(grayTemp)), gray);

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

                        WriteGray(dstRow + pixelIndex, (byte)gValue);
                    }
                }

                for (; x < spec.Width; x++)
                {
                    int pixelIndex = x * 4;
                    float gray = srcRow[pixelIndex + 2] * RedWeightFloat + srcRow[pixelIndex + 1] * GreenWeightFloat + srcRow[pixelIndex + 0] * BlueWeightFloat + 0.5f;
                    byte grayByte = (byte)Math.Clamp((int)gray, 0, 255);
                    WriteGray(dstRow + pixelIndex, grayByte);
                }
            }
        }

        private static void ParallelAvx2Grayscale(byte* source, byte* destination, in ImageSpec spec)
        {
            // An in parameter cannot be captured by a lambda. Capture scalar values.
            int stride = spec.Stride;
            int width = spec.Width;
            Parallel.For(0, spec.Height, RowParallelOptions, y =>
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
                Vector256<byte> first = Unsafe.ReadUnaligned<Vector256<byte>>(srcRow + offset);
                Vector256<byte> second = Unsafe.ReadUnaligned<Vector256<byte>>(srcRow + offset + 32);
                Vector256<byte> lowerGray = ConvertBlock256(first);
                Vector256<byte> upperGray = ConvertBlock256(second);
                Avx.Store(dstRow + offset, lowerGray);
                Avx.Store(dstRow + offset + 32, upperGray);
            }

            ConvertScalarRange(srcRow, dstRow, x, width);
        }

        private static void AdvSimdGrayscale(byte* source, byte* destination, in ImageSpec spec)
        {
            if (!AdvSimd.Arm64.IsSupported)
            {
                ScalarGrayscale(source, destination, spec);
                return;
            }

            Span<byte> lane = stackalloc byte[16];
            byte* lanePtr = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetReference(lane));

            for (int y = 0; y < spec.Height; y++)
            {
                byte* srcRow = source + y * spec.Stride;
                byte* dstRow = destination + y * spec.Stride;

                int x = 0;
                int simdWidth = spec.Width & ~3;

                for (; x < simdWidth; x += 4)
                {
                    int offset = x * 4;
                    Vector128<byte> data = AdvSimd.LoadVector128(srcRow + offset);
                    Unsafe.WriteUnaligned(ref lane[0], data);

                    for (int i = 0; i < 4; i++)
                    {
                        int laneIndex = i * 4;
                        int pixelIndex = offset + laneIndex;
                        ConvertScalarPixel(lanePtr + laneIndex, dstRow + pixelIndex);
                    }
                }

                ConvertScalarRange(srcRow, dstRow, x, spec.Width);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteGray(byte* dstPixel, byte gray)
        {
            dstPixel[0] = gray;
            dstPixel[1] = gray;
            dstPixel[2] = gray;
            dstPixel[3] = AlphaOpaque;
        }

        private static Vector256<byte> ConvertBlock256(Vector256<byte> bgra)
        {
            // One 32-bit lane per BGRA pixel avoids signed-byte weights, saturating
            // partial sums and 128-bit shuffle lane mistakes. Match scalar exactly.
            Vector256<int> pixels = bgra.AsInt32();
            Vector256<int> mask = Vector256.Create(255);
            Vector256<int> blue = Avx2.And(pixels, mask);
            Vector256<int> green = Avx2.And(Avx2.ShiftRightLogical(pixels, 8), mask);
            Vector256<int> red = Avx2.And(Avx2.ShiftRightLogical(pixels, 16), mask);
            Vector256<int> sum = Avx2.Add(
                Avx2.Add(Avx2.MultiplyLow(blue, Vector256.Create(BlueWeight)),
                         Avx2.MultiplyLow(green, Vector256.Create(GreenWeight))),
                Avx2.MultiplyLow(red, Vector256.Create(RedWeight)));
            Vector256<int> gray = Avx2.ShiftRightLogical(Avx2.Add(sum, Vector256.Create(128)), 8);
            Vector256<int> packed = Avx2.Or(gray, Avx2.ShiftLeftLogical(gray, 8));
            packed = Avx2.Or(packed, Avx2.ShiftLeftLogical(gray, 16));
            return Avx2.Or(packed, Vector256.Create(unchecked((int)0xFF000000))).AsByte();
        }

    }

    public readonly record struct ImageSpec(int Width, int Height, string Name)
    {
        public static ImageSpec Hd { get; } = new(1280, 720, "1280x720");
        public static ImageSpec FullHd { get; } = new(1920, 1080, "1920x1080");

        public int Stride => Width * 4;
        public int BufferLength => Stride * Height;

        public override string ToString() => Name;
    }
}
