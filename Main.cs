using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace BitmapSIMDGrayScare
{
    public class Program
    {
        public static void Main(string[] args)
        {
            BenchmarkRunner.Run<GrayscaleBenchmark>();
        }
    }

    [MemoryDiagnoser]
    public class GrayscaleBenchmark
    {
        private const int Width = 1280;
        private const int Height = 720;
        private const int Iterations = 100;

        private byte[] _source = Array.Empty<byte>();
        private byte[] _destination = Array.Empty<byte>();

        [GlobalSetup]
        public void Setup()
        {
            _source = new byte[Width * Height * 4];
            _destination = new byte[_source.Length];

            var random = new Random(42);
            random.NextBytes(_source);
        }

        [Benchmark(Baseline = true)]
        public void Baseline()
        {
            RunIterations(ScalarGrayscale);
        }

        [Benchmark]
        public void Avx2Optimized()
        {
            if (Avx2.IsSupported && Avx.IsSupported)
            {
                RunIterations(Avx2Grayscale);
            }
            else
            {
                RunIterations(ScalarGrayscale);
            }
        }

        private void RunIterations(Action<byte[], byte[], int, int> converter)
        {
            for (int i = 0; i < Iterations; i++)
            {
                converter(_source, _destination, Width, Height);
            }
        }

        private static void ScalarGrayscale(byte[] source, byte[] destination, int width, int height)
        {
            const float rWeight = 0.299f;
            const float gWeight = 0.587f;
            const float bWeight = 0.114f;

            for (int y = 0; y < height; y++)
            {
                int rowOffset = y * width * 4;
                for (int x = 0; x < width; x++)
                {
                    int pixelIndex = rowOffset + x * 4;
                    byte b = source[pixelIndex + 0];
                    byte g = source[pixelIndex + 1];
                    byte r = source[pixelIndex + 2];

                    var gray = (byte)Math.Clamp((int)(r * rWeight + g * gWeight + b * bWeight + 0.5f), 0, 255);

                    destination[pixelIndex + 0] = gray;
                    destination[pixelIndex + 1] = gray;
                    destination[pixelIndex + 2] = gray;
                    destination[pixelIndex + 3] = 255;
                }
            }
        }

        private static unsafe void Avx2Grayscale(byte[] source, byte[] destination, int width, int height)
        {
            const float rWeightValue = 0.299f;
            const float gWeightValue = 0.587f;
            const float bWeightValue = 0.114f;

            fixed (byte* sourcePtr = source)
            fixed (byte* destinationPtr = destination)
            {
                int stride = width * 4;
                int simdWidth = width & ~7; // process 8 pixels per iteration

                Vector256<int> maskByte = Vector256.Create(0xFF);
                Vector256<int> alphaMask = Vector256.Create(unchecked((int)0xFF000000));
                Vector256<int> zero = Vector256<int>.Zero;
                Vector256<int> max = Vector256.Create(255);

                Vector256<float> rWeight = Vector256.Create(rWeightValue);
                Vector256<float> gWeight = Vector256.Create(gWeightValue);
                Vector256<float> bWeight = Vector256.Create(bWeightValue);
                Vector256<float> rounding = Vector256.Create(0.5f);

                for (int y = 0; y < height; y++)
                {
                    byte* srcRow = sourcePtr + y * stride;
                    byte* dstRow = destinationPtr + y * stride;

                    int x = 0;
                    for (; x < simdWidth; x += 8)
                    {
                        int byteOffset = x * 4;

                        Vector256<int> pixels = Avx.LoadVector256((int*)(srcRow + byteOffset));

                        Vector256<int> blue = Avx2.And(pixels, maskByte);
                        Vector256<int> green = Avx2.And(Avx2.ShiftRightLogical(pixels, 8), maskByte);
                        Vector256<int> red = Avx2.And(Avx2.ShiftRightLogical(pixels, 16), maskByte);

                        Vector256<float> blueF = Avx.ConvertToVector256Single(blue);
                        Vector256<float> greenF = Avx.ConvertToVector256Single(green);
                        Vector256<float> redF = Avx.ConvertToVector256Single(red);

                        Vector256<float> grayFloat = Avx.Add(Avx.Add(Avx.Multiply(redF, rWeight), Avx.Multiply(greenF, gWeight)), Avx.Multiply(blueF, bWeight));
                        grayFloat = Avx.Add(grayFloat, rounding);

                        Vector256<int> grayInt = Avx.ConvertToVector256Int32(grayFloat);
                        grayInt = Avx2.Max(grayInt, zero);
                        grayInt = Avx2.Min(grayInt, max);

                        Vector256<int> grayShift8 = Avx2.ShiftLeftLogical(grayInt, 8);
                        Vector256<int> grayShift16 = Avx2.ShiftLeftLogical(grayInt, 16);

                        Vector256<int> grayPacked = Avx2.Or(grayInt, grayShift8);
                        grayPacked = Avx2.Or(grayPacked, grayShift16);
                        grayPacked = Avx2.Or(grayPacked, alphaMask);

                        Avx.Store((int*)(dstRow + byteOffset), grayPacked);
                    }

                    for (; x < width; x++)
                    {
                        int pixelIndex = x * 4;
                        byte b = srcRow[pixelIndex + 0];
                        byte g = srcRow[pixelIndex + 1];
                        byte r = srcRow[pixelIndex + 2];

                        var gray = (byte)Math.Clamp((int)(r * rWeightValue + g * gWeightValue + b * bWeightValue + 0.5f), 0, 255);

                        dstRow[pixelIndex + 0] = gray;
                        dstRow[pixelIndex + 1] = gray;
                        dstRow[pixelIndex + 2] = gray;
                        dstRow[pixelIndex + 3] = 255;
                    }
                }
            }
        }
    }
}
