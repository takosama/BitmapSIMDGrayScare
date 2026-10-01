using System.Reflection;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using ConsoleApp67;

unsafe class RegressionTests
{
    private delegate void NativeConverter(byte* source, byte* destination, in ImageSpec spec);
    private delegate void ManagedConverter(byte[] source, byte[] destination, int width, int height);
    static int Main()
    {
        int count = 0;
        var methods = new List<string> { "ScalarGrayscale", "AdvSimdGrayscale", "FmaGrayscale" };
        if (Avx2.IsSupported) methods.AddRange(new[] { "Avx2Grayscale", "ParallelAvx2Grayscale" });
        if (Avx512F.IsSupported && Avx512BW.IsSupported && Avx2.IsSupported) methods.Add("Avx512Grayscale");
        foreach (int width in new[] { 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65 })
        {
            var spec = new ImageSpec(width, 3, $"{width}x3");
            var source = new byte[spec.BufferLength + 64];
            new Random(width).NextBytes(source);
            // Saturated primaries and low values exercise every coefficient and rounding.
            byte[] cases = { 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 0,
                255, 0, 0, 0, 0, 0, 255, 0, 1, 1, 1, 0 };
            cases.AsSpan(0, Math.Min(cases.Length, spec.BufferLength)).CopyTo(source.AsSpan(32));
            var expected = new byte[spec.BufferLength];
            for (int i = 0; i < expected.Length; i += 4)
            {
                int j = i + 32;
                byte gray = (byte)((29 * source[j] + 150 * source[j + 1] + 77 * source[j + 2] + 128) >> 8);
                expected[i] = expected[i + 1] = expected[i + 2] = gray;
                expected[i + 3] = 255;
            }
            foreach (string method in methods)
            {
                var destination = Enumerable.Repeat((byte)0xCC, source.Length).ToArray();
                var converter = typeof(test).GetMethod(method, BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate<NativeConverter>();
                fixed (byte* src = source, dst = destination) converter(src + 32, dst + 32, spec);
                if (!destination.AsSpan(32, expected.Length).SequenceEqual(expected))
                    throw new Exception($"{method} differs from integer scalar at width {width}");
                if (destination.AsSpan(0, 32).ContainsAnyExcept((byte)0xCC) || destination.AsSpan(32 + expected.Length).ContainsAnyExcept((byte)0xCC))
                    throw new Exception($"{method} wrote outside output at width {width}");
                count++;
            }
        }
        if (Avx2.IsSupported)
        {
            var type = typeof(BitmapSIMDGrayScare.GrayscaleBenchmark);
            var scalar = type.GetMethod("ScalarGrayscale", BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate<ManagedConverter>();
            var vector = type.GetMethod("Avx2Grayscale", BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate<ManagedConverter>();
            foreach (int width in new[] { 1, 7, 8, 9, 16, 17, 65 })
            {
                byte[] source = new byte[width * 3 * 4];
                new Random(width + 100).NextBytes(source);
                source[0] = source[1] = source[2] = 1;
                var expected = new byte[source.Length]; var actual = new byte[source.Length];
                scalar(source, expected, width, 3); vector(source, actual, width, 3);
                if (!actual.SequenceEqual(expected)) throw new Exception($"Main SIMD mismatch width {width}");
                count++;
            }
        }
        Console.WriteLine($"PASS {count} grayscale cases. AVX2={Avx2.IsSupported}; AVX512={Avx512F.IsSupported && Avx512BW.IsSupported}; FMA={Fma.IsSupported}; ARM64={AdvSimd.Arm64.IsSupported}");
        Console.WriteLine("Unsupported hardware paths were skipped or tested only through their scalar fallback. No performance benchmark was run.");
        return 0;
    }
}
