# Correctness regression tests

Run `dotnet run --project tests/RegressionTests.csproj -c Release` with .NET 8.

The harness calls the actual private converter methods through typed delegates. It checks widths 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33 and 65, saturated colors, rounding, alpha, output guards and scalar/SIMD agreement. The newer benchmark's scalar/AVX2/AVX512/FMA/ARM implementation uses the common integer formula `(77*R + 150*G + 29*B + 128) >> 8`; FMA uses the exactly representable equivalent coefficients divided by 256. The separate legacy Main.cs converter retains its existing 0.299/0.587/0.114 formula, with SIMD conversion truncating after +0.5 just like its scalar version.

Unsupported hardware branches are skipped or exercised only through their scalar fallback, and the program prints the available instruction sets. This test is not a throughput benchmark. Earlier benchmark performance claims have not been remeasured after the correctness fixes.
