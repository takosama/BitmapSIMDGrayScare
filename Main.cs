using System;
using System.Threading;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;

namespace ConsoleApp67

{
    class Program
    {
        unsafe static void Main(string[] args)
        {
          
            BenchmarkDotNet.Running.BenchmarkRunner.Run<test>();
          
        }
    }

    public class test
    {
        [Benchmark]
        unsafe public static void Run()
        {


            byte[] img = new byte[1920 * 1080 * 4 + 32];
            byte[] canvus = new byte[1920 * 1080 * 4 + 32];

            for (int i = 0; i < 255; i++)
            {

                img[0] = (byte)i;
            }


            fixed (byte* ptr = &img[0])
            fixed (byte* p = &canvus[0])
            {
                //nomal code
                {
                    var pptr = ptr;
                    var pp = p;
                    for (int y = 0; y < 1080; y++)
                    {
                        for (int x = 0; x < 1920; x++)
                        {
                            byte l = (byte)(0.1 * pptr[0] + 0.6 * pptr[1] + 0.2 * pptr[2]);
                            pptr += 4;
                            pp[0] = l;
                            pp[1] = l;
                            pp[2] = l;
                            pp += 4;
                        }
                    }
                }

                //avxcode
                {
                    var pprt = (byte*)(((IntPtr)ptr).ToInt64() + 32 - ((IntPtr)ptr).ToInt64() % 32);
                    var pp = (byte*)(((IntPtr)p).ToInt64() + 32 - ((IntPtr)p).ToInt64() % 32);
                    int h = 1080;
                    int w = 1920;
                    Vector256<float> r = Avx.SetVector256(.333f, .333f, .333f, .333f, .333f, .333f, .333f, .333f);
                    Vector256<float> g = Avx.SetVector256(.666f, .666f, .666f, .666f, .666f, .666f, .666f, .666f);
                    Vector256<float> b = Avx.SetVector256(.112f, .112f, .112f, .112f, .112f, .112f, .112f, .112f);
                    Vector256<byte> mask = Avx.SetVector256(0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0);
                    Vector256<byte> zero = Avx.SetZeroVector256<byte>();
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x += 8)
                        {
                            var tmp0 = Avx.LoadAlignedVector256(pprt);
                            var t0 = Avx2.BlendVariable(tmp0, zero, mask);

                            var tmp1 = Avx2.ShiftRightLogical128BitLane(tmp0, 1);
                            var tmp2 = Avx2.ShiftRightLogical128BitLane(tmp0, 2);

                            var t1 = Avx2.BlendVariable(tmp1, zero, mask);
                            var t2 = Avx2.BlendVariable(tmp2, zero, mask);


                            var tmp6 = Avx.ConvertToVector256Single(Avx.StaticCast<byte, int>(t0));
                            var tmp7 = Avx.ConvertToVector256Single(Avx.StaticCast<byte, int>(t1));
                            var tmp8 = Avx.ConvertToVector256Single(Avx.StaticCast<byte, int>(t2));


                            var tmp13 = Avx.Add(Avx.Add(Avx.Multiply(tmp6, r), Avx.Multiply(tmp7, g)), Avx.Multiply(tmp8, b));

                            var tmp14 = Avx.ConvertToVector256Int32(tmp13);
                            var tmp15 = Avx2.ShiftLeftLogical(tmp14, 8);
                            var tmp16 = Avx2.ShiftLeftLogical(tmp14, 16);

                            var tmp17 = Avx2.Add(tmp14, tmp15);
                            var tmp18 = Avx2.Add(tmp16, tmp17);
                            Avx.StoreAligned(pp, Avx.StaticCast<int, byte>(tmp18));

                            pp += 32;
                            pprt += 32;
                        }
                    }
                }



            }
        
        }
    }
}
