using System;
using BenchmarkDotNet.Running;

namespace ConsoleApp67
{
    class Program
    {
        unsafe static void Main(string[] args)
        {
            BenchmarkRunner.Run<test>();
        }
    }

    public unsafe partial class test : IDisposable
    {
    }
}
