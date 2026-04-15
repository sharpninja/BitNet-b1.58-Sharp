using System.Runtime.CompilerServices;

// Allow the shared BitNetSharp test project to exercise the internal types
// that make up the worker's startup calibration and task-sizing math.
[assembly: InternalsVisibleTo("BitNetSharp.Tests")]
