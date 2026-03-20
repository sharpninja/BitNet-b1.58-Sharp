# BitNet b1.58 Sharp

BitNet b1.58 Sharp is a .NET 10 C# reference implementation of the paper-aligned BitNet b1.58 decoder-only transformer architecture with ternary `-1/0/+1` weights, BitLinear projections, RoPE, RMSNorm, SwiGLU, and Microsoft Agent Framework-oriented hosting.

## What is included

- A paper-aligned BitNet core model in `/src/BitNetSharp.Core`
- A decoder-only transformer implementation with `BitLinear`, `RmsNorm`, RoPE, causal attention, SwiGLU, and `BitNetTransformer`
- Microsoft Agent Framework-oriented hosting in `/src/BitNetSharp.App`
- BenchmarkDotNet-based local model comparison in `/src/BitNetSharp.App`
- DataGen synthetic dataset generation from JSON seed examples
- Default American English interaction behavior
- Seeded transformer inspection and ternary weight summaries
- GitBook-formatted project documentation in `/docs`

## Quick start

```bash
dotnet build BitNet-b1.58-Sharp.slnx
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- chat "hello"
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- datagen --domain "customer-support" --count 10 --seeds examples/seed-examples.json --output data/customer-support.jsonl
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- visualize
dotnet test BitNet-b1.58-Sharp.slnx
```

## Documentation map

- [Architecture](architecture.md)
- [Benchmarking and model comparison](benchmarking.md)
- [DataGen guide](datagen-guide.md)
- [Implementation plan](implementation-plan-v3.md)
- [Releases and packaging](releases-and-packaging.md)
- [Usage](usage.md)
- [Training and visualization](training-and-visualization.md)
