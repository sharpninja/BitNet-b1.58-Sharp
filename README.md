# BitNet-b1.58-Sharp

Project documentation now lives in GitBook format under `/docs`.

- Start here: [`/docs/README.md`](docs/README.md)
- Navigation: [`/docs/SUMMARY.md`](docs/SUMMARY.md)

## Windows development focus

This repository is optimized for Windows development with Visual Studio 2022/2025, .NET 9/10, and PowerShell.

Use the `dotnet` CLI from the repository root for the standard validation flow:

```powershell
dotnet build BitNet-b1.58-Sharp.slnx
dotnet test BitNet-b1.58-Sharp.slnx
```

When documentation needs concrete local paths, prefer Windows-style examples such as `C:\src\BitNet-b1.58-Sharp`.
