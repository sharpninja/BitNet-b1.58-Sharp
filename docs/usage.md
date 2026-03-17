# Usage

## Build

```bash
dotnet build /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp.slnx
```

## Chat

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- chat "how are you hosted"
```

Optional verbosity:

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- chat "hello" --verbosity=quiet
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- chat "hello" --verbosity=verbose
```

## Host summary

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- host
```

This command confirms that the application is wired for Microsoft Agent Framework hosting and reports the current language and verbosity configuration.
