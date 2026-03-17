# Training and visualization

## Train the default corpus

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- train
```

The trainer fits ternary transition weights from the built-in American English prompt/response pairs and reports the average loss by epoch.

## Visualize results

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- visualize
```

The visualization output includes:

- An ASCII loss chart
- A ternary weight histogram
- CSV-formatted epoch data that can be pasted into external tools

## Extending training data

Use `BitNetTrainer.Train` with your own `TrainingExample` instances if you want to replace or extend the built-in corpus. The default implementation is intentionally small so the repository remains easy to inspect and modify.
