# Training and Visualization

## Train the default corpus

To train the model using the built-in dataset, run the following command from your terminal:

```bash
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- train
```
*(Note: Using relative paths is recommended so the command works seamlessly across different development environments).*

The trainer fits ternary transition weights (-1, 0, 1) from the built-in American English prompt/response pairs and reports the average loss by epoch.

## Visualize results

To generate diagnostic visualizations of the training process and the resulting weights:

```bash
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- visualize
```

The visualization output includes:

- An ASCII loss chart tracking model convergence.
- A ternary weight histogram showing the distribution of `-1`, `0`, and `1` weights.
- CSV-formatted epoch data that can be exported and pasted into external tools (like Excel or Python notebooks) for further analysis.

---

## Extending Training Data and Custom Training

The default implementation is intentionally small so the repository remains easy to clone and test. If you want to fine-tune the model on a specific domain or replace the built-in corpus entirely, you can use `BitNetTrainer.Train` with your own `TrainingExample` instances.

### 1. Creating Custom Training Examples

`TrainingExample` is a positional record with two required constructor parameters: `Prompt` and `Response`. You can construct instances inline, read them from a JSON file, or pull them from a database.

```csharp
using System.Collections.Generic;
using BitNetSharp.Core;

// Define a custom dataset using the positional record constructor
var customCorpus = new List<TrainingExample>
{
    new TrainingExample("What is the capital of France?", "Paris"),
    new TrainingExample(
        "Explain BitNet b1.58.",
        "BitNet b1.58 is a 1-bit LLM architecture where weights are ternary: -1, 0, or 1."),
    new TrainingExample(
        "Write a C# console greeting.",
        "Console.WriteLine(\"Hello, World!\");")
};
```

### 2. Instantiating the Trainer

`BitNetTrainer` is constructed with a `BitNetModel`. Create a model first, then pass it to the trainer.

```csharp
using BitNetSharp.Core;

// Create a model with the default vocabulary
var model = BitNetModel.CreateDefault();

// Wrap it in a trainer
var trainer = new BitNetTrainer(model);
```

### 3. Running the Training Loop

Pass your dataset and the number of epochs into `Train`. It returns a `TrainingReport` that summarises loss history and the ternary weight distribution.

```csharp
Console.WriteLine("Starting training...");

// Run 10 training epochs over the custom corpus
TrainingReport report = trainer.Train(customCorpus, epochs: 10);

Console.WriteLine($"Training completed. Average loss: {report.AverageLoss:F4}");
Console.WriteLine($"Epochs: {report.Epochs}, Samples seen: {report.SamplesSeen}");
Console.WriteLine($"Weights — negative: {report.NegativeWeights}, " +
                  $"zero: {report.ZeroWeights}, positive: {report.PositiveWeights}");
```

### 4. Running Inference

After training, call `GenerateResponse` on the model to produce a response. The method returns a `BitNetGenerationResult` whose `ResponseText` property contains the decoded output.

```csharp
BitNetGenerationResult result = model.GenerateResponse("What is the capital of France?");
Console.WriteLine($"Model output: {result.ResponseText}");
```

> **Note:** Model persistence (save/load) is not yet implemented. To reuse a trained model across sessions, re-run training at startup or extend `BitNetModel` with your own serialization logic.

### Tips for Training

* **Data Formatting:** Ensure your prompts and responses are cleaned and tokenized using the same tokenizer the model expects during inference.
* **Epochs:** Because weights are heavily quantized, training dynamics differ from standard FP16 LLMs. You may need to experiment with the number of epochs to ensure convergence without catastrophic forgetting.
* **Batching:** Group examples of similar token lengths together to optimize processing time if padding is required.
