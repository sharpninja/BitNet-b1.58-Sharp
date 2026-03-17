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

A `TrainingExample` represents a single prompt/response pair for the LLM to learn from. You can construct these inline, read them from a JSON file, or pull them from a database.

```csharp
using System.Collections.Generic;
using BitNetSharp.Core; // Replace with actual namespaces
using BitNetSharp.Training;

// Define a custom dataset
var customCorpus = new List<TrainingExample>
{
    new TrainingExample 
    { 
        Prompt = "What is the capital of France?", 
        Response = "Paris" 
    },
    new TrainingExample 
    { 
        Prompt = "Explain BitNet b1.58.", 
        Response = "BitNet b1.58 is a 1-bit LLM architecture where weights are ternary: -1, 0, or 1." 
    },
    new TrainingExample 
    { 
        Prompt = "Write a C# console greeting.", 
        Response = "Console.WriteLine(\"Hello, World!\");" 
    }
};
```

### 2. Configuring the Trainer

Before starting the training process, instantiate the `BitNetTrainer`. Depending on the API, you may want to pass hyperparameters such as the learning rate, number of epochs, and batch size.

```csharp
// Example configuration for the trainer
var trainerOptions = new TrainerOptions
{
    MaxEpochs = 100,
    LearningRate = 1e-3f,
    BatchSize = 8,
    // Determines how often to report loss metrics
    LogInterval = 10 
};

var trainer = new BitNetTrainer(trainerOptions);
```

### 3. Running the Training Loop

Pass your dataset into the `Train` method. The trainer will apply the b1.58 quantization strategy during the forward/backward passes, ensuring the final weights adhere to the ternary constraints.

```csharp
Console.WriteLine("Starting training...");

// Execute training. You can optionally hook into callbacks to report progress.
var trainedModel = trainer.Train(customCorpus, (epoch, metrics) =>
{
    Console.WriteLine($"[Epoch {epoch}] Loss: {metrics.AverageLoss:F4}");
});

Console.WriteLine("Training completed successfully!");
```

### 4. Saving and Loading the Model

Once your model has learned from the custom corpus, you should save the quantized weights to disk so you can load them later for inference without retraining.

```csharp
// Save the trained ternary weights
string modelOutputPath = "models/custom-bitnet-v1.bin";
trainedModel.Save(modelOutputPath);
Console.WriteLine($"Model saved to {modelOutputPath}");

// Later, load the model for inference
var loadedModel = BitNetModel.Load(modelOutputPath);
var response = loadedModel.Generate("What is the capital of France?");
Console.WriteLine($"Model Output: {response}");
```

### Tips for Training

* **Data Formatting:** Ensure your prompts and responses are cleaned and tokenized using the same tokenizer the model expects during inference.
* **Epochs:** Because weights are heavily quantized, training dynamics differ from standard FP16 LLMs. You may need to experiment with the learning rate and epochs to ensure convergence without catastrophic forgetting.
* **Batching:** Group examples of similar token lengths together to optimize processing time if padding is required.
