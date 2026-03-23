using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using BitNetSharp.Core;

namespace BitNetSharp.Core.Training;

public sealed class BitNetDataLoader
{
    private readonly BitNetTokenizer _tokenizer;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly int _beginTokenId;
    private readonly int _endTokenId;
    private readonly int _unknownTokenId;

    public BitNetDataLoader(IReadOnlyList<string> vocabulary, BitNetDataLoaderOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);

        Options = options ?? new BitNetDataLoaderOptions();
        _tokenizer = new BitNetTokenizer(vocabulary);
        _tokenToId = CreateTokenLookup(vocabulary);
        _beginTokenId = _tokenToId[BitNetTokenizer.BeginToken];
        _endTokenId = _tokenToId[BitNetTokenizer.EndToken];
        _unknownTokenId = _tokenToId[BitNetTokenizer.UnknownToken];
    }

    public BitNetDataLoader(BitNetOptions options, BitNetDataLoaderOptions? dataLoaderOptions = null)
        : this(options.Vocabulary, dataLoaderOptions)
    {
        ArgumentNullException.ThrowIfNull(options);
    }

    public BitNetDataLoaderOptions Options { get; }

    public BitNetTokenizer Tokenizer => _tokenizer;

    public IReadOnlyDictionary<BitNetDataSplit, IReadOnlyList<BitNetTokenSequence>> Load(IEnumerable<TrainingExample> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var orderedExamples = examples.ToList();
        if (orderedExamples.Count == 0)
        {
            throw new ArgumentException("At least one training example is required.", nameof(examples));
        }

        var splitExamples = orderedExamples
            .Select((example, index) => new
            {
                Example = example,
                Index = index,
                Split = SelectSplit(example, index)
            })
            .GroupBy(item => item.Split)
            .ToDictionary(
                group => group.Key,
                group => PackSplit(group.Select(item => item.Example), group.Key),
                EqualityComparer<BitNetDataSplit>.Default);

        foreach (BitNetDataSplit split in Enum.GetValues(typeof(BitNetDataSplit)))
        {
            if (!splitExamples.ContainsKey(split))
            {
                splitExamples[split] = [];
            }
        }

        return splitExamples;
    }

    public IReadOnlyList<TrainingBatch> CreateBatches(IEnumerable<TrainingExample> examples, BitNetDataSplit split = BitNetDataSplit.Training)
    {
        var sequences = Load(examples)[split];
        if (sequences.Count == 0)
        {
            return [];
        }

        var orderedSequences = Options.Shuffle
            ? Shuffle(sequences, Options.Seed + (int)split)
            : sequences;

        var batches = new List<TrainingBatch>();
        for (var index = 0; index < orderedSequences.Count; index += Options.BatchSize)
        {
            var batchSize = Math.Min(Options.BatchSize, orderedSequences.Count - index);
            if (batchSize < Options.BatchSize && Options.DropLast && index > 0)
            {
                break;
            }

            batches.Add(new TrainingBatch(
                split,
                orderedSequences.Skip(index).Take(batchSize).ToArray(),
                batches.Count));
        }

        return batches;
    }

    public IReadOnlyList<BitNetTokenSequence> LoadSplit(IEnumerable<TrainingExample> examples, BitNetDataSplit split = BitNetDataSplit.Training)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var splitExamples = examples
            .Select((example, index) => new { Example = example, Index = index, Split = SelectSplit(example, index) })
            .Where(item => item.Split == split)
            .Select(item => item.Example);

        return PackSplit(splitExamples, split);
    }

    private IReadOnlyList<BitNetTokenSequence> PackSplit(
        IEnumerable<TrainingExample> examples,
        BitNetDataSplit split)
    {
        var rawStream = new List<int>();
        foreach (var example in examples)
        {
            var rawTokens = EncodeExample(example);
            if (rawTokens.Count == 0)
            {
                continue;
            }

            rawStream.AddRange(rawTokens);
        }

        if (rawStream.Count < 2)
        {
            return [];
        }

        var sequences = new List<BitNetTokenSequence>();
        var windowSize = Options.RawSequenceLength;
        var sourceLabelPrefix = $"{split.ToString().ToLowerInvariant()}-window";
        for (var offset = 0; offset + windowSize <= rawStream.Count; offset += windowSize)
        {
            var window = rawStream.GetRange(offset, windowSize);
            sequences.Add(CreateWindow(split, window, $"{sourceLabelPrefix}-{sequences.Count}"));
        }

        if (!Options.DropLast)
        {
            var remainder = rawStream.Count % windowSize;
            if (remainder > 1)
            {
                var tail = rawStream.Skip(rawStream.Count - remainder).ToArray();
                sequences.Add(CreateWindow(split, tail, $"{sourceLabelPrefix}-tail"));
            }
        }

        if (Options.Shuffle && sequences.Count > 1)
        {
            sequences = Shuffle(sequences, Options.Seed + (int)split).ToList();
        }

        return sequences;
    }

    private BitNetDataSplit SelectSplit(TrainingExample example, int index)
    {
        var bucket = ComputeBucket(example, index);
        if (bucket < Options.ValidationFraction)
        {
            return BitNetDataSplit.Validation;
        }

        if (bucket < Options.ValidationFraction + Options.TestFraction)
        {
            return BitNetDataSplit.Test;
        }

        return BitNetDataSplit.Training;
    }

    private double ComputeBucket(TrainingExample example, int index)
    {
        var input = $"{Options.Seed}\u0000{index}\u0000{example.Prompt}\u0000{example.Response}";
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(input));
        var value = BinaryPrimitives.ReadUInt64BigEndian(hash);
        return value / (double)ulong.MaxValue;
    }

    private IReadOnlyList<int> EncodeExample(TrainingExample example)
    {
        var tokens = new List<int> { _beginTokenId };
        tokens.AddRange(_tokenizer.Tokenize(example.Prompt).Select(GetId));
        tokens.AddRange(_tokenizer.Tokenize(example.Response).Select(GetId));
        tokens.Add(_endTokenId);
        return tokens;
    }

    private BitNetTokenSequence CreateWindow(BitNetDataSplit split, IReadOnlyList<int> tokenIds, string source)
    {
        if (tokenIds.Count < 2)
        {
            throw new ArgumentException("Token windows must contain at least two tokens.", nameof(tokenIds));
        }

        return new BitNetTokenSequence(split, tokenIds.ToArray(), source);
    }

    private IReadOnlyList<T> Shuffle<T>(IReadOnlyList<T> items, int seed)
    {
        var shuffled = items.ToArray();
        var random = new Random(seed);
        for (var index = shuffled.Length - 1; index > 0; index--)
        {
            var swapIndex = random.Next(index + 1);
            (shuffled[index], shuffled[swapIndex]) = (shuffled[swapIndex], shuffled[index]);
        }

        return shuffled;
    }

    private int GetId(string token) => _tokenToId.TryGetValue(token, out var id) ? id : _unknownTokenId;

    private static Dictionary<string, int> CreateTokenLookup(IReadOnlyList<string> vocabulary)
    {
        var tokens = new[]
        {
            BitNetTokenizer.BeginToken,
            BitNetTokenizer.EndToken,
            BitNetTokenizer.UnknownToken
        };

        var lookup = new Dictionary<string, int>(StringComparer.Ordinal);
        var index = 0;
        foreach (var token in tokens.Concat(vocabulary.Select(token => token.ToLowerInvariant())))
        {
            if (lookup.ContainsKey(token))
            {
                continue;
            }

            lookup[token] = index++;
        }

        return lookup;
    }
}
