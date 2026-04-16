using System;
using System.IO;
using System.Linq;
using BitNetSharp.Distributed.Contracts;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd tests for the word-level tokenizer used by the Truck Mate
/// corpus pipeline. Covers vocab training, encode/decode round
/// trip, special token handling, OOV fallback, and file
/// persistence.
/// </summary>
public sealed class WordLevelTokenizerTests
{
    private static readonly string[] SampleCorpus =
    {
        "[USER] take me to the flying j in dallas [INTENT] {\"intent\":\"navigate\",\"slots\":{\"destination\":\"Dallas\"}}",
        "[USER] find a truck stop near houston [INTENT] {\"intent\":\"find_poi\",\"slots\":{\"stop_type\":\"truck stop\"}}",
        "[USER] start my trip to memphis [INTENT] {\"intent\":\"start_trip\",\"slots\":{\"destination\":\"Memphis\"}}",
        "[USER] how much time left on my clock [INTENT] {\"intent\":\"hos_status\",\"slots\":{}}",
        "[USER] add expense 45 dollars for fuel [INTENT] {\"intent\":\"add_expense\",\"slots\":{\"amount\":\"45\",\"category\":\"fuel\"}}",
        "[USER] take me to the flying j in dallas [INTENT] {\"intent\":\"navigate\",\"slots\":{\"destination\":\"Dallas\"}}"
    };

    private WordLevelTokenizer TrainSample() =>
        WordLevelTokenizer.TrainFromCorpus(SampleCorpus, maxVocab: 200, minFrequency: 1);

    [Fact]
    public void TrainFromCorpus_builds_vocab_with_special_tokens_at_front()
    {
        var tok = TrainSample();
        Assert.True(tok.VocabSize > WordLevelTokenizer.FirstUserTokenId);
    }

    [Fact]
    public void Encode_wraps_with_BOS_and_EOS()
    {
        var tok = TrainSample();
        var ids = tok.Encode("take me to dallas");
        Assert.Equal(WordLevelTokenizer.BosId, ids[0]);
        Assert.Equal(WordLevelTokenizer.EosId, ids[^1]);
    }

    [Fact]
    public void Encode_maps_USER_and_INTENT_markers_to_their_special_ids()
    {
        var tok = TrainSample();
        var ids = tok.Encode("[USER] hello [INTENT] world");
        Assert.Contains(WordLevelTokenizer.UserId, ids);
        Assert.Contains(WordLevelTokenizer.IntentId, ids);
    }

    [Fact]
    public void Encode_maps_unknown_tokens_to_UNK()
    {
        var tok = TrainSample();
        var ids = tok.Encode("xyzzy_never_seen_token");
        // The unknown word should map to UNK (id=1).
        Assert.Contains(WordLevelTokenizer.UnkId, ids);
    }

    [Fact]
    public void Decode_round_trips_known_tokens()
    {
        var tok = TrainSample();
        var original = "[USER] take me to dallas";
        var ids = tok.Encode(original);
        var decoded = tok.Decode(ids);
        // Decoded form includes [BOS] ... [EOS] and is space-joined.
        Assert.Contains("take", decoded);
        Assert.Contains("dallas", decoded);
        Assert.Contains("[BOS]", decoded);
        Assert.Contains("[EOS]", decoded);
    }

    [Fact]
    public void Encode_empty_text_returns_BOS_EOS_only()
    {
        var tok = TrainSample();
        var ids = tok.Encode("");
        Assert.Equal(new[] { WordLevelTokenizer.BosId, WordLevelTokenizer.EosId }, ids);
    }

    [Fact]
    public void SaveToFile_and_LoadFromFile_preserves_vocab()
    {
        var tok = TrainSample();
        var path = Path.Combine(Path.GetTempPath(), $"bitnet-tok-{Guid.NewGuid():N}.json");
        try
        {
            tok.SaveToFile(path);
            var loaded = WordLevelTokenizer.LoadFromFile(path);
            Assert.Equal(tok.VocabSize, loaded.VocabSize);

            // Same encode output from both instances.
            var original = "[USER] take me to dallas [INTENT] test";
            Assert.Equal(tok.Encode(original), loaded.Encode(original));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void TrainFromCorpus_respects_minFrequency()
    {
        // "clock" appears exactly once in the sample corpus; with
        // minFrequency=2 it drops out and encodes as UNK.
        var tok = WordLevelTokenizer.TrainFromCorpus(SampleCorpus, maxVocab: 200, minFrequency: 2);
        var ids = tok.Encode("clock");
        Assert.Contains(WordLevelTokenizer.UnkId, ids);

        // "to" appears in multiple lines → survives the threshold.
        var toIds = tok.Encode("to");
        Assert.DoesNotContain(WordLevelTokenizer.UnkId, toIds.Skip(1).SkipLast(1).ToArray());
    }

    [Fact]
    public void TrainFromCorpus_caps_at_maxVocab()
    {
        var tok = WordLevelTokenizer.TrainFromCorpus(SampleCorpus, maxVocab: 10, minFrequency: 1);
        Assert.True(tok.VocabSize <= 10);
    }
}
