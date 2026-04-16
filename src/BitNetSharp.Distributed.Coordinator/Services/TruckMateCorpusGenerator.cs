using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// Generates synthetic training examples for the Truck Mate voice
/// assistant intent-classification SLM. Each example is a
/// <c>[USER]</c> utterance (simulating noisy ASR output) paired
/// with a <c>[INTENT]</c> JSON containing the classified intent
/// name and extracted slots.
///
/// <para>
/// The generator covers every intent family from PLAN-AIVOICE-001:
/// trip management, navigation, find-POI, route preferences,
/// compliance/HOS, logistics, and status queries. Variations
/// include CB-radio shorthand, ASR noise, geographic aliases, and
/// multiple phrasings per intent so the SLM learns to generalize.
/// </para>
///
/// <para>
/// Output format per example (one per line in the shard file):
/// <code>
/// [USER] take me to the flying j in dallas [INTENT] {"intent":"navigate","slots":{"destination":"Flying J","city":"Dallas"}}
/// </code>
/// This is a simple text-pair format that a tokenizer can split on
/// the <c>[INTENT]</c> marker to produce (input, target) pairs for
/// next-token prediction training.
/// </para>
/// </summary>
public static class TruckMateCorpusGenerator
{
    private static readonly string[] Cities = {
        "Dallas", "Houston", "Atlanta", "Memphis", "Nashville",
        "Chicago", "Indianapolis", "Louisville", "St. Louis", "Kansas City",
        "Denver", "Phoenix", "Las Vegas", "Salt Lake City", "Boise",
        "Portland", "Seattle", "Sacramento", "Los Angeles", "San Antonio",
        "El Paso", "Albuquerque", "Oklahoma City", "Little Rock", "Jackson",
        "Birmingham", "Charlotte", "Richmond", "Baltimore", "Philadelphia",
        "Columbus", "Cincinnati", "Detroit", "Milwaukee", "Minneapolis",
        "Omaha", "Des Moines", "Tulsa", "Shreveport", "Jacksonville",
        "Tampa", "Miami", "Savannah", "Charleston", "Raleigh",
        "Pittsburgh", "Buffalo", "Hartford", "Providence", "Boston"
    };

    private static readonly string[] TruckStopChains = {
        "Flying J", "Pilot", "Love's", "TA Travel Centers", "Petro",
        "Buc-ee's", "QuikTrip", "Sapp Bros", "Kwik Trip", "Casey's"
    };

    private static readonly string[] Interstates = {
        "I-10", "I-20", "I-30", "I-35", "I-40", "I-44", "I-55",
        "I-65", "I-70", "I-75", "I-80", "I-81", "I-85", "I-90",
        "I-94", "I-95", "I-5", "I-15", "I-25", "I-45"
    };

    private static readonly string[] StopTypes = {
        "truck stop", "fuel", "diesel", "parking", "rest area",
        "weigh station", "scale", "repair shop", "tire shop",
        "restaurant", "shower", "laundry"
    };

    private static readonly string[] RoutePrefs = {
        "avoid tolls", "no tolls", "skip the toll road",
        "avoid hazmat restrictions", "stay off restricted routes",
        "avoid low clearances", "watch for low bridges",
        "avoid steep grades", "no mountain passes",
        "shortest route", "fastest route", "most fuel efficient"
    };

    private static readonly string[] AsrNoise = {
        "uh ", "um ", "like ", "you know ", "", "", "", "", ""
    };

    /// <summary>
    /// Generates <paramref name="count"/> synthetic training examples
    /// and writes them to <paramref name="outputDirectory"/> as
    /// sharded text files. Returns a manifest listing every shard
    /// with its example count and byte size.
    /// </summary>
    public static CorpusManifest Generate(
        string outputDirectory,
        int count = 50_000,
        int examplesPerShard = 5_000)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (examplesPerShard <= 0) throw new ArgumentOutOfRangeException(nameof(examplesPerShard));

        Directory.CreateDirectory(outputDirectory);

        var rng = new Random(42);
        var shards = new List<CorpusShardInfo>();
        var shardIndex = 0;
        var remaining = count;

        while (remaining > 0)
        {
            var batchSize = Math.Min(remaining, examplesPerShard);
            var shardId = $"truckmate-shard-{shardIndex:D4}";
            var shardPath = Path.Combine(outputDirectory, $"{shardId}.txt");

            using (var writer = new StreamWriter(shardPath, false, Encoding.UTF8))
            {
                for (var i = 0; i < batchSize; i++)
                {
                    var example = GenerateExample(rng);
                    writer.WriteLine(example);
                }
            }

            var fileInfo = new FileInfo(shardPath);
            shards.Add(new CorpusShardInfo(shardId, shardPath, batchSize, fileInfo.Length));
            shardIndex++;
            remaining -= batchSize;
        }

        var manifest = new CorpusManifest(
            Name: "truckmate-v1",
            TotalExamples: count,
            Shards: shards);

        var manifestPath = Path.Combine(outputDirectory, "manifest.json");
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(manifestPath, json, Encoding.UTF8);

        return manifest;
    }

    private static string GenerateExample(Random rng)
    {
        var intentFamily = rng.Next(10);
        return intentFamily switch
        {
            0 => GenerateTripCommand(rng),
            1 => GenerateNavigateCommand(rng),
            2 => GenerateFindPoiCommand(rng),
            3 => GenerateRoutePreferenceCommand(rng),
            4 => GenerateHosCommand(rng),
            5 => GenerateAddTodoCommand(rng),
            6 => GenerateAddExpenseCommand(rng),
            7 => GenerateStatusCommand(rng),
            8 => GenerateRerouteCommand(rng),
            _ => GenerateLoadCommand(rng)
        };
    }

    private static string GenerateTripCommand(Random rng)
    {
        var start = rng.Next(2) == 0;
        var city = Pick(rng, Cities);
        if (start)
        {
            var utterance = Pick(rng, new[] {
                $"{Noise(rng)}start my trip to {city}",
                $"{Noise(rng)}begin trip {city}",
                $"starting trip headed to {city}",
                $"new trip to {city} {Noise(rng)}please",
                $"kick off the trip to {city}"
            });
            return Format(utterance, "start_trip", $"\"destination\":\"{city}\"");
        }
        else
        {
            var utterance = Pick(rng, new[] {
                $"{Noise(rng)}stop trip",
                $"end my trip",
                $"trip complete",
                $"close out this trip",
                $"i'm done driving {Noise(rng)}stop the trip"
            });
            return Format(utterance, "stop_trip", "");
        }
    }

    private static string GenerateNavigateCommand(Random rng)
    {
        var city = Pick(rng, Cities);
        var interstate = Pick(rng, Interstates);
        var variant = rng.Next(6);
        var utterance = variant switch
        {
            0 => $"{Noise(rng)}take me to {city}",
            1 => $"navigate to {city}",
            2 => $"directions to {city} {Noise(rng)}please",
            3 => $"head towards {city} on {interstate}",
            4 => $"route me to the pickup in {city}",
            _ => $"get me to {city} for delivery"
        };
        var slots = $"\"destination\":\"{city}\"";
        if (variant == 3) slots += $",\"via\":\"{interstate}\"";
        if (variant == 4) slots += ",\"purpose\":\"pickup\"";
        if (variant == 5) slots += ",\"purpose\":\"delivery\"";
        return Format(utterance, "navigate", slots);
    }

    private static string GenerateFindPoiCommand(Random rng)
    {
        var stopType = Pick(rng, StopTypes);
        var chain = Pick(rng, TruckStopChains);
        var city = Pick(rng, Cities);
        var variant = rng.Next(5);
        var utterance = variant switch
        {
            0 => $"find me a {stopType} near {city}",
            1 => $"where's the nearest {stopType}",
            2 => $"{Noise(rng)}i need {stopType}",
            3 => $"find a {chain} near here",
            _ => $"closest {stopType} off {Pick(rng, Interstates)}"
        };
        var slots = $"\"stop_type\":\"{stopType}\"";
        if (variant == 0) slots += $",\"near\":\"{city}\"";
        if (variant == 3) slots = $"\"chain\":\"{chain}\"";
        if (variant == 4) slots += $",\"near_interstate\":\"{Pick(rng, Interstates)}\"";
        return Format(utterance, "find_poi", slots);
    }

    private static string GenerateRoutePreferenceCommand(Random rng)
    {
        var pref = Pick(rng, RoutePrefs);
        var utterance = Pick(rng, new[] {
            pref,
            $"{Noise(rng)}{pref} please",
            $"i want to {pref}",
            $"set route to {pref}",
            $"change preference to {pref}"
        });
        return Format(utterance, "route_preference", $"\"preference\":\"{pref}\"");
    }

    private static string GenerateHosCommand(Random rng)
    {
        var variant = rng.Next(6);
        var utterance = variant switch
        {
            0 => $"how much time do i got left on my clock",
            1 => $"{Noise(rng)}check my hours",
            2 => $"when do i need to stop for my break",
            3 => $"what's my hours of service status",
            4 => $"am i good to keep driving",
            _ => $"how many hours till my thirty minute break"
        };
        var intent = variant <= 1 ? "hos_status" : variant <= 3 ? "hos_break_check" : "hos_drive_remaining";
        return Format(utterance, intent, "");
    }

    private static string GenerateAddTodoCommand(Random rng)
    {
        var tasks = new[] {
            "check tire pressure", "call dispatch", "fuel up before Memphis",
            "pick up load in Nashville", "get oil change", "weigh at next scale",
            "grab food at next stop", "check refrigeration unit", "submit paperwork"
        };
        var task = Pick(rng, tasks);
        var utterance = Pick(rng, new[] {
            $"add todo {task}",
            $"remind me to {task}",
            $"{Noise(rng)}new task {task}",
            $"put {task} on my list"
        });
        return Format(utterance, "add_todo", $"\"task\":\"{task}\"");
    }

    private static string GenerateAddExpenseCommand(Random rng)
    {
        var amounts = new[] { "45.50", "89.99", "120.00", "32.75", "67.40" };
        var categories = new[] { "fuel", "food", "tolls", "maintenance", "parking", "supplies" };
        var amount = Pick(rng, amounts);
        var category = Pick(rng, categories);
        var utterance = Pick(rng, new[] {
            $"add expense {amount} dollars for {category}",
            $"log {category} expense {amount}",
            $"{Noise(rng)}spent {amount} on {category}"
        });
        return Format(utterance, "add_expense", $"\"amount\":\"{amount}\",\"category\":\"{category}\"");
    }

    private static string GenerateStatusCommand(Random rng)
    {
        var variant = rng.Next(5);
        var utterance = variant switch
        {
            0 => $"what's my eta",
            1 => $"how far to {Pick(rng, Cities)}",
            2 => $"{Noise(rng)}trip status",
            3 => $"what's my next stop",
            _ => $"show me the route summary"
        };
        var intent = variant <= 1 ? "eta_query" : variant == 3 ? "next_stop_query" : "trip_status";
        var slots = variant == 1 ? $"\"destination\":\"{Pick(rng, Cities)}\"" : "";
        return Format(utterance, intent, slots);
    }

    private static string GenerateRerouteCommand(Random rng)
    {
        var reasons = new[] {
            "traffic", "construction", "weather", "accident",
            "road closure", "detour", "mountain pass closed"
        };
        var reason = Pick(rng, reasons);
        var utterance = Pick(rng, new[] {
            $"reroute around {reason}",
            $"{Noise(rng)}avoid {reason} ahead",
            $"find alternate route {reason}",
            $"there's {reason} on {Pick(rng, Interstates)} reroute me"
        });
        return Format(utterance, "reroute", $"\"reason\":\"{reason}\"");
    }

    private static string GenerateLoadCommand(Random rng)
    {
        var actions = new[] { "picked up", "delivered", "delayed", "weighed" };
        var action = Pick(rng, actions);
        var loadId = $"LD-{rng.Next(10000, 99999).ToString(CultureInfo.InvariantCulture)}";
        var utterance = Pick(rng, new[] {
            $"mark load {loadId} as {action}",
            $"load {loadId} {action}",
            $"{Noise(rng)}update load {loadId} to {action}"
        });
        return Format(utterance, "update_load", $"\"load_id\":\"{loadId}\",\"action\":\"{action}\"");
    }

    private static string Format(string utterance, string intent, string slots)
    {
        var slotsJson = string.IsNullOrWhiteSpace(slots) ? "{}" : "{" + slots + "}";
        return $"[USER] {utterance.Trim()} [INTENT] {{\"intent\":\"{intent}\",\"slots\":{slotsJson}}}";
    }

    private static string Noise(Random rng) => Pick(rng, AsrNoise);

    private static T Pick<T>(Random rng, T[] array) => array[rng.Next(array.Length)];
}

/// <summary>Manifest for a generated corpus, written as manifest.json.</summary>
public sealed record CorpusManifest(
    string Name,
    int TotalExamples,
    IReadOnlyList<CorpusShardInfo> Shards);

/// <summary>Metadata for a single shard file in the corpus.</summary>
public sealed record CorpusShardInfo(
    string ShardId,
    string Path,
    int ExampleCount,
    long SizeBytes);
