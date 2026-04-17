using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// Pool version for the Truck Mate synthetic corpus. V1 is the frozen
/// pool set shipped with <c>truckmate-v1</c> (seed=42 → 50,000
/// examples). V2 is a superset for <c>truckmate-v2</c> (seed=42 →
/// 200,000 examples) with ~2× US freight-hub cities, new weather and
/// time-of-day pools, a multi-stop <c>start_trip</c> variant, and a
/// weather-alert <c>reroute</c> variant. V1 byte parity is preserved:
/// when <see cref="CorpusPoolVersion.V1"/> is selected the generator
/// takes exactly the same code paths it always did.
/// </summary>
public enum CorpusPoolVersion { V1, V2 }

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
    /// <summary>
    /// Generates <paramref name="count"/> synthetic training examples
    /// and writes them to <paramref name="outputDirectory"/> as
    /// sharded text files. Returns a manifest listing every shard
    /// with its example count and byte size.
    ///
    /// <para>
    /// Defaults (<c>seed=42</c>, <c>poolVersion=V1</c>,
    /// <c>manifestName="truckmate-v1"</c>) reproduce the original
    /// <c>truckmate-v1</c> 50K corpus byte-for-byte. For v2 pass
    /// <c>count=200000</c>, <c>poolVersion=V2</c>, and
    /// <c>manifestName="truckmate-v2"</c>.
    /// </para>
    /// </summary>
    public static CorpusManifest Generate(
        string outputDirectory,
        int count = 50_000,
        int examplesPerShard = 5_000,
        int seed = 42,
        CorpusPoolVersion poolVersion = CorpusPoolVersion.V1,
        string manifestName = "truckmate-v1")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestName);
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (examplesPerShard <= 0) throw new ArgumentOutOfRangeException(nameof(examplesPerShard));

        Directory.CreateDirectory(outputDirectory);

        var rng = new Random(seed);
        var shards = new List<CorpusShardInfo>();
        var shardIndex = 0;
        var remaining = count;

        while (remaining > 0)
        {
            var batchSize = Math.Min(remaining, examplesPerShard);
            var shardId = $"{manifestName}-shard-{shardIndex:D4}";
            var shardPath = Path.Combine(outputDirectory, $"{shardId}.txt");

            using (var writer = new StreamWriter(shardPath, false, Encoding.UTF8))
            {
                for (var i = 0; i < batchSize; i++)
                {
                    var example = GenerateExample(rng, poolVersion);
                    writer.WriteLine(example);
                }
            }

            var fileInfo = new FileInfo(shardPath);
            shards.Add(new CorpusShardInfo(shardId, shardPath, batchSize, fileInfo.Length));
            shardIndex++;
            remaining -= batchSize;
        }

        var manifest = new CorpusManifest(
            Name: manifestName,
            TotalExamples: count,
            Seed: seed,
            PoolVersion: poolVersion.ToString(),
            Shards: shards);

        // manifest.{name}.json so v1 and v2 manifests coexist in the
        // same corpus dir without either overwriting the other.
        var manifestPath = Path.Combine(outputDirectory, $"manifest.{manifestName}.json");
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(manifestPath, json, Encoding.UTF8);

        // Legacy symlink: v1 default also writes plain manifest.json
        // so existing tooling that reads "manifest.json" keeps working.
        if (string.Equals(manifestName, "truckmate-v1", StringComparison.OrdinalIgnoreCase))
        {
            File.WriteAllText(Path.Combine(outputDirectory, "manifest.json"), json, Encoding.UTF8);
        }

        return manifest;
    }

    /// <summary>
    /// Emits a single <c>[USER] ... [INTENT] {...}</c> line from the
    /// shared intent-pool distribution. Exposed as <c>internal</c> so
    /// the multi-turn generator can concatenate N of these per example
    /// line without forking the pool definitions or the RNG histogram.
    /// </summary>
    internal static string GenerateExample(Random rng, CorpusPoolVersion v)
    {
        var intentFamily = rng.Next(10);
        return intentFamily switch
        {
            0 => GenerateTripCommand(rng, v),
            1 => GenerateNavigateCommand(rng, v),
            2 => GenerateFindPoiCommand(rng, v),
            3 => GenerateRoutePreferenceCommand(rng, v),
            4 => GenerateHosCommand(rng, v),
            5 => GenerateAddTodoCommand(rng, v),
            6 => GenerateAddExpenseCommand(rng, v),
            7 => GenerateStatusCommand(rng, v),
            8 => GenerateRerouteCommand(rng, v),
            _ => GenerateLoadCommand(rng, v)
        };
    }

    private static string GenerateTripCommand(Random rng, CorpusPoolVersion v)
    {
        var cities = TruckMatePools.GetCities(v);
        var start = rng.Next(2) == 0;

        // V2: 20% of start-trip rolls emit a multi-stop variant.
        // Distribution stays inside the start_trip family — no new
        // top-level intent — so rng.Next(10) intent histogram is
        // unchanged vs v1.
        if (start && v == CorpusPoolVersion.V2 && rng.Next(100) < 20)
        {
            var c1 = Pick(rng, cities);
            var c2 = Pick(rng, cities);
            var utter = Pick(rng, new[] {
                $"pick up in {c1} then drop in {c2}",
                $"start trip {c1} to {c2}",
                $"multi stop trip {c1} and then {c2}",
                $"{Noise(rng, v)}two stop trip {c1} and {c2}"
            });
            return Format(utter, "start_trip", $"\"stops\":[\"{c1}\",\"{c2}\"]");
        }

        var city = Pick(rng, cities);
        if (start)
        {
            var utterance = Pick(rng, new[] {
                $"{Noise(rng, v)}start my trip to {city}",
                $"{Noise(rng, v)}begin trip {city}",
                $"starting trip headed to {city}",
                $"new trip to {city} {Noise(rng, v)}please",
                $"kick off the trip to {city}"
            });
            return Format(utterance, "start_trip", $"\"destination\":\"{city}\"");
        }
        else
        {
            var utterance = Pick(rng, new[] {
                $"{Noise(rng, v)}stop trip",
                $"end my trip",
                $"trip complete",
                $"close out this trip",
                $"i'm done driving {Noise(rng, v)}stop the trip"
            });
            return Format(utterance, "stop_trip", "");
        }
    }

    private static string GenerateNavigateCommand(Random rng, CorpusPoolVersion v)
    {
        var cities = TruckMatePools.GetCities(v);
        var interstates = TruckMatePools.GetInterstates(v);
        var city = Pick(rng, cities);
        var interstate = Pick(rng, interstates);
        var variant = rng.Next(6);
        var utterance = variant switch
        {
            0 => $"{Noise(rng, v)}take me to {city}",
            1 => $"navigate to {city}",
            2 => $"directions to {city} {Noise(rng, v)}please",
            3 => $"head towards {city} on {interstate}",
            4 => $"route me to the pickup in {city}",
            _ => $"get me to {city} for delivery"
        };
        var slots = $"\"destination\":\"{city}\"";
        if (variant == 3) slots += $",\"via\":\"{interstate}\"";
        if (variant == 4) slots += ",\"purpose\":\"pickup\"";
        if (variant == 5) slots += ",\"purpose\":\"delivery\"";

        // V2: 30% chance to add a time-of-day slot. Never removes an
        // existing slot — purely additive. Keeps token count low.
        if (v == CorpusPoolVersion.V2 && rng.Next(100) < 30)
        {
            var when = Pick(rng, TruckMatePools.GetTimeOfDay(v));
            slots += $",\"when\":\"{when}\"";
            utterance += $" {when}";
        }

        return Format(utterance, "navigate", slots);
    }

    private static string GenerateFindPoiCommand(Random rng, CorpusPoolVersion v)
    {
        var cities = TruckMatePools.GetCities(v);
        var chains = TruckMatePools.GetTruckStopChains(v);
        var stopTypes = TruckMatePools.GetStopTypes(v);
        var interstates = TruckMatePools.GetInterstates(v);
        var stopType = Pick(rng, stopTypes);
        var chain = Pick(rng, chains);
        var city = Pick(rng, cities);
        var variant = rng.Next(5);
        var utterance = variant switch
        {
            0 => $"find me a {stopType} near {city}",
            1 => $"where's the nearest {stopType}",
            2 => $"{Noise(rng, v)}i need {stopType}",
            3 => $"find a {chain} near here",
            _ => $"closest {stopType} off {Pick(rng, interstates)}"
        };
        var slots = $"\"stop_type\":\"{stopType}\"";
        if (variant == 0) slots += $",\"near\":\"{city}\"";
        if (variant == 3) slots = $"\"chain\":\"{chain}\"";
        if (variant == 4) slots += $",\"near_interstate\":\"{Pick(rng, interstates)}\"";
        return Format(utterance, "find_poi", slots);
    }

    private static string GenerateRoutePreferenceCommand(Random rng, CorpusPoolVersion v)
    {
        var pref = Pick(rng, TruckMatePools.GetRoutePrefs(v));
        var utterance = Pick(rng, new[] {
            pref,
            $"{Noise(rng, v)}{pref} please",
            $"i want to {pref}",
            $"set route to {pref}",
            $"change preference to {pref}"
        });
        return Format(utterance, "route_preference", $"\"preference\":\"{pref}\"");
    }

    private static string GenerateHosCommand(Random rng, CorpusPoolVersion v)
    {
        var variant = rng.Next(6);
        var utterance = variant switch
        {
            0 => $"how much time do i got left on my clock",
            1 => $"{Noise(rng, v)}check my hours",
            2 => $"when do i need to stop for my break",
            3 => $"what's my hours of service status",
            4 => $"am i good to keep driving",
            _ => $"how many hours till my thirty minute break"
        };
        var intent = variant <= 1 ? "hos_status" : variant <= 3 ? "hos_break_check" : "hos_drive_remaining";
        return Format(utterance, intent, "");
    }

    private static string GenerateAddTodoCommand(Random rng, CorpusPoolVersion v)
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
            $"{Noise(rng, v)}new task {task}",
            $"put {task} on my list"
        });
        return Format(utterance, "add_todo", $"\"task\":\"{task}\"");
    }

    private static string GenerateAddExpenseCommand(Random rng, CorpusPoolVersion v)
    {
        var amounts = new[] { "45.50", "89.99", "120.00", "32.75", "67.40" };
        var categories = new[] { "fuel", "food", "tolls", "maintenance", "parking", "supplies" };
        var amount = Pick(rng, amounts);
        var category = Pick(rng, categories);
        var utterance = Pick(rng, new[] {
            $"add expense {amount} dollars for {category}",
            $"log {category} expense {amount}",
            $"{Noise(rng, v)}spent {amount} on {category}"
        });
        return Format(utterance, "add_expense", $"\"amount\":\"{amount}\",\"category\":\"{category}\"");
    }

    private static string GenerateStatusCommand(Random rng, CorpusPoolVersion v)
    {
        var cities = TruckMatePools.GetCities(v);
        var variant = rng.Next(5);
        var utterance = variant switch
        {
            0 => $"what's my eta",
            1 => $"how far to {Pick(rng, cities)}",
            2 => $"{Noise(rng, v)}trip status",
            3 => $"what's my next stop",
            _ => $"show me the route summary"
        };
        var intent = variant <= 1 ? "eta_query" : variant == 3 ? "next_stop_query" : "trip_status";
        var slots = variant == 1 ? $"\"destination\":\"{Pick(rng, cities)}\"" : "";
        return Format(utterance, intent, slots);
    }

    private static string GenerateRerouteCommand(Random rng, CorpusPoolVersion v)
    {
        var interstates = TruckMatePools.GetInterstates(v);

        // V2: 25% of reroute rolls emit a weather-alert variant.
        // Still classified under "reroute" so the intent-family
        // histogram is preserved; only the slot schema differs.
        if (v == CorpusPoolVersion.V2 && rng.Next(100) < 25)
        {
            var weather = Pick(rng, TruckMatePools.GetWeather(v));
            var interstate = Pick(rng, interstates);
            var utterance = Pick(rng, new[] {
                $"reroute around {weather}",
                $"there's {weather} on {interstate} reroute me",
                $"avoid {weather} ahead",
                $"{Noise(rng, v)}{weather} on {interstate} find another way"
            });
            return Format(utterance, "reroute", $"\"weather\":\"{weather}\",\"via\":\"{interstate}\"");
        }

        var reasons = new[] {
            "traffic", "construction", "weather", "accident",
            "road closure", "detour", "mountain pass closed"
        };
        var reason = Pick(rng, reasons);
        var utter = Pick(rng, new[] {
            $"reroute around {reason}",
            $"{Noise(rng, v)}avoid {reason} ahead",
            $"find alternate route {reason}",
            $"there's {reason} on {Pick(rng, interstates)} reroute me"
        });
        return Format(utter, "reroute", $"\"reason\":\"{reason}\"");
    }

    private static string GenerateLoadCommand(Random rng, CorpusPoolVersion v)
    {
        var actions = new[] { "picked up", "delivered", "delayed", "weighed" };
        var action = Pick(rng, actions);
        var loadId = $"LD-{rng.Next(10000, 99999).ToString(CultureInfo.InvariantCulture)}";
        var utterance = Pick(rng, new[] {
            $"mark load {loadId} as {action}",
            $"load {loadId} {action}",
            $"{Noise(rng, v)}update load {loadId} to {action}"
        });
        return Format(utterance, "update_load", $"\"load_id\":\"{loadId}\",\"action\":\"{action}\"");
    }

    private static string Format(string utterance, string intent, string slots)
    {
        var slotsJson = string.IsNullOrWhiteSpace(slots) ? "{}" : "{" + slots + "}";
        return $"[USER] {utterance.Trim()} [INTENT] {{\"intent\":\"{intent}\",\"slots\":{slotsJson}}}";
    }

    private static string Noise(Random rng, CorpusPoolVersion v) => Pick(rng, TruckMatePools.GetAsrNoise(v));

    private static T Pick<T>(Random rng, T[] array) => array[rng.Next(array.Length)];
}

/// <summary>
/// Frozen V1 pools + additive V2 extensions for the Truck Mate
/// corpus generator. V1 getters return the exact arrays shipped
/// with <c>truckmate-v1</c>. V2 getters return cached concatenations
/// so V2 examples can reference both V1 and V2-only tokens.
/// </summary>
internal static class TruckMatePools
{
    // ----- V1 (frozen) -----

    private static readonly string[] V1Cities = {
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

    private static readonly string[] V1TruckStopChains = {
        "Flying J", "Pilot", "Love's", "TA Travel Centers", "Petro",
        "Buc-ee's", "QuikTrip", "Sapp Bros", "Kwik Trip", "Casey's"
    };

    private static readonly string[] V1Interstates = {
        "I-10", "I-20", "I-30", "I-35", "I-40", "I-44", "I-55",
        "I-65", "I-70", "I-75", "I-80", "I-81", "I-85", "I-90",
        "I-94", "I-95", "I-5", "I-15", "I-25", "I-45"
    };

    private static readonly string[] V1StopTypes = {
        "truck stop", "fuel", "diesel", "parking", "rest area",
        "weigh station", "scale", "repair shop", "tire shop",
        "restaurant", "shower", "laundry"
    };

    private static readonly string[] V1RoutePrefs = {
        "avoid tolls", "no tolls", "skip the toll road",
        "avoid hazmat restrictions", "stay off restricted routes",
        "avoid low clearances", "watch for low bridges",
        "avoid steep grades", "no mountain passes",
        "shortest route", "fastest route", "most fuel efficient"
    };

    private static readonly string[] V1AsrNoise = {
        "uh ", "um ", "like ", "you know ", "", "", "", "", ""
    };

    // ----- V2 extensions (additive only) -----

    private static readonly string[] V2CityExtensions = {
        "Toledo", "Reno", "Spokane", "Tucson", "Fort Worth",
        "Knoxville", "Chattanooga", "Mobile", "Greensboro", "Lexington",
        "Fresno", "Bakersfield", "Scranton", "Harrisburg", "Roanoke",
        "Macon", "Montgomery", "Gulfport", "Beaumont", "Lubbock",
        "Amarillo", "Wichita", "Topeka", "Sioux Falls", "Fargo",
        "Rapid City", "Billings", "Cheyenne", "Grand Junction", "Flagstaff",
        "Medford", "Eugene", "Yakima", "Twin Falls", "Pocatello",
        "Laramie", "Casper", "Bismarck", "Duluth", "Eau Claire",
        "Green Bay", "Dubuque", "Davenport", "Springfield", "Bloomington",
        "Evansville", "Bowling Green", "Asheville", "Wilmington", "Erie"
    };

    private static readonly string[] V2TruckStopChainExtensions = {
        "Stuckey's", "Iowa 80", "Roady's", "AmBest", "Pilot Travel",
        "FleetPride", "AllStar Travel", "Rip Griffin", "Little America", "Travel America"
    };

    private static readonly string[] V2AsrNoiseExtensions = {
        "so ", "well ", "okay ", "kinda ", "basically ", "I mean "
    };

    private static readonly string[] V2WeatherOnly = {
        "rain", "heavy rain", "snow", "ice", "fog",
        "high winds", "thunderstorm", "freezing rain", "blizzard", "whiteout",
        "dust storm", "wildfire smoke", "flooding", "black ice", "sleet"
    };

    private static readonly string[] V2TimeOfDayOnly = {
        "this morning", "tonight", "by sunrise", "before dark", "late",
        "first thing tomorrow", "this afternoon", "after my break", "by end of shift", "at dawn"
    };

    // ----- Cached V2 concatenations -----
    private static readonly string[] V2Cities = Concat(V1Cities, V2CityExtensions);
    private static readonly string[] V2TruckStopChains = Concat(V1TruckStopChains, V2TruckStopChainExtensions);
    private static readonly string[] V2AsrNoise = Concat(V1AsrNoise, V2AsrNoiseExtensions);

    public static string[] GetCities(CorpusPoolVersion v) =>
        v == CorpusPoolVersion.V1 ? V1Cities : V2Cities;

    public static string[] GetTruckStopChains(CorpusPoolVersion v) =>
        v == CorpusPoolVersion.V1 ? V1TruckStopChains : V2TruckStopChains;

    public static string[] GetInterstates(CorpusPoolVersion v) => V1Interstates; // v1 covers major US interstates

    public static string[] GetStopTypes(CorpusPoolVersion v) => V1StopTypes;

    public static string[] GetRoutePrefs(CorpusPoolVersion v) => V1RoutePrefs;

    public static string[] GetAsrNoise(CorpusPoolVersion v) =>
        v == CorpusPoolVersion.V1 ? V1AsrNoise : V2AsrNoise;

    public static string[] GetWeather(CorpusPoolVersion v) =>
        v == CorpusPoolVersion.V1 ? Array.Empty<string>() : V2WeatherOnly;

    public static string[] GetTimeOfDay(CorpusPoolVersion v) =>
        v == CorpusPoolVersion.V1 ? Array.Empty<string>() : V2TimeOfDayOnly;

    private static string[] Concat(string[] a, string[] b)
    {
        var r = new string[a.Length + b.Length];
        Array.Copy(a, r, a.Length);
        Array.Copy(b, 0, r, a.Length, b.Length);
        return r;
    }
}

/// <summary>Manifest for a generated corpus, written as manifest.{name}.json.</summary>
public sealed record CorpusManifest(
    string Name,
    int TotalExamples,
    int Seed,
    string PoolVersion,
    IReadOnlyList<CorpusShardInfo> Shards);

/// <summary>Metadata for a single shard file in the corpus.</summary>
public sealed record CorpusShardInfo(
    string ShardId,
    string Path,
    int ExampleCount,
    long SizeBytes);
