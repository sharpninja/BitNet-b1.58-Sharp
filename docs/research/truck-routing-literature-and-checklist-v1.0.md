# Truck Routing Literature & Design Checklist v1.0

**Project:** TruckMate Driver-Assistant (BitNet b1.58 / Microsoft Agent Framework)
**Author:** Research compilation, Claude Sonnet 4.6
**Date:** 2026-04-18
**Status:** Living document — v1.0 baseline

---

## Executive Summary

1. **The primary safety gap is bridge clearance routed by consumer GPS.** New York State alone recorded 350 bridge strikes in 2024; 80% are attributed to consumer navigation systems. UK rail networks report ~1,800 annual bridge strikes. The Pentecostal Bridge collapse (2020), Rapps Dam covered bridge (PA, 2014/2022), and the Glenridge Road incident (NY, on-camera, 2026) are directly traced to GPS routing that ignored vehicle height. This is our Rule 1 canary category with the strongest real-world failure evidence.

2. **Authoritative academic algorithms exist for every constraint we need.** The resource-constrained shortest path (RCSP) formulation with label-correcting dynamic programming covers HOS, weight, clearance, and parking simultaneously. Time-dependent variants handle traffic and permit windows. Turn-restriction routing with the OSM `no_u_turn` relation type covers Rule 2. Existing papers give us algorithm templates rather than requiring ground-up design.

3. **Commercial engines (HERE, Trimble/PC*MILER, PTV xRoute, Valhalla, GraphHopper) converge on the same parameter set** for the primary constraints but differ sharply on axle-group weight (HERE and Trimble do it; Valhalla and open-source engines do not by default), ADR tunnel categories (PTV excels; open-source weak), and HAZMAT route registry integration (none open-source; Trimble has curated data). Open-source engines require significant enrichment to reach commercial parity.

4. **Regulations create hard constraint anchors.** STAA National Network (23 CFR 658 Appendix A) defines which roads must be available. 49 CFR 397 Subpart C defines HAZMAT route obligations. FHWA NBIS requires bridge load ratings updated within 90–180 days of inspection; inspections are risk-tiered at 12, 24, or 48 months. These regulatory cycles directly define our data-freshness SLA.

5. **The validator layer must be the last deterministic line of defense.** The LLM must never be asked to approve a route segment — only to describe/explain one. Every constraint in this document must be checked by deterministic C# code with numeric comparisons, not natural-language reasoning. The validator should emit structured rejection messages (template-driven, not generated) with exact numeric values so drivers and logs can be audited.

---

## 1. Papers & Academic Sources

### 1.1 Constrained Shortest Path with Vehicle Attributes

**[RCSP-HOS-PA]**
> Mahmoudi, M., Zhou, X., & Paz, A. (2021). *Scheduling and shortest path for trucks with working hours and parking availability constraints.* Transportation Research Part B, 148, 1–37. https://www.sciencedirect.com/science/article/abs/pii/S0191261521000588

**Covers:** Models the Shortest Path and Truck Driver Scheduling Problem with Parking Availability Constraints (SPTDSP-PA). Formulates HOS compliance, delivery time windows, and parking availability as a resource-constrained shortest path (RCSP) problem. Uses a label-correcting dynamic programming algorithm treating time, cost, and HOS regulation counters as multi-dimensional resources.

**Why it matters to us:** This is the canonical algorithm template for integrating HOS with routing. The label-correcting RCSP approach maps directly to our routing-tool backend. The key finding — that ignoring parking availability can significantly distort cost estimates and that paying for longer routes with guaranteed parking may outperform shorter routes without — directly informs our rest-stop POI integration design.

---

**[RCSP-HOS-HEURISTIC]**
> Goel, A. (2022). *Using state-space shortest-path heuristics to solve the long-haul point-to-point vehicle routing and driver scheduling problem subject to hours-of-service regulatory constraints.* Journal of Heuristics, 28. https://link.springer.com/article/10.1007/s10732-021-09489-7

**Covers:** Long-haul single-truck point-to-point routing with HOS. Builds a multi-dimensional state-space graph iteratively using heuristics. Stops are treated as nodes; the algorithm picks routes that optimize stoppages within HOS constraints.

**Why it matters to us:** Demonstrates that heuristic state-space expansion is tractable for single-vehicle long-haul even under complex HOS. Our on-device use case (single truck, single trip) is a direct fit; we do not need fleet-scale solvers.

---

**[TDSP-TRUCK]**
> Batz, G. V., Geisberger, R., Luxen, D., & Sanders, P. (2017). *Time-Dependent Route Planning for Truck Drivers.* In: Proceedings, Springer LNCS. https://link.springer.com/chapter/10.1007/978-3-319-68496-3_8

**Covers:** Computing time-dependent shortest routes where truck drivers must obey non-stop driving limits, requiring pre-planned breaks at parking lots. Combines time-dependent travel times with working-hour constraints into multiple time-dependent profiles.

**Why it matters to us:** Traffic congestion affects HOS compliance — a route that looks compliant at 2 AM may force a driver into a HOS violation at 5 PM. This paper gives us the framework to reason about time-of-day effects on route legality.

---

**[VRP-HOS-REVIEW]**
> (2025). *Vehicle routing and scheduling under hours of service regulations: A review.* Transportation Research Part A. https://www.sciencedirect.com/science/article/abs/pii/S0965856425002939

**Covers:** Comprehensive literature review of HOS-constrained VRP variants. Covers exact, heuristic, and metaheuristic approaches.

**Why it matters to us:** Good survey to identify which HOS constraint modeling choices are well-studied vs. open research questions.

---

**[VRP-PICKUP-HOS]**
> (2025). *Algorithms for pickup and delivery problems with hours of service constraints.* Computers & Operations Research. https://www.sciencedirect.com/science/article/pii/S0305054825001510

**Covers:** Branch-and-price exact algorithms for Vehicle Routing and Truck Driver Scheduling Problem (VRPTDSP) incorporating route scheduling into resource-constrained shortest path pricing problems.

**Why it matters to us:** If we ever extend to multi-stop itinerary planning, this is the algorithmic foundation. For single-trip use we use simpler RCSP.

---

**[TURN-RESTRICTION-ROUTING]**
> Pepper, P., Rutschmann, F., & Zipf, A. (2013). *Route planning with turn restrictions: A computational experiment.* Operations Research Letters, 41(2). https://www.sciencedirect.com/science/article/abs/pii/S0167637712000752

**Covers:** Graph-theoretic treatment of turn restrictions including U-turns, left/right prohibitions, and multi-via restricted turns. Compares arc-based vs. edge-expansion graph models for encoding turn constraints.

**Why it matters to us:** Rule 2 (no U-turn) requires a routing graph that models prohibited turns as first-class constraints. The edge-expansion model (where a node in the expanded graph corresponds to a directed edge in the original graph) is the standard approach used by OSRM and others. This paper validates the approach and benchmarks its computational cost.

---

**[MULTIOBJECTIVE-TRUCK]**
> Xiao, Y., & Konak, A. (2016). *A new truck-routing approach for reducing fuel consumption and pollutants emission.* Transportation Research Part D, 16(5). https://www.sciencedirect.com/science/article/abs/pii/S1361920910001239

**Covers:** Multi-objective truck routing minimizing fuel consumption and emissions by sequencing deliveries to unload heavy items first, considering idle time at customer sites.

**Why it matters to us:** Fuel efficiency is a secondary objective our drivers will ask about. The 4.9–6.9% fuel savings over shortest-distance baseline is actionable for explaining route choices.

---

**[TDSP-CIRRELT]**
> Jaballah, R., Gendreau, M., & Potvin, J.-Y. (2019). *The Time-Dependent Shortest Path and Vehicle Routing Problem.* CIRRELT Working Paper. https://www.cirrelt.ca/documentstravail/cirrelt-2019-12.pdf

**Covers:** TDSP with time-dependent costs — a foundation for understanding how congestion and time-of-day affect route legality beyond distance.

**Why it matters to us:** Provides formal TDSP models that can be combined with HOS constraints.

---

**[TRUCK-LAST-MILE-ACCESS]**
> (2024). *Quantification of truck accessibility in urban last-mile deliveries using GPS probe data.* Transportation Research Part E. https://www.sciencedirect.com/science/article/pii/S1366554524001273

**Covers:** Quantitative analysis of truck accessibility to POIs using GPS probe data; identifies approach-road constraints in dense urban environments.

**Why it matters to us:** Directly addresses our POI reachability requirement — not just route reachability but approach-road legality for the final segment. Provides methodology for assessing whether a truck can legally access a specific address.

---

### 1.2 Bridge Clearance & Infrastructure-Specific Research

**[TOPO-ROUTING]**
> (2024). *Applying Topological Information for Routing Commercial Vehicles Around Traffic Congestion.* Applied Sciences, 14(22). https://www.mdpi.com/2076-3417/14/22/10134

**Covers:** Using topological road network properties (bridges, ramps, one-ways) to improve truck routing around congestion. Identifies bridge height and weight limit as structural constraints separate from traffic state.

**Why it matters to us:** Confirms bridge attributes must be treated as topological network properties, not soft preferences.

---

**[FHWA-BRIDGE-STRIKE-STUDY]**
> US DOT / University Transportation Center. *Strikes on Low Clearance Bridges by Over-Height Trucks in New York State.* https://www.transportation.gov/utc/strikes-low-clearance-bridges-over-height-trucks-new-york-state

**Covers:** Research project analyzing causes and patterns of over-height truck strikes on low-clearance bridges in New York State. Examines GPS/navigation technology as a contributing factor.

**Why it matters to us:** Primary empirical evidence for our Rule 1 canary test set. Establishes that consumer GPS is causally linked to bridge strike incidents.

---

---

## 2. Standards & Regulations

### 2.1 Federal Vehicle Size and Weight / Route Designation

**[STAA-23CFR658]**
> 23 CFR Part 658 — Truck Size and Weight, Route Designations. https://www.ecfr.gov/current/title-23/chapter-I/subchapter-G/part-658

**Summary:** Defines the National Network of approximately 200,000 miles (Interstate System plus designated non-Interstate Federal-aid Primary routes) open to STAA-dimensioned commercial vehicles. Maximum width: 102 inches. Standard trailer lengths: 48-foot semitrailers or 28-foot doubles. 53-foot single trailers are now legal in 25 states (not yet reflected in 23 CFR 658 — a known gap). States may impose restrictions during peak hours, construction, adverse weather, or for structural/clearance deficiencies.

**Routing implications:** Any compliant routing engine must identify whether a proposed segment is on the National Network and whether a proposed vehicle dimension (especially 53-foot trailers) is permitted on that state's roads. Restrictions on National Network segments must be modeled as time-dependent edge weights.

---

**[FHWA-SIZE-REGS]**
> FHWA. *Federal Size Regulations for Commercial Motor Vehicles.* https://ops.fhwa.dot.gov/freight/publications/size_regs_final_rpt/

**Summary:** Full summary of federal size and weight limits including gross weight (80,000 lbs), single axle (20,000 lbs), tandem axle (34,000 lbs), and the Federal Bridge Gross Weight Formula (W = 500(LN/(N-1) + 12N + 36)). States may have different limits; interstate routes are federally governed.

**Routing implications:** A routing validator must apply the Bridge Formula, not just gross weight, when routing near posted bridges. Any single axle group exceeding its limit makes the load overweight regardless of GVW.

---

**[FHWA-BRIDGE-FORMULA]**
> FHWA. *Bridge Formula Weights.* https://ops.fhwa.dot.gov/FREIGHT/publications/brdg_frm_wghts/index.htm

**Summary:** The Federal Bridge Gross Weight Formula governs axle spacing vs. weight distribution to limit stress on bridge structures. Formula: W = 500 × (LN/(N-1) + 12N + 36), where W = max weight in lbs, L = distance in feet between outer axles of group, N = number of axles in group.

**Routing implications:** Routing engines that only check gross weight will miss Bridge Formula violations. Our validator must implement the formula or accept pre-computed compliance flags from the routing engine.

---

### 2.2 National Bridge Inspection Standards

**[FHWA-NBIS]**
> 23 CFR Part 650 Subpart C — National Bridge Inspection Standards. https://www.ecfr.gov/current/title-23/chapter-I/subchapter-G/part-650/subpart-C

**Summary:** Risk-based inspection intervals: Method 1 classifies each bridge into risk levels with intervals of 12, 24, or 48 months. Maximum interval is 48 months for routine inspections (72 months for underwater). After inspection, SI&A (Structure Inventory and Appraisal) data must be entered into the state/federal inventory within 90 days (state/federal agency bridges) or 180 days (other bridges). Load posting is mandatory when maximum legal load exceeds operating rating.

**Routing implications:** Bridge clearance and weight data in routing graphs can legally be up to 48 months old for low-risk bridges, 90 days from inspection to database entry. Our graph refresh SLA must account for this. Recommended graph refresh: quarterly at minimum; continuous delta-feed where possible.

---

### 2.3 HAZMAT Routing

**[49CFR397-HAZMAT]**
> 49 CFR Part 397, Subpart C — Routing of Non-Radioactive Hazardous Materials. https://www.ecfr.gov/current/title-49/subtitle-B/chapter-III/subchapter-B/part-397/subpart-C

**Summary:** Drivers transporting placarded HAZMAT must avoid: heavily populated areas, places with large gatherings, tunnels (unless no practical alternative), narrow streets, alleys. Carriers may operate on Interstate routes or state-designated preferred routes. States may designate preferred routes (must be published in the FMCSA National Hazardous Materials Route Registry within 60 days of designation).

**Key rule:** Motor carriers transporting Class 1 explosives (Divisions 1.1, 1.2, 1.3) must prepare a written route plan before operating. Radioactive materials (HRCQ/RAM, Class 7) must use preferred routes — Interstate highways unless an alternate is specifically designated.

**Routing implications:** The FMCSA National Hazardous Materials Route Registry (NHMRR) must be queried at route-planning time when HAZMAT load class is specified. Our routing tool must accept HAZMAT class as an input parameter and apply both 49 CFR 397 restrictions and any state-designated route requirements. Restriction codes (by material type) and designation codes must be resolved per-segment.

---

**[FMCSA-NHMRR]**
> FMCSA. *National Hazardous Materials Route Registry.* https://www.fmcsa.dot.gov/regulations/hazardous-materials/national-hazardous-materials-route-registry

**Summary:** Authoritative registry of all designated, preferred, and restricted HAZMAT routes by state, including radioactive and non-radioactive categories. States must report new/changed routes within 60 days.

**Routing implications:** Routing engines must consume this registry. It is updated on a 60-day lag. Our data pipeline must pull the NHMRR at least monthly.

---

**[ADR-TUNNEL]**
> European Agreement Concerning the International Carriage of Dangerous Goods by Road (ADR) — Tunnel Categories A-E.

**Summary:** Tunnels are classified A (no restriction) through E (highest restriction). Vehicles carrying dangerous goods receive a tunnel restriction code based on cargo type. EU/international routing engines (PTV xRoute) apply these codes. For US-only operations, 49 CFR 397 provides the equivalent framework.

**Routing implications:** If our routing engine covers international routes or uses an EU engine (PTV), ADR tunnel categories are mandatory. For domestic US, 49 CFR 397 tunnel restrictions apply.

---

### 2.4 Permit and Oversize/Overweight

**[FHWA-OSOW-PERMITS]**
> FHWA. *Oversize/Overweight Load Permits.* https://ops.fhwa.dot.gov/freight/sw/permit_report/index.htm

**Summary:** Vehicles exceeding legal dimensions (over 80,000 lbs GVW, over 102" width, over 13'6" height in most states, or over legal length) require permits. Permits are state-specific, may require route surveys, escort vehicles, and often impose time-of-day windows (no night travel, no weekend travel for wide loads). Some states require notification of law enforcement on the route.

**Routing implications:** Permit class is a secondary attribute in our system. When activated, the routing engine must: (a) identify permit-required roads, (b) apply time windows, (c) flag escort requirements. Our validator should block non-permitted segments for permitted loads.

---

### 2.5 Navigation System Standards

**[SAE-J2364]**
> SAE J2364 (rev. 2015). *Navigation and Route Guidance Function Accessibility While Driving.* https://www.sae.org/standards/content/j2364_201506

**Summary:** Governs HMI design for in-vehicle navigation — when guidance functions can be accessed while driving, display requirements, and voice guidance timing. Primarily targets driver distraction rather than route constraint completeness.

**Routing implications:** Relevant to our voice/visual output design (the LLM's conversational interface). Rejection messages must be communicable safely while driving — short, structured, no driver data entry while moving.

---

**[ISO-15638-TARV]**
> ISO 15638 series — Intelligent Transport Systems: Telematics Applications for Regulated Commercial Freight Vehicles (TARV).

**Summary:** Framework for telematics data exchange for regulated commercial vehicles. Addresses data interoperability between fleet operators, enforcement authorities, and service providers.

**Routing implications:** Relevant if we integrate with fleet telematics systems. Not critical for the single-driver assistant use case but important for future fleet-management integration.

---

---

## 3. Commercial & Open-Source Engine Feature Matrices

### 3.1 HERE Routing API v8 (Truck Transport Mode)

| Parameter | API Field | Notes |
|-----------|-----------|-------|
| Gross weight | `truck[grossWeight]` | In kg |
| Weight per axle | `truck[weightPerAxle]` | In kg; use when country doesn't differentiate axle groups |
| Weight per single axle | `truck[weightPerSingleAxle]` | US-specific axle group |
| Weight per tandem axle | `truck[weightPerTandemAxle]` | US-specific axle group |
| Height | `vehicle[height]` | In cm |
| Width | `vehicle[width]` | In cm |
| Length | `vehicle[length]` | In cm |
| Axle count | `truck[axleCount]` | Integer |
| Trailer count | `truck[trailerCount]` | Integer |
| Tunnel category | `vehicle[tunnelCategory]` | B, C, D, or E (ADR) |
| Shipped hazardous goods | `vehicle[shippedHazardousGoods]` | Enum: explosive, flammable, combustible, organic, poison, radioactive, corrosive, poisonousInhalation, harmfulToWater, other |
| Time-dependent restrictions | Built-in | Routes penalized if vehicle reaches restricted road during active window |
| Truck turn restrictions | Built-in | Avoids sharp driver-side and passenger-side turns dangerous for long trucks |
| Physical restrictions | Built-in | Height, width, weight, length edge attributes |
| Legal restrictions | Built-in | Prohibited truck maneuvers, prohibited truck roads |

**Strengths:** US-specific axle-group weight; full ADR tunnel categories; strong HAZMAT classification; time-dependent restrictions; production-grade data.

**Gaps for us:** STAA National Network overlay not natively exposed as a filter parameter; FMCSA NHMRR integration is data-sourced, not API-configurable. Pricing is per-request.

---

### 3.2 Trimble PC*MILER / Trimble Maps API

| Parameter | Details |
|-----------|---------|
| Length | 8 ft – 82 ft 10 in (2.44–25 m); default 48 ft |
| Width | 96 in / 98 in / 102+ in (tiered) |
| Height | Custom; bridge clearance database |
| Weight (GVW) | 1,500–156,748 lbs (681–71,099 kg); default 80,000 lbs |
| Axle weight | Supported; Bridge Formula integrated |
| Hazmat | Supported; FMCSA NHMRR-integrated preferred/restricted routes |
| Routing network | STAA National Network-aware; state-by-state permit routing |
| Physical restrictions | Roads unsuitable/forbidden for vehicle dimensions excluded |
| Legal restrictions | "No Thru Traffic," "Delivery Only," truck-banned roads |
| Permit routing | Oversize/overweight permit class routing with time windows |
| Historical traffic | Yes |

**Strengths:** Industry-standard for US commercial routing; STAA National Network embedded in data; FMCSA NHMRR integrated for HAZMAT; Bridge Formula weight calculation; permit class routing.

**Gaps for us:** Commercial licensing cost; API latency for on-device use; OSRM/open-source fallback is weaker.

---

### 3.3 PTV xRoute / PTV Logistics Routing API

| Parameter | Details |
|-----------|---------|
| Height | Tunnel clearance |
| Width | Road width restrictions |
| Weight | Bridge weight limits (maximum weight only — no gross vs. posted distinction) |
| Hazmat | ADR tunnel codes B–E; specific dangerous goods categories |
| Time-dependent | Clean air/action plan zones with time windows; conditional access |
| Through-traffic restrictions | Origin/destination-based access zones (weight-threshold triggered) |
| Routing network | PTV proprietary map data; European strength |

**Strengths:** ADR tunnel category handling is best-in-class for European operations; time-restricted zone modeling.

**Gaps for us:** Weight data limitation — "maximum weight" only, no differentiation between empty+load vs. posted bridge weight (documented limitation in PTV's own docs). Weaker US STAA/NHMRR integration vs. Trimble. Primarily EU-focused.

---

### 3.4 Valhalla (Open Source, Mapbox-hosted)

| Parameter | API Field | Default (truck) |
|-----------|-----------|-----------------|
| Height | `height` | 4.11 m |
| Width | `width` | 2.6 m |
| Length | `length` | 21.64 m |
| Weight (GVW) | `weight` | 21.77 metric tons |
| Axle load | `axle_load` | 9.07 metric tons |
| Axle count | `axle_count` | 5 |
| Hazmat | `hazmat` | false (boolean) |
| HGV no-access penalty | `hgv_no_access_penalty` | Applied |
| Truck route preference | `use_truck_route` | 0–1 factor |
| Unpaved road exclusion | `exclude_unpaved` | Configurable |

**OSM tags consumed:** `maxheight`, `maxheight:physical`, `maxwidth`, `maxwidth:physical`, `maxlength`, `maxweight`, `maxaxleload`, `hgv`, `hazmat`, `hgv:conditional`

**Strengths:** Open-source; self-hostable; good OSM tag coverage for dimensions; axle load support.

**Gaps for us:** Hazmat is boolean (no class differentiation); no ADR tunnel categories; no FMCSA NHMRR; no US axle-group weight (single/tandem/tridem differentiation); no STAA network overlay. Requires extensive data enrichment for production truck safety.

---

### 3.5 OSRM (Open Source Routing Machine)

OSRM has no official truck profile in its core distribution. Community profiles exist (rodo/osrm-profiles):

| Feature | Implementation |
|---------|---------------|
| Truck profile | `truck.lua` — avoids roads where `maxweight <= 3.5` or `maxheight <= 4.0` |
| Hazmat profile | `truck_hazmat.lua` — additional hazmat access restrictions |
| Turn restrictions | `use_turn_restrictions = true` in properties |
| No U-turn | Honored via OSM `restriction=no_u_turn` relations |
| Axle weight | Not implemented in community profiles |
| Time-dependent | Not natively supported; requires CH (Contraction Hierarchies) rebuild |
| HAZMAT classes | Not differentiated (binary hazmat flag only) |

**Strengths:** Extremely fast routing (CH-based); open source; good turn restriction support.

**Gaps for us:** No time-dependent routing natively; no axle-group weight; no HAZMAT classification; community truck profiles are maintenance-dependent; production use requires significant fork work.

---

### 3.6 GraphHopper

| Feature | Details |
|---------|---------|
| Encoded values | `surface`, `toll`, `max_width`, `max_weight`, `max_height`, `hazmat`, `hazmat_tunnel`, `hazmat_water`, `toll`, `track_type` |
| Custom model syntax | JSON/YAML; exclude `hazmat=no` roads when truck carries hazmat |
| Height | `max_height` — exclude roads below threshold |
| Weight | `max_weight` — exclude roads below threshold |
| Width | `max_width` — exclude roads below threshold |
| Axle load | `axle_load` — available as path detail |
| Hazmat subtypes | `hazmat`, `hazmat_tunnel`, `hazmat_water` — basic differentiation |
| Turn restrictions | Via OSM turn restriction relations |
| Time-dependent | Available in enterprise version (not open source) |

**Strengths:** Flexible Custom Model API; decent hazmat subtype differentiation; good OSM coverage.

**Gaps for us:** No US axle-group weight differentiation; no FMCSA NHMRR; time-dependent routing requires enterprise license.

---

### 3.7 Engine Feature Summary Matrix

| Feature | HERE | Trimble | PTV | Valhalla | OSRM | GraphHopper |
|---------|------|---------|-----|----------|------|-------------|
| Height clearance | Yes | Yes | Yes | Yes | Community | Yes |
| Width restriction | Yes | Yes | Yes | Yes | Community | Yes |
| Gross weight | Yes | Yes | Yes | Yes | Community | Yes |
| Per-axle weight (US groups) | Yes | Yes | No* | No | No | No |
| Hazmat (classed) | Yes | Yes | ADR codes | Boolean | Boolean | Basic |
| ADR tunnel categories | Yes | Partial | Yes | No | No | Partial |
| STAA National Network | Data | Yes | No | No | No | No |
| FMCSA NHMRR | Data | Yes | No | No | No | No |
| Time-dependent restrictions | Yes | Yes | Yes | No | No | Enterprise |
| Permit class routing | No | Yes | No | No | No | No |
| U-turn prohibition | Yes | Yes | Yes | Yes | Yes | Yes |
| Bridge Formula (axle spacing) | Partial | Yes | No | No | No | No |

*PTV limitation: weight data is maximum weight only, no empty vs. loaded distinction.

---

## 4. Published Checklists & Requirements Catalogs

### 4.1 FHWA Human Factors Design Guidelines for ATIS/CVO (FHWA-RD-98-057, Chapter 5)

Source: https://www.fhwa.dot.gov/publications/research/safety/98057/ch05.cfm

Relevant requirements extracted (paraphrased):
- Route guidance must present information at a level of detail appropriate for the driver's current task (driving) — not data dumps.
- Voice guidance timing must account for reaction time at highway speeds.
- Reject/reroute messages must indicate the reason for rejection in plain language.
- Driver interface must minimize the need for manual data entry while the vehicle is in motion.
- Navigation systems for commercial vehicles must support vehicle profile entry (dimensions, HAZMAT class) before trip start, not mid-trip.

### 4.2 FMCSA Commercial Vehicle GPS Navigation Guidance

Source: FMCSA Safety Advisory (referenced via Clearinghouse Navigator and FMCSA bridge strike communications)

Key published requirements:
- Commercial navigation systems must accept vehicle height, weight, and length as input parameters.
- Systems must warn before routing through low-clearance structures, weight-restricted bridges, or roads legally prohibited to CMVs.
- FMCSA cautions carriers to "invest in navigation systems specifically designed for the truck and bus industry" and not use passenger-vehicle GPS.
- Systems should be updated regularly to reflect current bridge postings.

### 4.3 New York State DOT Bridge Strike Mitigation Requirements (2024)

Source: Governor Hochul announcement; NY Thruway Authority enforcement campaign

Requirements derived from public policy:
- Detection systems: infrared/sensor-based over-height vehicle detection at known strike-prone structures.
- Warning infrastructure: LED blank-out signs, flashing beacons, graphical "No Trucks" signage (including non-English).
- Turnaround facilities: truck-accessible turnaround areas provided before restriction points.
- Enforcement: dedicated commercial vehicle enforcement details with violations for over-height operation.
- Navigation guidance: explicit recommendation for commercial-grade GPS; awareness campaign targeting consumer-GPS use by CMV drivers.

### 4.4 ATA / OOIDA / Driver Community Requirements (synthesized from Trucker Path, Hammer GPS reviews, TruckersReport forums)

Most-cited features (priority order from driver feedback):
1. Truck-safe routing (height, weight, length) — non-negotiable baseline.
2. Real-time parking availability at truck stops.
3. Fuel price comparison along route.
4. Weigh station open/closed status (community-updated).
5. Rest area locations with amenities.
6. Low-bridge alerts with audible warning before approach.
7. Offline capability (dead zones on rural interstates).
8. Route comparison: distance vs. stops vs. estimated fuel.
9. POI with trucker-specific amenities (scales, showers, certified scales).
10. ELD/HOS remaining hours integration.

---

## 5. Known Failure Modes & Incident Case Studies

Each incident below includes a proposed canary test for our validator/routing test suite.

---

### FM-01: Consumer GPS Routes Truck Under Low-Clearance Bridge

**Incident:** Glenridge Road Rail Bridge, Glenville, NY (2026). Driver confirmed following GPS. Clearance: 10'11". Incident caught on camera by NYSDOT sensors.

**Incident:** Multiple covered bridges in Vermont (Miller's Run, 1878 construction, 24 documented strikes). 2019 delivery truck hit: ~$100,000 in engineering and repair costs. Driver cited GPS as cause. Fine: $5,000 + state penalties.

**Incident:** Rapps Dam Covered Bridge, East Pikeland Township, PA (2014). Tractor-trailer following GPS demolished most support beams. Driver subsequently fired.

**Incident:** York County, PA covered bridge (2022). Strike required temporary stabilizing frame. Second strike six months later pushed repair estimate to $1.5 million.

**Root cause:** Consumer GPS (Google Maps, Apple Maps, phone navigation) does not encode vehicle height and routes on shortest-time basis ignoring clearance restrictions.

**Statistics:** NY State — 350 bridge strikes in 2024, 231 on Thruway since 2020, 56 on Thruway in 2024. Senator Schumer: 80% of bridge strikes attributed to wrong GPS. UK: ~1,800 rail bridge strikes/year. Cost: $42M in recent NY repairs; £23M/year in UK rail.

**Proposed canary test CANARY-FM01:** Given a truck with height = 14'0" (standard), route it from Point A to Point B where the shortest path requires traversing a bridge with posted clearance of 13'6". Validator must reject the segment and propose an alternate route. Assert: (a) segment rejected before route is returned to user, (b) rejection message includes exact clearance value (13'6") and truck height (14'0"), (c) alternate route does not pass any clearance < 14'6" (14'0" + 6" safety margin).

---

### FM-02: Weight-Restricted Bridge Collapsed by Over-Weight Truck Following GPS

**Incident:** Dale Bend Bridge, Yell County, AR (January 30, 2019). Singh's GPS directed him to the bridge while hauling processed chicken. The 88-year-old bridge (posted 6-ton limit) collapsed under the semi truck. Truck partially submerged in Petit Jean River. Lawsuit filed.

**Incident:** Pentecostal Bridge (2020). A Freightliner Cascadia (GVW more than 8x the posted 5-ton limit) collapsed the bridge. Driver followed GPS directed route.

**Root cause:** GPS directed trucks across bridges without checking posted weight limits. Bridge collapse risk.

**Proposed canary test CANARY-FM02:** Given a truck with GVW = 44 tons, route it via a segment containing a bridge posted at 5 tons. Validator must reject. Assert: (a) rejection includes posted weight and truck GVW, (b) route returned uses only bridges rated for the truck's GVW with safety margin, (c) if no alternate route exists, validator returns NO_VIABLE_ROUTE rather than an unsafe route.

---

### FM-03: Consumer GPS Routes Truck onto Motorcycle/Unpaved/Restricted Road

**Incident:** Jakarta, Indonesia — truck driver drove off cliff following Google Maps route designated only for motorcycles.

**Incident:** Oregon couple stranded in snow after Google Maps directed them onto unmaintained road.

**Root cause:** Consumer routing does not distinguish road class by vehicle type. No minimum road-class filter for commercial vehicles.

**Proposed canary test CANARY-FM03:** Route a truck from A to B where shortest path includes a road tagged `hgv=no` or road class < residential. Validator must reject. Assert: route uses only `hgv=yes` or `hgv=destination` (if delivery endpoint) roads, or road class sufficient for commercial vehicles.

---

### FM-04: Google Maps Fails to Update Collapsed Bridge (9-Year Lag)

**Incident:** Philip Paxson, North Carolina (September 2022). Google Maps directed Paxson over Snow Creek Bridge, which had collapsed nearly 9 years earlier. Fatal fall into creek. Lawsuit filed alleging Google received multiple citizen reports of the collapse starting September 2020 and took no action. Google confirmed receipt of a report in November 2020 but made no correction before the fatal incident.

**Root cause:** Map data staleness combined with no audit process for reported hazards.

**Proposed canary test CANARY-FM04 (data-freshness test):** Synthetic test — inject a "bridge closed/reduced clearance" delta into the test routing graph with a known timestamp. Assert: (a) routing engine picks up the delta within the configured refresh interval (target: 24 hours for structural changes), (b) any route using that bridge segment after the delta injection is rejected, (c) a test query to explain the rejection includes the source timestamp of the restriction.

---

### FM-05: U-Turn Maneuver on Highway Routed by Navigation System

**Incident type (class):** Driver directed to U-turn on divided highway or high-speed road. No single named incident confirmed as GPS-directed, but U-turn crashes on highways are a documented crash pattern in FMCSA data.

**Root cause:** Routing algorithms that allow U-turns on divided roads as a rerouting maneuver when a missed turn is detected.

**Proposed canary test CANARY-FM05:** Route a truck where the on-route recalculation from a missed turn would naturally suggest a U-turn (opposite direction on a divided highway). Validator must prohibit the U-turn maneuver at any turn angle ≥ 150° on a road with divided carriageway OR road class ≥ secondary. Assert: (a) no route step with turn type `u_turn` or heading change ≥ 150° is returned, (b) rerouting instead uses next legal right/left turn to backtrack or identifies the nearest truck-accessible parking-lot turnaround.

---

### FM-06: Narrow Road Incident — Commercial Vehicle Cannot Maneuver on Sub-Standard Width Road

**Incident type (class):** Truck routed onto a road with insufficient width for the vehicle, requiring backing maneuvers or causing property damage.

**Root cause:** Routing engine uses road centerline data without checking carriageway width against vehicle width.

**Proposed canary test CANARY-FM06:** Route a truck with width = 8'6" (102") through a road segment tagged `maxwidth=2.5m` (~8'2"). Validator must reject or reroute. Assert: route uses only segments where `maxwidth >= vehicle_width + 0.2m` (safety margin).

---

### FM-07: Hazmat Truck Routed Through Tunnel / Populated Area

**Incident type (class):** HAZMAT truck routed through a tunnel prohibited for its cargo class, or through a heavily populated area when alternate routes exist.

**Root cause:** Routing engine does not check HAZMAT class against tunnel category or population-density restrictions.

**Proposed canary test CANARY-FM07:** Route a truck carrying Div 1.1 explosives through a path that includes a tunnel classified as ADR category B or higher restriction. Validator must reject. Assert: (a) route returned avoids all tunnel categories restricted for the cargo's ADR code, (b) rejection message specifies cargo class, tunnel ID, and the applicable restriction.

---

### FM-08: Routing to POI with Approach Road Below Truck Spec

**Incident type (class):** Driver navigated to a truck stop or distribution center via an approach road with a low bridge, weight-restricted road, or insufficient turning radius at the entrance. No single named incident in public record, but this is a commonly cited driver complaint in community forums.

**Root cause:** Routing engine validates the inter-city route but does not validate the final approach segment to the POI.

**Proposed canary test CANARY-FM08:** Route a truck to a POI where the last 0.5-mile approach road has a clearance restriction below the truck's height. Validator must detect the restriction on the approach segment, not just the main route. Assert: (a) rejection occurs even if all non-approach segments are compliant, (b) alternate approach route or notification of "no truck-legal approach" is returned.

---

---

## 6. Synthesized Design Checklist for Our Validator + Routing Engine

### Group A: Hard Safety — Rule 1 (Height / Width / Weight on Every Segment)

**A-01.** Every route segment, including the final approach segment to any POI or destination, must be checked against the truck's height, width, and GVW. A route is acceptable only if ALL segments pass; partial compliance is treated as full rejection.

**A-02.** The height check must apply a configurable safety margin above the posted clearance. Default: +6 inches (15 cm). The margin must be configurable but never zero. Rejection message must include: posted clearance, truck height, margin, and segment identifier.

**A-03.** The weight check must validate: (a) GVW ≤ posted bridge limit, (b) per-axle weight ≤ single-axle limit (20,000 lbs), (c) tandem-axle group ≤ 34,000 lbs, (d) Federal Bridge Formula compliance W = 500(LN/(N-1) + 12N + 36). A route segment fails if any one of these sub-checks fails.

**A-04.** Width validation must check carriageway width (not just legal restriction tags) when the data source provides physical width. Minimum passable carriageway = vehicle width + 0.4 m (20 cm each side). Rejection message must include: road width, vehicle width, shortfall.

**A-05.** The validator must not rely on the LLM to evaluate compliance. The LLM receives only the pre-validated route or a structured rejection object, never raw segment data for safety evaluation.

**A-06.** If no compliant route exists between origin and destination, the validator must return a `NO_VIABLE_ROUTE` status with the specific blocking constraint and its location, not a silent failure or a non-compliant route.

---

### Group B: Hard Safety — Rule 2 (No U-Turn Maneuver)

**B-01.** No route step may include a turn with heading change ≥ 150° (absolute value). This threshold captures all U-turns and near-U-turns. The threshold must be configurable; 150° is the default but should be validated against empirical turning data for standard 53-foot trailer configurations.

**B-02.** The U-turn prohibition applies to all road classes including local roads, parking lots shared with public roads, and access roads. Exception: a designated truck-turnaround area (tagged in routing data as a legal turning facility) is permitted.

**B-03.** Rerouting after a missed turn must never produce a U-turn. The rerouting algorithm must find the next legal turn (right or left at an intersection) to backtrack, or identify the nearest truck-accessible parking-lot turnaround (minimum turning radius ≥ the truck's off-tracking radius for the configured length/wheelbase).

**B-04.** All OSM `restriction=no_u_turn` relations on the routing graph must be honored. The edge-expansion graph model (turn-penalty model) must be used for the routing graph, not the simplified edge-weighted model, to ensure turn prohibitions are first-class constraints.

---

### Group C: Secondary Safety Attributes (Activated by Load Type)

**C-01.** Length: when `trailer_length` is provided, check `maxlength` on all segments. Rejection includes measured maxlength and trailer length.

**C-02.** Axle configuration: when `axle_count` and `axle_spacing` are provided, apply the Federal Bridge Formula per bridge segment. The validator must receive axle spacing data or default to a conservative assumption for the declared axle count.

**C-03.** HAZMAT class: when `hazmat_class` is provided (UN class + division), the validator must: (a) exclude roads tagged `hazmat=no`, (b) apply the correct ADR tunnel restriction code for the cargo, (c) query FMCSA NHMRR for state-designated preferred/restricted routes and enforce them, (d) exclude tunnels, narrow streets/alleys, and heavily populated areas per 49 CFR 397.

**C-04.** Permit class: when the load requires a permit (over-dimensional or overweight), the validator must check time-of-day windows (permits typically exclude night travel and weekends for wide loads), escort requirements, and bridge-specific permit exclusions. Reject if permit conditions cannot be satisfied on the proposed route at the proposed departure time.

**C-05.** Radioactive materials (Class 7, HRCQ/RAM): route must use FMCSA preferred routes (Interstate system or state-designated alternate). Validator must enforce this as a hard constraint, not a preference.

---

### Group D: Routing Quality

**D-01.** STAA National Network: the routing engine must prefer routes on the STAA National Network (23 CFR 658 Appendix A) for STAA-dimensioned standard commercial vehicles. Deviation from National Network should require a reason (origin/destination not reachable from National Network, construction closure, etc.).

**D-02.** Time-of-day restrictions: segments with time-dependent restrictions (peak-hour truck bans, construction windows, seasonal weight limits, permit windows) must be validated against the route's estimated time of arrival at each segment, not against departure time.

**D-03.** HOS compliance: the routing engine must accept remaining HOS hours as an input parameter and ensure the route can be completed (including mandatory stops) within available hours. Mandatory 30-minute break and 10-hour rest periods must be modeled. Parking availability at candidate rest stops must be a feasibility constraint (see [RCSP-HOS-PA]).

**D-04.** Traffic and speed: time-dependent travel times must be used for HOS calculations. A route that appears HOS-compliant under free-flow speeds may fail under congested conditions.

**D-05.** Route alternatives: the routing engine must return at minimum 2 alternative routes when the primary route is rejected or when alternatives differ by < 15% in distance. Each alternative must pass the full validator check.

**D-06.** Oversize load routing: for permitted loads, the route must avoid weight-restricted segments even if the permit allows the GVW, and must include all permit-required waypoints (weigh stations, inspection points).

---

### Group E: POI / Destination Access

**E-01.** Approach road check: every POI and destination must have its final approach road (last 1 mile minimum) validated independently against all active constraints. A POI is "truck-accessible" only if both the main route AND the approach are compliant.

**E-02.** Parking-lot turnaround feasibility: before routing to a POI, the validator must verify that a truck-legal turnaround maneuver is possible at or near the destination. Minimum criteria: a parking lot with truck-accessible entry/exit, sufficient swept area for the truck's turning radius, or a designated truck turnaround area within 0.5 miles.

**E-03.** Truck stop validation: when selecting rest/fuel/parking POIs along a route, the POI dataset must include approach-road clearance data, not just POI coordinates. A truck stop inaccessible to the vehicle's height must be excluded from recommendations.

**E-04.** Weigh station status: integrate real-time weigh station open/closed status (community data or state DOT feeds where available). Route must include mandatory weigh station stops for applicable loads.

**E-05.** Fuel stop range: the routing engine must compute fuel range for the configured vehicle (based on typical loaded MPG) and ensure a diesel-available fuel stop is reachable before range is exhausted. Alert if no truck-accessible fuel stop exists within 80% of range.

---

### Group F: Explanation Faithfulness

**F-01.** Every validator rejection must produce a structured rejection object with: (a) rejection code (e.g., `HEIGHT_VIOLATION`, `WEIGHT_VIOLATION`, `U_TURN_PROHIBITED`), (b) the segment identifier or coordinates where the violation occurs, (c) the constraint value (posted limit), (d) the vehicle value (truck spec), (e) the margin or delta. No free-text only rejections.

**F-02.** The LLM's explanation of a rejection to the driver must be generated from the structured rejection object, not from reasoning about the raw route. The template must include the numeric values from the rejection object.

**F-03.** Rejection messages for HAZMAT violations must cite the specific regulation (e.g., "49 CFR 397.67 — tunnel prohibited for your cargo class").

**F-04.** If an alternate route is proposed, the explanation must state why the original route was rejected and what makes the alternate compliant (e.g., "The alternate avoids the 12'6" clearance on Route 9 and uses the I-90 underpass at 15'2"").

**F-05.** The validator must log every rejection with timestamp, vehicle profile used, rejected route hash, and rejection object. Logs must be queryable for safety audits.

---

### Group G: Data Freshness

**G-01.** Bridge clearance and weight limit data must be refreshed at minimum quarterly. Target: monthly delta-feed from NBI (National Bridge Inventory) and state DOT databases.

**G-02.** FMCSA NHMRR (HAZMAT route registry) must be refreshed at minimum monthly. States must report changes within 60 days; our pipeline must consume within 30 days of state reporting.

**G-03.** Each restriction in the routing graph must carry a `source_timestamp` indicating when the data was last verified. Routes must expose the oldest source timestamp for any restriction on the route, enabling the validator to emit a data-age warning if any restriction is older than the configured staleness threshold (default: 180 days).

**G-04.** Community-reported restrictions (e.g., weigh station closures, construction) must be time-stamped and must expire after a configurable TTL (default: 24 hours) unless confirmed by authoritative source.

**G-05.** The graph must support incremental updates (delta feeds) without full rebuild for safety-critical changes (bridge closures, new restrictions). Full rebuild frequency: weekly at most. Delta ingestion: continuous or at minimum daily.

**G-06.** A data-freshness health check must be part of the routing service startup sequence. If the routing graph is older than the configured staleness threshold, the service must log a warning and optionally refuse to start (configurable per deployment).

---

### Group H: Test Coverage

The following canary categories must be present in the integration test suite. Each category must have at minimum 3 test cases (a known-safe route, a known-unsafe route that must be rejected, and a borderline case at the constraint boundary).

| Canary ID | Category | Rule |
|-----------|----------|------|
| CANARY-FM01 | Low-clearance bridge (height violation) | A-01, A-02 |
| CANARY-FM02 | Weight-restricted bridge (GVW and Bridge Formula) | A-03 |
| CANARY-FM03 | HGV-prohibited road (access tag) | A-01, C-01 |
| CANARY-FM04 | Stale data / recently closed bridge | G-01, G-03 |
| CANARY-FM05 | U-turn prohibition on highway | B-01, B-02 |
| CANARY-FM06 | Width-restricted road | A-04 |
| CANARY-FM07 | HAZMAT tunnel restriction | C-03 |
| CANARY-FM08 | POI approach road violation | E-01, E-02 |
| CANARY-NEW-01 | HOS-overrun route (route exceeds remaining hours without valid rest stop) | D-03 |
| CANARY-NEW-02 | Time-of-day restriction (truck ban active at ETA) | D-02 |
| CANARY-NEW-03 | Permit-class time window (night travel for wide load) | C-04 |
| CANARY-NEW-04 | No viable turnaround at destination | E-02, B-03 |
| CANARY-NEW-05 | STAA National Network deviation (unnecessary off-network routing) | D-01 |
| CANARY-NEW-06 | Radioactive material preferred-route violation | C-05 |
| CANARY-NEW-07 | Federal Bridge Formula violation (legal GVW but illegal axle spacing) | A-03 |
| CANARY-NEW-08 | State-specific axle weight limit (different from federal) | A-03 |
| CANARY-NEW-09 | Fuel range exhausted before next truck-accessible diesel stop | E-05 |
| CANARY-NEW-10 | Consumer-GPS-style misdirection on fastest path through restricted road | A-01 |

Minimum total: 18 categories × 3 cases = 54 integration tests. Boundary cases (at-limit) are the most important — pass them as regression tests against every routing engine version.

---

## 7. Open Questions / Gaps

The following areas have insufficient literature or no published standards. These must be solved by the project team.

**OQ-01: Turn Angle Threshold for Rule 2.** Our current draft is ≥ 150° = prohibited U-turn. The Waze algorithm uses ±5° of parallel (i.e., a ~170° heading change) as the geometric trigger for U-turn detection. For trucks, the relevant criterion may be different from geometry alone: a 150° turn on a wide boulevard is not dangerous; a 160° turn on a narrow two-lane road may be impossible. No published standard defines the truck-specific threshold. Recommended: empirically derive from real routing data by measuring heading changes at known U-turn locations in OSM, then validate against known impossible-turn incidents. Revisit threshold at 120°, 140°, 150°, 160°, and 170° for false-positive/false-negative analysis.

**OQ-02: Parking-Lot Turnaround Feasibility Computation.** No published standard defines the minimum swept area or turning radius required for a truck to perform a legal (non-U-turn) turnaround in a parking lot. The off-tracking calculation for a 53-foot trailer is well-understood geometrically, but translating this to a routing-graph attribute (which parking lots are feasible?) requires: (a) parking lot geometry data (OSM coverage is incomplete), (b) a computational geometry check (swept path model). This is original engineering work.

**OQ-03: Approach Road Definition.** What constitutes the "approach road" to a POI for validation purposes? The last 0.5 miles is a reasonable heuristic but has no regulatory basis. The research in [TRUCK-LAST-MILE-ACCESS] uses GPS probe data to empirically define approach road boundaries. We must either replicate this analysis for our service area or define an operational heuristic and document its limitations.

**OQ-04: STAA National Network as Routable Layer.** The STAA National Network is defined in 23 CFR 658 Appendix A but is not directly available as a routable geospatial layer in open datasets. Trimble/PC*MILER has curated this data. Open-source alternatives would require digitizing from FHWA's PDF appendix or purchasing from a data vendor. This is a significant data engineering task for any open-source routing approach.

**OQ-05: OSM Truck Data Quality by Region.** OSM truck-attribute coverage varies dramatically by region. Urban areas in the US have moderate coverage; rural areas are often missing `maxheight` tags on bridges. GraphHopper and Valhalla both route against whatever OSM data is present; missing data means missing restrictions (false negatives, not false positives). A systematic data quality audit by route corridor is needed before any open-source engine is used in production.

**OQ-06: Bridge Clearance Data Sources Beyond NBI.** The National Bridge Inventory covers federally cataloged bridges but does not cover all underpasses, railroad crossings, or private infrastructure. State DOT datasets vary in completeness. Commercial vendors (HERE, Trimble) augment NBI with field surveys and user reports. For an open-source approach, supplementary data from LowClearanceMap.com (23,000+ crowd-sourced verified entries) or similar projects may be needed.

**OQ-07: Real-Time Restriction Integration.** Bridge closures (structural), weight-restricted seasonal postings (spring thaw), and construction zone restrictions can change within hours. No open-source routing engine has a production-grade real-time restriction ingestion pipeline. This is a gap that requires custom infrastructure.

**OQ-08: Explanation Faithfulness vs. Model Verbosity.** When the validator rejects a route and the LLM explains the rejection to the driver, the explanation must be accurate but also concise and safe for voice delivery at highway speeds. There is no published standard for the appropriate level of detail in voice-delivered route rejection explanations for CMV drivers. We must define this through user testing.

**OQ-09: Multi-State Permit Coordination.** Oversize/overweight loads require separate permits from each state. No public API aggregates multi-state permit routing. Trimble has some integration; no open-source equivalent exists. For permit-class loads, the validator may need to flag manual permit verification steps.

---

## 8. Appendix: Full Link List

### Academic Papers
- [RCSP-HOS-PA] https://www.sciencedirect.com/science/article/abs/pii/S0191261521000588
- [RCSP-HOS-HEURISTIC] https://link.springer.com/article/10.1007/s10732-021-09489-7
- [TDSP-TRUCK] https://link.springer.com/chapter/10.1007/978-3-319-68496-3_8
- [VRP-HOS-REVIEW] https://www.sciencedirect.com/science/article/abs/pii/S0965856425002939
- [VRP-PICKUP-HOS] https://www.sciencedirect.com/science/article/pii/S0305054825001510
- [TURN-RESTRICTION-ROUTING] https://www.sciencedirect.com/science/article/abs/pii/S0167637712000752
- [MULTIOBJECTIVE-TRUCK] https://www.sciencedirect.com/science/article/abs/pii/S1361920910001239
- [TDSP-CIRRELT] https://www.cirrelt.ca/documentstravail/cirrelt-2019-12.pdf
- [TRUCK-LAST-MILE-ACCESS] https://www.sciencedirect.com/science/article/pii/S1366554524001273
- [TOPO-ROUTING] https://www.mdpi.com/2076-3417/14/22/10134
- Time-dependent VRP (INFOR) https://www.tandfonline.com/doi/full/10.1080/03155986.2021.1973785
- VRP survey (arxiv) https://arxiv.org/pdf/2303.04147
- Green VRP systematic review https://www.tandfonline.com/doi/full/10.1080/23311916.2020.1807082

### Standards & Regulations
- 23 CFR Part 658 (STAA National Network) https://www.ecfr.gov/current/title-23/chapter-I/subchapter-G/part-658
- 23 CFR Part 650 Subpart C (NBIS) https://www.ecfr.gov/current/title-23/chapter-I/subchapter-G/part-650/subpart-C
- 49 CFR Part 397 (HAZMAT Driving/Parking) https://www.ecfr.gov/current/title-49/subtitle-B/chapter-III/subchapter-B/part-397
- 49 CFR 397 Subpart C (NRHM Routing) https://www.ecfr.gov/current/title-49/subtitle-B/chapter-III/subchapter-B/part-397/subpart-C
- FHWA Bridge Formula Weights https://ops.fhwa.dot.gov/FREIGHT/publications/brdg_frm_wghts/index.htm
- FHWA Federal Size Regulations https://ops.fhwa.dot.gov/freight/publications/size_regs_final_rpt/
- FHWA National Network page https://ops.fhwa.dot.gov/freight/infrastructure/national_network.htm
- FMCSA NHMRR https://www.fmcsa.dot.gov/regulations/hazardous-materials/national-hazardous-materials-route-registry
- FMCSA Current HAZMAT Route List https://www.fmcsa.dot.gov/regulations/hazardous-materials/current-list-designated-preferred-and-restricted-hazardous-materials
- FHWA NBIS Inspection Frequency https://www.fhwa.dot.gov/bridge/nbis/frequency.cfm
- FHWA Oversize/Overweight Permits https://ops.fhwa.dot.gov/freight/sw/permit_report/index.htm
- SAE J2364 (Nav Accessibility While Driving) https://www.sae.org/standards/content/j2364_201506
- FHWA ATIS/CVO Human Factors Ch.5 https://www.fhwa.dot.gov/publications/research/safety/98057/ch05.cfm
- FHWA Advancing Bridge Load Rating https://www.fhwa.dot.gov/bridge/loadrating/pubs/hif22059.pdf

### Commercial Engine Documentation
- HERE Truck Routing (blog) https://www.here.com/learn/blog/truck-routing
- HERE 5 Truck Challenges https://www.here.com/learn/blog/truck-challenges-here-routing-api-v8
- HERE AWS Truck Routing https://www.here.com/learn/blog/truck-routing-aws
- Trimble PC*MILER https://transportation.trimble.com/en/solutions/mapping-and-routing/pcmiler
- Trimble Bridge Strikes https://transportation.trimble.com/blog/avoid-bridge-strikes-with-commercial-vehicle-specific-routing
- Trimble Vehicle Dimensions https://learn.transportation.trimble.com/wp-content/uploads/tte/ebcbe19c93c746dd320c/olhlp/a1f22fffb258/docs/Current/RouteOptions/3.6.14-VehicleDimensions.html
- PTV Truck Restrictions https://developer.myptv.com/en/documentation/routing-api/concepts/truck-restrictions
- PTV Truck Attributes https://xtour-eu-n-test.cloud.ptvgroup.com/manual/Content/Use%20cases/xRoute/DSC_TruckAttributesFormat.htm
- Valhalla API Reference https://valhalla.github.io/valhalla/api/turn-by-turn/api-reference/
- Valhalla GitHub https://github.com/valhalla/valhalla
- OSRM Truck Profile (community) https://github.com/rodo/osrm-profiles/blob/master/truck.lua
- OSRM Profile Docs https://github.com/Project-OSRM/osrm-backend/blob/master/docs/profiles.md
- GraphHopper Truck Docs https://docs.graphhopper.com/openapi/map-data-and-routing-profiles
- GraphHopper Forum (maxWeight/maxHeight) https://discuss.graphhopper.com/t/how-i-can-add-maxweight-maxheight-restrictions-for-small_truck-profile-in-java-config/6962

### OSM Data References
- OSM Key:hgv https://wiki.openstreetmap.org/wiki/Key:hgv
- OSM Key:maxweight https://wiki.openstreetmap.org/wiki/Key:maxweight
- OSM Restrictions https://wiki.openstreetmap.org/wiki/Restrictions
- OSM Tag:restriction=no_u_turn https://wiki.openstreetmap.org/wiki/Tag:restriction=no_u_turn
- OSM Relation:restriction https://wiki.openstreetmap.org/wiki/Relation:restriction

### Incident Sources
- NY DOT Bridge Strike Mitigation (Hochul) https://www.governor.ny.gov/news/governor-hochul-announces-new-measures-mitigate-bridge-strikes-upstate-new-york
- NY Thruway Bridge Strike Data https://www.thruway.ny.gov/news/pressrel/2025/01/2025-01-09-com-vehicle-enforcement.html
- NYC Open Data Bridge Strikes https://data.cityofnewyork.us/Transportation/NYC-Bridge-Strike-Data/jdn9-td9w
- The Trucker — Covered Bridges GPS https://www.thetrucker.com/trucking-news/the-nation/historic-covered-bridges-are-under-threat-by-truck-drivers-relying-on-gps-meant-for-cars
- CDL Life — Yell County Bridge Collapse (GPS) https://cdllife.com/2021/trucking-company-and-driver-sued-after-following-gps-onto-6-ton-bridge-causing-collapse/
- CDL Life — NY Glenridge GPS Strike (2026) https://cdllife.com/2026/new-york-state-dot-says-semi-truck-driver-was-following-gps-in-bridge-strike-caught-on-video/
- Google Maps Snow Creek Bridge Lawsuit https://www.aljazeera.com/news/2023/9/21/google-sued-after-man-drove-off-collapsed-bridge-following-map-directions
- Bridge Strike Wikipedia https://en.wikipedia.org/wiki/Bridge_strike
- List of bridges known for strikes https://en.wikipedia.org/wiki/List_of_bridges_known_for_strikes

### Driver Community Sources
- Trucker Path App https://truckerpath.com/trucker-path-app
- TruckersReport Forum https://www.thetruckersreport.com/truckingindustryforum/
- Hammer GPS (TruckersReport community) https://play.google.com/store/apps/details?id=com.truckersreport.hammer
- FreightWaves Best Truck Route Apps https://www.freightwaves.com/news/playbook-gps-best-truck-routes
- Low Clearance Map (crowdsourced) https://lowclearancemap.com/
- NextBillion — Height/Weight Routing https://nextbillion.ai/blog/vehicle-height-and-weight-aware-routing
- eLogII — Google Maps for Trucks Limitations https://elogii.com/blog/google-maps-for-trucks

---

*End of document. Version 1.0. Intended audience: routing/validator engineering team. Review cycle: annually or upon major regulation change.*
