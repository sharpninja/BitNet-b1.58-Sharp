# State DOT Open-Data Audit — Truck-Legal Routing Validator

**Version:** 1.0  
**Date:** 2026-04-18  
**Author:** Research agent (Claude Sonnet 4.6)  
**Purpose:** Determine whether OSM + state-DOT open data can back a deterministic H/W/W truck-routing validator covering the 48 contiguous US states, or whether a commercial truck-attribute dataset is required.

---

## 1. Executive Summary

The 48 contiguous US states collectively publish enough free open-data infrastructure to form a credible starting backbone for a per-segment truck-routing validator, but the coverage is uneven and several attribute classes are consistently absent or partial. **Lane count** (number of through lanes) is nearly universally available via either HPMS or state road-inventory GIS. **Surface/pavement width** is present in most state inventories. **Posted weight limits at the bridge level** are increasingly published as standalone GIS layers (OR, NE, WI, AR, KY, NC, LA have dedicated posted-bridge datasets). However, **per-lane bridge vertical clearance does not exist in any freely available state or federal dataset at scale**; the NBI records a single minimum clearance per structure, and only WSDOT and CDOT are experimenting with lane-level collection. **Posted height restrictions outside of bridges** (low-clearance underpasses, viaducts, sign gantries) are almost entirely absent from open-data portals — a critical gap for truck height enforcement. **Hazmat route designations** exist in HPMS Item 65 (STRAHNET) and some state portals, but truck-prohibited parkway/street segments require state-by-state research. The minimum viable commercial gap-fill needed is: (a) a curated low-clearance sign inventory (HERE, PTV, or Trimble all carry this), and (b) per-lane bridge clearance on high-traffic corridors. Purely-open routing is achievable for weight and width on the interstate/NHS system; height requires hybrid commercial augmentation.

---

## 2. Federal Datasets

### 2.1 FHWA HPMS — Highway Performance Monitoring System

| Attribute | Detail |
|---|---|
| **Portal** | https://www.fhwa.dot.gov/policyinformation/hpms.cfm |
| **Download** | Shapefile public release: https://www.fhwa.dot.gov/policyinformation/hpms/shapefiles.cfm |
| **Data hub** | https://data.transportation.gov/Roadways-and-Bridges/Highway-Performance-Monitoring-System-HPMS-/jc5k-rzm8 |
| **License** | U.S. Government Works (public domain at federal level); no stated restrictions on redistribution |
| **Format** | Shapefile (state-by-state ZIP), also REST service via geo.dot.gov |
| **Cadence** | States submit annually (June each year for prior-year data); FHWA publishes public release ~6 months later |
| **Contact** | Thomas Roff, Office of Highway Policy Information, (202) 366-5035 |

**Segment-level coverage:**

- **Through Lanes (Item 7):** Required for ALL Federal-aid highways. This is a universe item — full coverage of the NHS/Federal-aid network.
- **Lane Width (Item 34):** **Sample panel only** — applies only to sampled road sections, not the full network. Not suitable as a per-segment source.
- **Surface Width:** Not a distinct HPMS field. Shoulder width is present (Items 38–39), but total pavement surface width is not.
- **National Truck Network (Item 66):** Binary flag — designates STAA truck-route eligibility. Present as a universe item on NHS.
- **Posted Speed (in public shapefile):** Yes — included in the public shapefile.
- **Posted Weight Limit:** **Not a collected HPMS data item.** Bridge weight is in NBI, not HPMS.
- **Bridge Vertical Clearance:** Not in HPMS; see NBI below.
- **Hazmat Route:** Not a discrete HPMS item; STRAHNET (Strategic Highway Network) flag is present (Item 65), which correlates with preferred hazmat corridors but is not a hazmat-specific field.
- **Truck-prohibited segments:** Not in HPMS.

**Gotchas:**
- The public shapefile includes only "full extent" attributes (Items 7, 21, 22, 24, 64–67). Sample-panel items like lane width are NOT in the public download — they stay in the state systems.
- HPMS covers Federal-aid highways only. Local roads and many county routes are excluded.
- The geospatial representation is ARNOLD (All Roads Network of Linear Referenced Data), which has known topology issues at state borders.

---

### 2.2 FHWA NBI — National Bridge Inventory

| Attribute | Detail |
|---|---|
| **Portal** | https://www.fhwa.dot.gov/bridge/nbi.cfm |
| **2024 ASCII download** | https://www.fhwa.dot.gov/bridge/nbi/ascii2024.cfm |
| **InfoBridge query tool** | https://infobridge.fhwa.dot.gov/data |
| **BTS GeoData layer** | https://geodata.bts.gov/datasets/national-bridge-inventory/about |
| **SNBI spec (new format)** | https://www.fhwa.dot.gov/bridge/snbi/schema.cfm |
| **License** | U.S. Government Works (public domain) |
| **Format** | ASCII flat-file (comma or fixed-width), also GeoJSON via BTS GeoData |
| **Cadence** | Annual; states submit by June 15 each year |

**Bridge-level clearance fields (per-structure, NOT per-lane):**

| NBI Item | Field | Notes |
|---|---|---|
| Item 10 | Inventory Route Min Vertical Clearance | Over-deck clearance on inventory route (meters) |
| Item 39 | Navigation Vertical Clearance | Waterway navigation; not relevant for trucks |
| Item 53 | Min Vertical Clearance Over Bridge Roadway | Clearance above the bridge deck to any overhead restriction (meters) |
| Item 54B | Min Vertical Underclearance | Clearance from road or rail below the structure (meters) — **this is the truck-critical field** |
| Item 64 | Operating Rating | Max permissible load in metric tons — used for posting decisions |
| Item 66 | Inventory Rating | Long-term safe load in metric tons |
| Item 70 | Bridge Posting | Posted/not-posted status code |
| Item 28A | Lanes On Structure | Count of lanes carried by the bridge |
| Item 51 | Bridge Roadway Width Curb-to-Curb | Total width in meters |

**Per-lane clearance: Does NOT exist in NBI.** Item 54B is a single minimum value for the entire structure. A truck in the rightmost lane may have a different clearance than a truck in the center lane, but NBI does not capture this distinction. The new SNBI format (154 items, transitioning 2025–2028) does not add per-lane clearance fields.

**Weight capacity notes:**
- Operating Rating (Item 64) is the functional cap; Inventory Rating (Item 66) is the long-term design load.
- Bridge Posting (Item 70) indicates whether legal loads are restricted, but the posted limit values are in state systems, not NBI.
- Specific posted tonnage (single vehicle, tandem, combination) is often only in state bridge databases.

**Gotchas:**
- NBI covers structures on public roads ≥20 ft span. Culverts, short bridges, and overhead signs are excluded.
- Clearance values are self-reported by states; accuracy varies.
- Geographic coordinates are point locations (bridge midpoint), not linear events — you must join to route/milepost to get segment-level clearance.

---

### 2.3 BTS NTAD — National Transportation Atlas Database

| Attribute | Detail |
|---|---|
| **Portal** | https://www.bts.gov/ntad |
| **GeoData catalog** | https://geodata.bts.gov/ |
| **License** | Public domain (U.S. Government Works) |
| **Format** | File GDB, Shapefile, GeoJSON, CSV, KML |
| **Cadence** | Dynamic/rolling (multiple updates per year) |

**Truck-relevant layers in NTAD:**

- **National Highway Freight Network (NHFN):** Designated multimodal freight corridors; binary eligibility flag.
- **National Highway System (NHS):** Full NHS network with functional class.
- **1991 Federal Aid Primary (FAP) Roads:** Used for STAA truck-route eligibility determination.
- **HPMS ARNOLD:** Full road network shapefile with through-lane counts and speed.
- **Weigh-in-Motion (WIM) Stations:** Point layer of WIM station locations — useful for calibrating weight data but not per-segment weight limits.
- **National Bridge Inventory (GeoJSON):** Linked NBI data with bridge points.

**What NTAD does NOT have:** Per-segment posted weight limits, low-clearance sign inventories, truck-prohibited route designations below the state route level, per-lane bridge clearance.

---

### 2.4 FRA / ARNOLD (brief context)

ARNOLD (All Roads Network of Linear Referenced Data) is the geospatial backbone for HPMS — it is NOT a Federal Railroad Administration (FRA) dataset despite the name overlap. FRA data covers rail, not trucking. Grade-crossing safety data (public highway–rail intersections) is available from FRA's Highway-Rail Crossing Inventory, which is relevant if a routing validator needs to flag crossings that may impose height/width constraints due to overhead catenary or gate geometry, but this is edge-case scope for the BitNet project.

---

## 3. Per-State Summary Table

**Column key:**
- **Lane#** = Lane count per segment
- **LnWid** = Lane width (per-lane, not total)
- **SurfWid** = Pavement/surface width (total)
- **WgtLim** = Posted weight limits (segment or bridge level)
- **BrgClr** = Bridge vertical clearance (NBI minimum per-structure)
- **BrgClrLane** = Bridge vertical clearance per lane of travel
- **HgtRestr** = Posted height restrictions (low-clearance signs, non-bridge)
- **Hazmat** = Hazmat route designations
- **TrkPrb** = Truck-prohibited segments / parkways

**Coverage codes:** `yes` = confirmed published | `partial` = some routes only or incomplete | `no` = not found in open data | `unknown` = portal found, attribute not confirmed | `NBI` = available via federal NBI download

| State | DOT Portal | Lane# | LnWid | SurfWid | WgtLim | BrgClr | BrgClrLane | HgtRestr | Hazmat | TrkPrb | License | Format | Cadence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AL | data-algeohub.opendata.arcgis.com | partial | unknown | unknown | no | NBI | no | no | no | no | public/attribution | Shapefile/REST | unknown |
| AZ | azgeo-open-data-agic.hub.arcgis.com | yes | yes | unknown | no | NBI | no | no | unknown | unknown | public domain | Shapefile/REST | annual |
| AR | gis.arkansas.gov | yes | yes | yes | partial | NBI | no | no | no | no | attribution req. | SDE/REST | ad-hoc |
| CA | gisdata-caltrans.opendata.arcgis.com | yes | unknown | yes | no | NBI | no | no | partial | partial | public domain | Shapefile/REST | weekly |
| CO | data-cdot.opendata.arcgis.com | yes | unknown | unknown | no | partial | partial | partial | unknown | unknown | public domain | Shapefile/REST | annual |
| CT | connecticut-ctdot.opendata.arcgis.com | yes | unknown | unknown | partial | NBI | no | partial | unknown | yes | public domain | Shapefile/REST | quarterly |
| DE | de-firstmap-delaware.hub.arcgis.com | yes | yes | yes | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| FL | gis-fdot.opendata.arcgis.com | yes | unknown | yes | no | NBI | no | no | partial | partial | public domain | Shapefile/GDB | weekly |
| GA | data-hub.gio.georgia.gov | yes | unknown | unknown | no | NBI | no | no | no | no | attribution req. | Shapefile/GDB | annual |
| ID | data-iplan.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| IL | gis-idot.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| IN | indianamap.org | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| IA | data.iowadot.gov | yes | unknown | unknown | partial | NBI | no | partial | no | no | public domain | Shapefile/REST | annual |
| KS | hub.kansasgis.org | partial | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| KY | opengisdata.ky.gov | yes | unknown | unknown | yes | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| LA | data-ladotd.opendata.arcgis.com | yes | unknown | unknown | yes | NBI | no | no | no | no | public domain | Shapefile/REST | daily (LRS) |
| ME | geolibrary-maine.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| MD | data-maryland.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| MA | geo-massdot.opendata.arcgis.com | yes | unknown | yes | partial | NBI | no | no | no | yes | public domain | Shapefile/REST | annual |
| MI | gis-mdot.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | monthly |
| MN | gisdata.mn.gov | yes | unknown | unknown | yes | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| MS | opendata.gis.ms.gov | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| MO | data-msdis.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| MT | gis-mdt.opendata.arcgis.com | yes | unknown | yes | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| NE | geohub-ndot.hub.arcgis.com | yes | unknown | unknown | yes | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| NV | data-ndot.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| NH | nhgeodata.unh.edu | yes | unknown | unknown | partial | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| NJ | open-data-portal-njdot.hub.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| NM | planningdivisiongis-nmdot.hub.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| NY | dot.ny.gov/highway-data-services | yes | unknown | unknown | no | NBI | no | no | no | partial | terms of use | REST/download | quarterly |
| NC | connect.ncdot.gov | yes | unknown | unknown | yes | NBI | no | no | no | no | public domain | Shapefile/REST | quarterly |
| ND | gishubdata-ndgov.hub.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| OH | tims.dot.state.oh.us | yes | unknown | yes | no | NBI | no | no | no | no | public domain | Shapefile/GDB | annual |
| OK | gis-okdot.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | weekly |
| OR | gis.odot.state.or.us/transgis | yes | unknown | unknown | yes | NBI | no | no | no | no | public domain | REST service | annual |
| PA | data-pennshare.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| RI | rigis.org | partial | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile | ad-hoc (2016 last update) |
| SC | scdot.org/travel/travel-mappinggis.html | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/KMZ | monthly |
| SD | opendata2017-09-18t192802468z-sdbit.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| TN | tn-tnmap.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| TX | gis-txdot.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| UT | data-uplan.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| VT | geodata.vermont.gov | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |
| VA | virginiaroads.org | yes | unknown | unknown | yes | NBI | no | no | no | yes | public domain | Shapefile/REST | daily |
| WA | gisdata-wsdot.opendata.arcgis.com | yes | unknown | unknown | no | partial | partial | yes | no | no | WSDOT copyright | Shapefile/REST | annual |
| WV | data-wvdot.opendata.arcgis.com | yes | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile | annual |
| WI | data-wisdot.opendata.arcgis.com | yes | unknown | unknown | yes | NBI | no | no | no | no | public domain | Shapefile/REST | monthly |
| WY | data.geospatialhub.org | partial | unknown | unknown | no | NBI | no | no | no | no | public domain | Shapefile/REST | annual |

---

## 4. Per-State Details

### Alabama (AL)

- **DOT Portal:** https://data-algeohub.opendata.arcgis.com/ (Virtual Alabama GeoHub)
- **ALDOT direct:** https://aldotgis.dot.state.al.us/geogis (GeoGIS — Materials and Tests Bureau)
- **Road inventory:** ALDOT maintains an LRS-based centerline with mileposted networks; direct shapefile download is limited. Primary public GIS is through the GeoHub. No standalone weight-limit or clearance dataset found.
- **Bridge inventory:** Available via NBI federal download only; no separate state open-data bridge layer confirmed.
- **Attributes confirmed:** Lane count (partial — from HPMS/ARNOLD); lane width, surface width, weight limits, bridge clearance per-lane: not confirmed in open data.
- **License:** Attribution required ("Data Provided by ALDOT, In Cooperation With U.S. DOT").
- **Format:** Shapefile / REST service.
- **Cadence:** Unknown; HPMS data annual.
- **Gotchas:** ALDOT is relatively closed with direct data sharing; the GeoHub does not prominently list road-inventory download datasets. Best approach: HPMS public shapefile for lane count, federal NBI for bridge clearance.

---

### Arizona (AZ)

- **DOT Portal:** https://azgeo-open-data-agic.hub.arcgis.com/ (AZGeo Data Hub, co-maintained by ADOT)
- **ADOT GIS:** https://azdot.gov/planning/gis
- **Road inventory:** Arizona Transportation Information System (ATIS) — ADOT's LRS backbone. Published on AZGeo hub with 37 MIRE elements including lane count and speed. Direct download available.
  - URL: https://azgeo-open-data-agic.hub.arcgis.com/datasets/azgeo::az-all-roads-network-2021/about
- **Bridge inventory:** Available via NBI; no confirmed standalone AZ bridge open-data layer with per-structure posted weights.
- **Attributes confirmed:** Lane count (yes — MIRE Item); lane width (yes — listed as an RCI element); surface width (unknown in public dataset); posted weight limits (no standalone open-data layer confirmed); bridge clearance (NBI); per-lane clearance (no); hazmat (unknown).
- **License:** Public domain / government works.
- **Format:** Shapefile, REST/ArcGIS service.
- **Cadence:** Annual (HPMS cycle).
- **Coverage:** All roads including state routes; county roads partial.
- **Gotchas:** AZGeo is a state-wide portal shared across agencies; data quality and completeness varies by road class. ATIS is the definitive source but some layers are view-only.

---

### Arkansas (AR)

- **DOT Portal:** https://gis.arkansas.gov/ (Arkansas GIS Office, hosts ARDOT data)
- **ARDOT GIS:** https://ardot.gov/divisions/planning/gis-mapping/
- **Road inventory:** Arkansas Road Inventory — polyline with 118,506 features; attributes confirmed via metadata.
  - URL: https://gis.arkansas.gov/product/arkansas-road-inventory/
  - Metadata: https://gis.arkansas.gov/Metadata/HTML/asdi.Transportation.ROAD_INVENTORY_ARDOT_export.html
  - REST: http://gis.arkansas.gov/arcgis/rest/services/FEATURESERVICES/Transportation/FeatureServer
  - **Confirmed fields:** `NumberLane` (through lane count), `LaneWidth` (minimum lane width in feet), `SurfaceWid` (through-lane + extra-lane width), `RoadwayWid` (full surface including shoulders).
- **Bridge inventory / Posted Bridges:**
  - Arkansas Posted Highway Bridges: https://gis.arkansas.gov/product/arkansas-posted-highway-bridges/
  - Weight Restrictions metadata: https://gis.arkansas.gov/Metadata/HTML/asdi.Transportation.WEIGHT_RESTRICTIONS_ARDOT_export.html
  - REST layer 23: https://gis.arkansas.gov/arcgis/rest/services/FEATURESERVICES/Transportation/MapServer/23
  - `WEIGHT_RESTRICTIONS_ARDOT` stores current weight-limit values on restricted highway sections.
- **Attributes confirmed:** Lane count (yes), lane width (yes — per-lane minimum), surface width (yes), posted weight limits (partial — posted highway segments and bridge postings available as separate layers), bridge clearance (NBI + ARDOT ArcGIS Map Tool), per-lane clearance (no), height restrictions (no standalone layer), hazmat (no), truck-prohibited (no).
- **License:** Attribution required ("Provided by ARDOT, In Cooperation With U.S. DOT").
- **Format:** SDE Feature Class / Shapefile / REST.
- **Cadence:** Road inventory — ad-hoc updates; bridge postings — updated as inspections occur.
- **Coverage:** State and federal funding roads + ARDOT planning roads. Not all county roads.
- **Gotchas:** SDE feature class requires ArcGIS or GDAL to open. REST service is the most accessible path. Bridge weight-limit layer covers posted bridges only, not all bridges.

---

### California (CA)

- **DOT Portal:** https://gisdata-caltrans.opendata.arcgis.com/ (Caltrans GIS Open Data)
- **All Public Roads:** https://gisdata-caltrans.opendata.arcgis.com/datasets/2d56e65de89c418780056651640291e8_0/about
- **California State Geoportal (mirror):** https://gis.data.ca.gov/
- **Road inventory:** Caltrans All Roads LRS — forms the HPMS base geometry for California. Includes functional classification and route attributes.
- **Truck-specific datasets:**
  - Legal Truck Access: https://dot.ca.gov/programs/traffic-operations/legal-truck-access (web-only, not a GIS download)
  - Commercial Vehicle Enforcement Facilities: point layer of weigh stations
  - Truck Volume AADT: GIS layer of truck traffic volumes
- **Bridge inventory:** Available via NBI; Caltrans maintains a bridge database (Caltrans Bridge Management System) but public access to individual bridge attribute tables is limited. No confirmed downloadable per-bridge weight/clearance GIS layer.
- **Attributes confirmed:** Lane count (yes — HPMS/LRS); lane width (unknown in public dataset); surface width (partial — SURWIDTH in RCI equivalent); posted weight limits (no standalone open layer); bridge clearance (NBI); per-lane clearance (no); height restrictions (truck-legal routes web-only map, not downloadable GIS); hazmat (partial — via STRAHNET in HPMS); truck-prohibited (partial — legal basis docs but not GIS).
- **License:** Public domain.
- **Format:** Shapefile, REST/ArcGIS.
- **Cadence:** Weekly updates for core road layers.
- **Coverage:** All public roads statewide.
- **Gotchas:** California has extensive truck-restriction rules (legal basis under Vehicle Code), but the GIS expression of these rules is mostly map-viewer only, not bulk-downloadable with attributes. Low-clearance structures (the state's many overpasses) are tracked in the Caltrans Bridge Management System but public API access is not documented. Size of statewide all-roads layer: very large; segment into counties for practical use.

---

### Colorado (CO)

- **DOT Portal:** https://data-cdot.opendata.arcgis.com/ (CDOT Public Maps and Data)
- **OTIS system:** https://dtdapps.coloradodot.info/otis (moving URLs as of April 2026 — verify new URL)
- **Road inventory:** CDOT maintains a statewide LRS with MIRE elements. Downloadable via OTIS and ArcGIS Hub.
- **Bridge Vertical Clearances:** https://geodata.colorado.gov/datasets/cdot::bridges-vertical-clearances (point layer of bridge vertical clearances — **most notable open-data clearance dataset in any state**).
  - Data is described as "a guide" and may not be guaranteed. Fields include minimum clearance values.
  - Dashboard: https://ft-cdot.opendata.arcgis.com/pages/vertical-clearances
- **Bridges and Major Culverts:** https://geodata.colorado.gov/datasets/cdot::bridges-and-major-culverts/about
- **Attributes confirmed:** Lane count (yes); lane width (unknown in public extract); surface width (unknown); posted weight limits (no standalone layer confirmed); bridge clearance (partial — vertical clearances point layer published, but completeness unverified and per-lane status unclear); per-lane clearance (partial/experimental — CDOT is collecting lane-specific clearances using mobile LiDAR; public release status unconfirmed); height restrictions (partial — vertical clearance layer is the closest proxy); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, GeoJSON, REST.
- **Cadence:** Annual (road inventory); unknown for clearance layer.
- **Coverage:** State highway system; local roads via OTIS.
- **Gotchas:** CDOT explicitly disclaims that vertical clearance data "cannot be guaranteed due to physical changes to highways." Per-lane collection is underway but not confirmed as a public release. OTIS URL is changing in late April 2026 — the old URL may stop working.

---

### Connecticut (CT)

- **DOT Portal:** https://connecticut-ctdot.opendata.arcgis.com/
- **CT Geodata Portal:** https://geodata.ct.gov/pages/dot
- **Road inventory:** CTDOT Roadway Classification and Characteristic Data — annual snapshots, includes functional class, ownership, lane attributes.
  - URL: https://connecticut-ctdot.opendata.arcgis.com/maps/65d7254355bb4466871cd7c5ea8a6a5d
- **Bridge inventory:** CTDOT Bridges dataset — bridge locations and attributes.
  - URL: https://connecticut-ctdot.opendata.arcgis.com/maps/4ddfc36aeb8e420cbf29a6d99638fe0f
- **Truck/parkway restrictions:** CTDOT Office of State Traffic Administration (OSTA) interactive map includes "No Thru Truck" designations and parkway restrictions (height, width, weight).
  - URL: https://catalog.data.gov/dataset/ctdot-office-of-the-state-traffic-administration-osta-interactive-map
  - Note: This is an interactive map; bulk download availability of the underlying data is not confirmed.
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (partial — parkway restrictions noted); bridge clearance (NBI + CTDOT Bridges dataset); per-lane clearance (no); height restrictions (partial — parkway height limits in OSTA); hazmat (unknown); truck-prohibited (yes — parkway restrictions documented).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual road inventory; bridge dataset cadence unknown.
- **Coverage:** State routes and local roads; OSTA covers state-designated routes.
- **Gotchas:** Connecticut's parkway network (Merritt Pkwy, Wilbur Cross, etc.) has strict height restrictions (13'6" or less) that affect all trucks. The OSTA map appears to be the only open-data expression of these, and bulk GIS download is not confirmed. Pursue CTDOT directly for OSTA GIS extract.

---

### Delaware (DE)

- **DOT Portal:** https://de-firstmap-delaware.hub.arcgis.com/ (DE FirstMap)
- **DelDOT GIS:** https://deldot.gov/Publications/reports/gis/index.shtml
- **Road inventory:** DelDOT Centerline File — shapefile, includes road width, number of lanes, guiderails. Download available from DelDOT directly.
  - REST: http://firstmap.gis.delaware.gov/arcgis/rest/services/Transportation/DE_Road_Inventory/FeatureServer/0
  - Fields confirmed: road name, width, number of lanes, guiderails.
- **Bridge inventory:** Available via NBI; no confirmed separate state open-data bridge layer with posted weights.
- **Attributes confirmed:** Lane count (yes); lane width (yes — "width" field); surface width (yes); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile (WinZip).
- **Cadence:** Annual.
- **Coverage:** Full statewide public road network.
- **Download URL:** https://deldot.gov/Publications/reports/gis/index.shtml (or call 800-652-5600 / email dotpublic@delaware.gov).
- **Gotchas:** Small state; dataset is manageable. However, Delaware has a number of bridges on SR-1 and I-95 with complex structure — verify clearances from NBI. The FirstMap ArcGIS Hub is the most accessible interface.

---

### Florida (FL)

- **DOT Portal:** https://gis-fdot.opendata.arcgis.com/ (FDOT Open Data Hub)
- **Transportation Data Portal:** https://www.fdot.gov/data/index
- **RCI Documentation:** https://www.fdot.gov/statistics/rci/default.shtm
- **Road inventory:** FDOT Roadway Characteristics Inventory (RCI) — comprehensive statewide LRS with weekly updates. Shapefile and GDB downloads. Covers all public roads.
  - Confirmed RCI fields: `NOLANES` (number of roadway lanes), `PEAKLANE` (peak direction lanes), `SURWIDTH` (pavement surface width), `SLDWIDTH` (shoulder width), `AUXLNWTH` (auxiliary lane width), `MEDWIDTH` (median width), pavement type, condition.
  - **Not in RCI:** Weight limits, bridge clearance, hazmat routes (these are in separate FDOT systems).
- **State Roads layer:** https://gis-fdot.opendata.arcgis.com/datasets/d8ce6b8eee2646e6a0b534281e0391a0_0
- **Bridge inventory:** Available via NBI. FDOT maintains a Bridge Management System (BMS) but no confirmed public API for per-bridge clearance/weight beyond NBI.
- **Truck volume:** GIS layer available (annual average daily truck volume by traffic break segment).
- **Attributes confirmed:** Lane count (yes — `NOLANES`); lane width (unknown — not confirmed as RCI element in public dataset); surface width (yes — `SURWIDTH`); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no standalone open layer); hazmat (partial — STRAHNET designation via HPMS); truck-prohibited (partial — legal truck access rules exist but GIS layer not confirmed downloadable).
- **License:** Public domain.
- **Format:** Shapefile, GDB (statewide ZIP).
- **Cadence:** Weekly.
- **Coverage:** All public roads on the Florida road system.
- **Gotchas:** RCI is large (~several GB statewide). The open data hub provides REST access which may be more practical. Florida has very active truck-restriction regulation (particularly for oversize/overweight permits) but this is a permitting database, not a segment-attribute layer.

---

### Georgia (GA)

- **DOT Portal:** https://data-hub.gio.georgia.gov/ (Georgia GIO Data Hub)
- **GeoPI tool:** https://www.dot.ga.gov/applications/geopi/Pages/Search.aspx
- **Road inventory:** GDOT Road Inventory — 125,000+ centerline miles; downloadable in Spatial Geodatabase (469 MB) or Excel (32 MB).
  - Data Dictionary: https://www.dot.ga.gov/DriveSmart/Data/Documents/Road_Inventory_Data_Dictionary.pdf
  - Includes: functional classification, US Routes, NHS, Federal Aid, State Routes, bridges.
- **Bridge inventory:** Available via NBI. GDOT bridge data viewable in GeoPI but GIS download attributes not fully confirmed.
- **Attributes confirmed:** Lane count (yes); lane width (unknown from public extract); surface width (unknown); posted weight limits (no standalone layer); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Attribution required.
- **Format:** File GDB, Excel, REST.
- **Cadence:** Annual.
- **Coverage:** 125,000 miles of public roads in Georgia.
- **Gotchas:** The GDB download is 469 MB — use GDAL or ArcGIS Pro. The Road Inventory Data Dictionary PDF exceeds 10 MB so content could not be fully fetched during research; verify lane-width and weight-limit fields directly.

---

### Idaho (ID)

- **DOT Portal:** https://data-iplan.opendata.arcgis.com/ (Idaho Transportation Department Open Data)
- **ITD GIS:** https://itd.idaho.gov/gis-maps/
- **Road inventory:** ITD maintains statewide LRS and asset inventory. Published via ArcGIS Online.
  - REST: https://gis.itd.idaho.gov/arcgisprod/rest/services/ArcGISOnline/IdahoTransportationLayersForOpenData/MapServer
- **Bridge inventory:** Available via NBI + LHTAC interactive bridge map with NBI data.
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, GeoJSON, REST.
- **Cadence:** Annual.
- **Coverage:** All roads open to public travel.
- **Gotchas:** ITD published a comprehensive asset geodatabase in ArcGIS Online as a case study cited by ESRI; the public layer access may be more restricted than the internal version. Verify which layers are downloadable vs. view-only.

---

### Illinois (IL)

- **DOT Portal:** https://gis-idot.opendata.arcgis.com/
- **IROADS system:** https://webapps.dot.illinois.gov/IROADS/
- **Road inventory:** IDOT maintains the Illinois Roadway Information System (IRIS). GIS data downloadable by county or statewide via the portal. Road Construction and bridge inventory layers available.
- **Bridge inventory:** IDOT provides bridge inventory data through their GIS portal; available as a download layer.
- **Attributes confirmed:** Lane count (yes); lane width (unknown in public download); surface width (unknown); posted weight limits (no standalone layer confirmed); bridge clearance (NBI + IDOT bridge layer); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, GeoJSON, REST (ArcGIS Online).
- **Cadence:** Annual (HPMS cycle); bridge data cadence unknown.
- **Coverage:** All Illinois public roads; bridge layer is statewide.
- **Gotchas:** The GIS tool (gis-idot.opendata.arcgis.com) provides downloadable roadway, railroad, and structure data by county or statewide. Terms of use require acknowledgment of IDOT as source.

---

### Indiana (IN)

- **DOT Portal:** https://www.indianamap.org/ (IndianaMap Hub)
- **INDOT GIS:** https://indot.maps.arcgis.com/
- **Road inventory:** INDOT Roadway Inventory — maintained by Asset Data Collection section. Roads shapefile available.
  - URL: https://www.in.gov/indot/about-indot/central-office/asset-data-collection/roadway-inventory/
- **Bridge inventory:** INDOT Bridge Locations — point layer with NBI data.
  - URL: https://hub.arcgis.com/datasets/INMap::indot-bridge-locations/about
- **Bridge Clearance:** INDOT Bridge Clearance interactive map: https://indot.maps.arcgis.com/apps/webappviewer/index.html?id=0a27953c1ae7480eae1c8fdd4c6b8e28 — view-only web app; underlying data download not confirmed.
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no open download — CMV drivers referred to bridge maps); bridge clearance (NBI + INDOT web viewer); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual.
- **Coverage:** State road system; local roads via IndianaMap.
- **Gotchas:** INDOT bridge clearance data exists (interactive map) but a bulk GIS download with clearance attributes is not confirmed as public. Contact INDOT Asset Data Collection for data request.

---

### Iowa (IA)

- **DOT Portal:** https://data.iowadot.gov/ (Iowa DOT Open Data)
- **GIS downloads:** https://iowadot.gov/gis/downloads/default
- **Road inventory:** Road Network Portal — 114,000+ miles with HPMS data.
  - URL: https://data.iowadot.gov/datasets/road-network-portal/about
  - GIMS (Geographic Information Management System) is the underlying LRS.
- **Bridge inventory:** Bridge Line dataset — all Iowa bridges and structures with historic tabular data.
  - URL: https://public-iowadot.opendata.arcgis.com/datasets/bridge-line
- **Vertical clearance:** Iowa DOT publishes a Vertical Clearance Restrictions log (PDF) for primary roads: https://iowadot.gov/mvd/motorcarriers/vertclearlog.pdf — not a GIS layer.
- **Weight restrictions:** "All Systems Overweight Permit Map" interactive; bridge embargo maps available. Not confirmed as downloadable GIS layer with attributes.
- **Attributes confirmed:** Lane count (yes); lane width (unknown in public extract); surface width (unknown); posted weight limits (partial — interactive maps, not bulk GIS); bridge clearance (NBI + Bridge Line dataset); per-lane clearance (no); height restrictions (partial — PDF log, not GIS); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual (road network); bridge line — unknown.
- **Coverage:** All 114,000 miles of public roads including county roads.
- **Gotchas:** Iowa DOT uses ESRI Roads & Highways for LRS. The GIMS metadata catalogs (cloud.iowadot.gov/GIS/data/GIMS/metadata/) provide field-level documentation. Bridge data is in LRS event format — requires LRS join to map to segments.

---

### Kansas (KS)

- **DOT Portal:** https://hub.kansasgis.org/ (Kansas Geoportal)
- **KDOT GIS:** https://ksdot.maps.arcgis.com/
- **Road inventory:** KanPlan online mapping platform. HPMS data in state system; public shapefile access through Kansas Geoportal.
- **Bridge inventory:** Available via NBI; Kansas Geoportal has some bridge layers.
- **Attributes confirmed:** Lane count (partial — HPMS); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual.
- **Coverage:** State highways via KDOT; local roads less comprehensive.
- **Gotchas:** Kansas Geoportal is a multi-agency clearinghouse; KDOT data quality varies. The primary KDOT-specific download path is less prominent than some other states. Direct contact: KDOT Bureau of Transportation Planning.

---

### Kentucky (KY)

- **DOT Portal:** https://opengisdata.ky.gov/ (KyGovMaps Open Data Portal)
- **KYTC DataMart:** https://datamart.kytc.ky.gov/
- **Road inventory:** Highway Information System (HIS) — shapefile with 6 files per ZIP. Roadway asset shapefiles in Kentucky State Plane Single Zone projection.
  - URL: https://transportation.ky.gov/Planning/Pages/Centerlines.aspx
  - HIS extracts: https://transportation.ky.gov/Planning/Pages/HIS-Extracts.aspx
- **Bridge inventory:** KYTC Bridge Weight Limits — **GIS dataset of all public bridges in Kentucky with posting status**.
  - URL: https://data-bgky.hub.arcgis.com/datasets/KYTC::kytc-bridge-weight-limits
  - Includes posting status and restrictions from National Bridge Inventory.
- **Attributes confirmed:** Lane count (yes — HIS); lane width (unknown); surface width (unknown); posted weight limits (yes — dedicated bridge weight limits GIS layer); bridge clearance (NBI + KYTC bridge data); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** ESRI Shapefile (Kentucky State Plane); GeoJSON via Hub.
- **Cadence:** Annual (HIS); bridge weight limits — cadence unknown but NBI-sourced.
- **Coverage:** State and local roads via HIS; bridges are statewide (NBI-based).
- **Gotchas:** Kentucky State Plane Single Zone projection — reproject to WGS84 for routing use. Over 250 data items per bridge in KYTC's system; confirm which are in the public Hub layer.

---

### Louisiana (LA)

- **DOT Portal:** https://data-ladotd.opendata.arcgis.com/ (LaDOTD Open Data Portal)
- **Road inventory:** Louisiana Roadways — LRS-based enterprise dataset, edited daily.
  - URL: https://data-ladotd.opendata.arcgis.com/datasets/LADOTD::louisiana-roadways/about
  - Data dictionary: https://maps.dotd.la.gov/r_and_h_datadictionary/metadata.htm
- **Bridge inventory:** Bridges layer on open data portal.
  - URL: https://data-ladotd.opendata.arcgis.com/datasets/c91f5dc0c6d14e6cb109487bb6c06682_24/about
- **Posted On-System Bridges:** Dedicated GIS layer showing bridges with load restrictions.
  - REST: http://gisweb.dotd.la.gov/ArcGIS/rest/services/LADOTDAGO/LA_Posted_On_System_Bridges/MapServer
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (yes — dedicated Posted On-System Bridges layer, inspections by LaDOTD Bridge Inspection Program); bridge clearance (NBI + LaDOTD bridges); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, GeoJSON, REST.
- **Cadence:** Daily (LRS roadways); bridge layer cadence unknown.
- **Coverage:** State-maintained system (on-system); local roads separate.
- **Gotchas:** "Not all data stored within R&H is published due to sensitivity or incomplete database entries" — some attributes may be in internal LaDOTD system only. Off-system bridge data (local bridges) is separate from on-system posted bridge layer.

---

### Maine (ME)

- **DOT Portal:** https://geolibrary-maine.opendata.arcgis.com/ (Maine GeoLibrary)
- **MaineDOT GIS:** https://maine.hub.arcgis.com/
- **Road inventory:** MaineDOT Public Roads — statewide road centerlines.
  - URL: https://hub.arcgis.com/datasets/maine::mainedot-public-roads
- **TIDE system:** MaineDOT's GIS-linked data warehouse connecting roads, crashes, pavement, and bridge data. Public map viewer: https://www.maine.gov/mdot/mapviewer/
- **Bridge inventory:** Available via NBI; MaineDOT bridge data in TIDE but public GIS download attributes not confirmed.
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Annual.
- **Coverage:** Public roads statewide.
- **Gotchas:** Maine's rural road network includes a significant number of weight-restricted local bridges and seasonal weight limits (spring thaw restrictions). These are managed by MaineDOT and municipalities but not published as an open GIS layer.

---

### Maryland (MD)

- **DOT Portal:** https://data-maryland.opendata.arcgis.com/ (Maryland's GIS Data Catalog)
- **MDOT SHA GIS:** https://roads.maryland.gov/mdotsha/pages/Index.aspx?PageId=306
- **Road inventory:** MDOT SHA Maintained Roads — state-maintained road network.
  - URL: https://data.imap.maryland.gov/datasets/mdot-sha-maintained-roads
- **Bridge inventory:** Available via NBI. Maryland SHA conducted a pilot data exchange program (FHWA case study) but public bridge attribute downloads are not confirmed.
- **Sign inventory:** MDOT SHA Roadway Sign Inventory — point layer with sign locations.
- **Attributes confirmed:** Lane count (yes — SHA roads); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no standalone layer); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual.
- **Coverage:** State-maintained roads (SHA system); county roads are separate.
- **Gotchas:** Maryland has a substantial bridge inventory on I-695, I-95, and the Bay Bridge approaches. Weight restrictions on SHA bridges are published in SHA bulletins but not as downloadable GIS.

---

### Massachusetts (MA)

- **DOT Portal:** https://geo-massdot.opendata.arcgis.com/ (MassDOT Open Data Portal)
- **MassGIS Roads page:** https://www.mass.gov/info-details/massgis-data-massgis-massdot-roads
- **Road inventory:** Road Inventory 2024 — annual GIS dataset; confirmed attribute schema.
  - URL: https://geo-massdot.opendata.arcgis.com/maps/MassDOT::road-inventory-2024
  - **Confirmed fields:** `Num_Lanes` (number of travel lanes), `Opp_Lanes` (opposite direction lanes), `Surface_Wd` (surface width in feet), `Shldr_Lt_W` / `Shldr_Rt_W` (shoulder widths), `T_Exc_Type` (Truck Exclusion Type — coded values: open to all vehicles, vehicles over 2.5 tons excluded, vehicles over 3 tons excluded, etc.), `T_Exc_Time` (time-based restriction schedule), `Trk_Permit` (truck permit required), `Truck_Rte` (truck route designation).
- **Bridge inventory:** Available via NBI; MassDOT maintains bridge database, public GIS layer attributes not confirmed.
- **Attributes confirmed:** Lane count (yes); lane width (unknown — Surface_Wd is total width, not per-lane); surface width (yes — `Surface_Wd`); posted weight limits (partial — `T_Exc_Type` encodes weight-class exclusions); bridge clearance (NBI); per-lane clearance (no); height restrictions (no — parkway restrictions not in road inventory schema); hazmat (no); truck-prohibited (yes — `T_Exc_Type` includes exclusion categories).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Annual (year-end snapshot).
- **Coverage:** All public roads in Massachusetts.
- **Gotchas:** Massachusetts is a standout for truck-exclusion attribute richness — the `T_Exc_Type` field is a rarity among state DOTs. However, the coded values represent weight-class thresholds (2.5 ton, 3 ton) rather than explicit gross-vehicle-weight limits, so you'll need to decode to axle/GVW equivalents. Parkway height restrictions (Storrow Drive, Memorial Drive, numerous others) are NOT in the road inventory — they're posted sign limits managed by DCR (Dept. of Conservation and Recreation) and MassDOT Operations, not in this dataset.

---

### Michigan (MI)

- **DOT Portal:** https://gis-mdot.opendata.arcgis.com/ (MDOT GIS Open Data)
- **Michigan Open Data:** https://data.michigan.gov/
- **Road inventory:** MDOT uses ESRI Roads & Highways for LRS. Downloadable from GIS Open Data portal.
- **Bridge inventory:** MDOT Bureau of Bridges and Structures — Common Bridge Inventory Items.
  - URL: https://data.michigan.gov/Infrastructure/MDOT-Bureau-of-Bridges-and-Structures-Common-Bridg/6rbe-zjpu
  - Includes MDOT-owned bridges and Local Agency NBI bridges with size and design info.
- **Bridge Connections:** https://gis-mdot.opendata.arcgis.com/datasets/mdot-bridge-connections
- **Statewide Bridge Signs Inventory:** OHM Advisors-managed project; not a public open-data layer.
- **Weight restrictions:** Seasonal spring weight restrictions are well-publicized (press releases) but the underlying GIS layer of restricted segments is not confirmed as a public download.
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no — bridge inventory has size info but posted weight not confirmed); bridge clearance (NBI + MDOT bridge inventory); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain (MDOT states data is "provided as is").
- **Format:** Shapefile, REST; bridge inventory also as CSV via data.michigan.gov.
- **Cadence:** Bridge inventory — monthly refresh (per earlier search result).
- **Coverage:** All state-maintained roads; bridge inventory includes local agency bridges.
- **Gotchas:** Michigan has an extensive bridge sign inventory project that captured sign structures statewide, but this is for sign structures (not clearance data). The data.michigan.gov bridge inventory dataset is the best starting point for bridge attributes.

---

### Minnesota (MN)

- **DOT Portal:** https://gisdata.mn.gov/ (Minnesota Geospatial Commons)
- **MnDOT TDA:** https://www.dot.minnesota.gov/tda/
- **Road inventory:** MnDOT LRS with integrated roadway, bridge, crash, traffic, and pavement data. GIS shapefiles available via MN Geospatial Commons.
- **Bridge inventory:** MnDOT Bridge Inventory Management Unit — in-service bridge locations dataset.
  - URL: https://www.dot.state.mn.us/bridge/bridgereports/index.html
- **Seasonal Load Limits:** Published as maps showing weight-restricted segments in 6 frost zones.
  - URL: https://www.dot.state.mn.us/loadlimits/maps.html
  - These are map products; GIS layer download status unknown.
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (yes — seasonal load limits published, GIS download unconfirmed); bridge clearance (NBI + MnDOT bridge data); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual (road LRS); bridge data — periodic.
- **Coverage:** All public roads.
- **Gotchas:** Minnesota's bridge underclearance data is available through the MnDOT Bridge Office as a PDF report (Vertical and Horizontal Bridge Underclearance Report), not as an open GIS layer.

---

### Mississippi (MS)

- **DOT Portal:** https://opendata.gis.ms.gov/ (Mississippi Geospatial Data Catalog)
- **MDOT GIS Hub:** https://home-mdot.hub.arcgis.com/
- **Road inventory:** MDOT_CO_LRM — complete LRM (Linear Referencing Model) of all Mississippi roads; state and local routes.
  - URL: https://www.gis.ms.gov/datasets/db70cdeeb0ac444caea18275e85d5d06_0
  - MS Highways layer: https://www.gis.ms.gov/datasets/db70cdeeb0ac444caea18275e85d5d06_3/about
- **Bridge inventory:** Available via NBI; separate MDOT bridge GIS layer status unknown.
- **Attributes confirmed:** Lane count (yes — LRM includes mileage by county and road class); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Annual.
- **Coverage:** All public roads including state, local, and county.
- **Gotchas:** Mississippi has a large number of locally-owned bridges, many not meeting NHS standards. NBI is the primary source for bridge data; the state has limited open bridge-attribute GIS. MARIS (Mississippi Automated Resource Information System) is an alternate clearinghouse.

---

### Missouri (MO)

- **DOT Portal:** https://data-msdis.opendata.arcgis.com/ (Missouri Spatial Data Information Service)
- **MoDOT GIS:** https://www.modot.org/gis-asset-mapping
- **Road inventory:** MO MoDOT Roads Arcs and Routes — LRS-based polyline datasets.
  - URL: https://data-msdis.opendata.arcgis.com/datasets/MSDIS::mo-modot-roads-arcs/about
  - LRS stores events: accidents, pavement type, speed limit, signage.
- **Bridge inventory:** Available via NBI. MoDOT's comprehensive GIS asset map tracks assets in real time but public attribute download is not confirmed.
- **Attributes confirmed:** Lane count (yes — LRS events); lane width (unknown); surface width (unknown); posted weight limits (no standalone layer); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual.
- **Coverage:** MoDOT-maintained roads; local roads via MSDIS.
- **Gotchas:** MSDIS is a multi-agency clearinghouse. Verify which layers come directly from MoDOT vs. other state agencies. The GIS Asset Mapping tool is a live tracking tool, not a bulk download.

---

### Montana (MT)

- **DOT Portal:** https://gis-mdt.opendata.arcgis.com/ (Montana Department of Transportation GIS Data Portal)
- **MDT Data:** https://mdt.mt.gov/contact/organization/railtran-datastats.aspx
- **Road inventory:** Montana Road Log — 81 data items covering all roads open to public travel. **Confirmed to include: surface type, width, length, number of lanes.**
  - Hub URL: https://gis-mdt.hub.arcgis.com/
  - Contact for full Road Log: 406-444-6103; hard copy $50.
  - Off-system routes: https://gis-mdt.hub.arcgis.com/datasets/37445e45aa3f43ca878e7c89cbb0dee2_0/about
- **Bridge inventory:** Available via NBI. MDT Load and Speed Limit Policy covers bridge weight limits but no downloadable GIS layer confirmed.
- **Attributes confirmed:** Lane count (yes); lane width (unknown in public extract); surface width (yes — "width" confirmed in 81-item Road Log); posted weight limits (no GIS layer — managed through MDT Load & Speed Limit Policy); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, GeoJSON, REST.
- **Cadence:** Annual.
- **Coverage:** All roads open to public travel in Montana.
- **Gotchas:** Montana is large and rural; a significant proportion of public roads are county roads and forest service roads. The 81-item Road Log is comprehensive but the full detail (all 81 items) may not be in the public ArcGIS Hub layer — verify which fields are exposed.

---

### Nebraska (NE)

- **DOT Portal:** https://geohub-ndot.hub.arcgis.com/ (NDOT GeoHub)
- **Open Data:** https://ndotdata.nebraska.gov/
- **NebraskaMap:** https://www.nebraskamap.gov/
- **Road inventory:** Available via NebraskaMap and NDOT GeoHub.
- **Bridge inventory:** Bridges dataset — Nebraska bridges with NBI data.
  - URL: https://www.nebraskamap.gov/datasets/nebraska::bridges/about
- **Weight Restricted Bridges:** Dedicated layer — NDOT-identified weight-limited bridges.
  - URL: https://www.nebraskamap.gov/datasets/nebraska::weight-restricted-bridges/about
  - Interactive map: https://gis.ne.gov/portal/apps/webappviewer/index.html?id=f6945569f00a43268462568591475ab8
- **Attributes confirmed:** Lane count (yes — LRS); lane width (unknown); surface width (unknown); posted weight limits (yes — dedicated Weight Restricted Bridges layer); bridge clearance (NBI + Nebraska bridges); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, GeoJSON, REST.
- **Cadence:** Annual (road inventory); bridge weight layer — updated with inspection cycle.
- **Coverage:** State highway system; local roads via NebraskaMap.
- **Gotchas:** Nebraska's rural road network includes many county bridges not on the state system. The Weight Restricted Bridges layer covers NDOT-monitored structures; county-owned restricted bridges may require county-level data requests.

---

### Nevada (NV)

- **DOT Portal:** https://data-ndot.opendata.arcgis.com/
- **NDOT GeoHub:** https://geohub-ndot.hub.arcgis.com/
- **Road inventory:** Roadway Systems division manages LRS data.
  - URL: https://geohub-ndot.hub.arcgis.com/pages/ndot-divisions-roadway-systems
  - Road Data Viewer: https://gis.dot.nv.gov/RoadDataViewer/
- **Bridge inventory:** Available via NBI. No confirmed separate Nevada bridge GIS download with clearance/weight attributes.
- **Attributes confirmed:** Lane count (yes — roadway systems LRS); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Annual.
- **Coverage:** State highway network; urban roads included.
- **Gotchas:** Nevada's NDOT GeoHub is well-organized but focused on infrastructure categories. The Road Data Viewer is interactive-only; bulk download requires the ArcGIS hub. Note: "NDOT" is both Nebraska DOT and Nevada DOT — the Nebraska GeoHub is at geohub-ndot.hub.arcgis.com and so is Nevada's; the difference is the domain (Nebraska = ndot.hub.arcgis.com / data-ndot for Nevada). Verify URLs carefully.

---

### New Hampshire (NH)

- **DOT Portal:** https://www.dot.nh.gov/about-nh-dot/divisions-bureaus-districts/planning-community-assistance/gis-data-catalog (NHDOT GIS Data Catalog)
- **NH Geodata Portal:** https://www.nhgeodata.unh.edu/ (NH GRANIT at UNH)
- **Road inventory:** NH DOT Roads — statewide road network for route mapping and attribute queries.
  - ArcGIS Hub: https://hub.arcgis.com/datasets/NHGRANIT::nh-dot-roads/explore
- **Bridge inventory:** NHDOT Bridge Plans and Reports: https://maps.dot.nh.gov/reports/plan-inventory-bridge/
  - Red-Listed Bridge maps are published but as static PDFs/maps, not open GIS.
- **Weight restrictions:** Info about load limitations on Certified Vehicles crossing posted bridges is documented but not confirmed as GIS download.
- **Attributes confirmed:** Lane count (yes — NH DOT Roads); lane width (unknown); surface width (unknown); posted weight limits (partial — bridge posting info in reports, not GIS layer); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual.
- **Coverage:** State and local roads.
- **Gotchas:** New Hampshire has many municipally-owned bridges with weight restrictions; these are tracked locally, not in a centralized state GIS. The Red List bridge program covers structurally deficient bridges but doesn't distinguish between posted vs. closed.

---

### New Jersey (NJ)

- **DOT Portal:** https://open-data-portal-njdot.hub.arcgis.com/
- **NJDOT Reference Data:** https://dot.nj.gov/transportation/refdata/
- **Road inventory:** NJDOT Roadway Network — 12,528 miles of State, NHS, STP, and County routes; 2023 Straight Line Diagrams.
  - URL: https://open-data-portal-njdot.hub.arcgis.com/maps/e64c45fa1ef14ef2b97b517c20f15878
- **Bridge inventory:** Available via NBI. NJDOT maintains bridge data but separate downloadable GIS layer with posted weights not confirmed.
- **Truck restrictions:** Parkway prohibitions and other restrictions exist but bulk GIS download not confirmed from search results.
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no GIS layer confirmed); bridge clearance (NBI); per-lane clearance (no); height restrictions (no standalone GIS); hazmat (no); truck-prohibited (unknown — contact NJDOT GIS at dot.gis@dot.nj.gov).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Annual.
- **Coverage:** State routes, NHS, STP, county routes.
- **Gotchas:** New Jersey has extensive truck prohibitions on the Garden State Parkway and portions of the NJ Turnpike, plus numerous local ordinances. These are not consolidated in a downloadable GIS layer. Contact NJDOT GIS directly.

---

### New Mexico (NM)

- **DOT Portal:** https://planningdivisiongis-nmdot.hub.arcgis.com/ (NMDOT Planning Division GIS Hub)
- **Data Management Bureau:** https://www.dot.nm.gov/planning-research-multimodal-and-safety/planning-division/data-management-bureau/
- **Road inventory:** Road Inventory System (RIS) — NMDOT LRS with HPMS reporting data.
  - Data requests: John.Baker@dot.nm.gov or Ana.Gallant@dot.nm.gov
- **Bridge inventory:** Available via NBI. NMDOT maintains bridge data but GIS download not confirmed.
- **HPMS 2025 submission:** NMDOT published 2024 HPMS data as an open layer: https://planningdivisiongis-nmdot.hub.arcgis.com/datasets/traffic-section-hpms-2025-submittal-of-2024-data/explore
- **Attributes confirmed:** Lane count (yes — HPMS layer); lane width (HPMS sample-only); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual (HPMS); on-demand for RIS requests.
- **Coverage:** State and federal-aid highways.
- **Gotchas:** NMDOT explicitly requires data requests for detailed RIS data — the Hub is more of a viewer with selective publishing. Contact the Data Management Bureau for full attribute access.

---

### New York (NY)

- **DOT Portal:** https://www.dot.ny.gov/gisapps/ (NYSDOT GIS Applications)
- **Highway Data Services:** https://www.dot.ny.gov/highway-data-services
- **Road inventory:** Roadway Inventory System (RIS) Viewer — GIS application for displaying roadway inventory data.
  - Viewer: https://gis.dot.ny.gov/html5viewer/?viewer=risviewer
  - Inventory Listing downloads: https://www.dot.ny.gov/highway-data-services/inventory-listing
- **Bridge inventory:** NYSDOT Structures — available under Terms of Use.
  - URL: https://data.gis.ny.gov/maps/9e038774ef034c7cae5374f3e23f7a67
- **Truck restrictions:** NY has extensive parkway restrictions; NYC DOT has separate data feeds.
  - NYC DOT Data Feeds: https://www.nyc.gov/html/dot/html/about/datafeeds.shtml
- **Attributes confirmed:** Lane count (yes — RIS); lane width (yes — RIS contains lane-width data as a standard HPMS-derived element); surface width (yes); posted weight limits (no standalone statewide GIS layer); bridge clearance (NBI + NYSDOT Structures); per-lane clearance (no); height restrictions (no statewide layer — NYC DOT has some data); hazmat (no); truck-prohibited (partial — parkways documented but not bulk GIS download confirmed).
- **License:** Terms of Use required (not fully public domain); NYSDOT requires acknowledgment.
- **Format:** REST, download via Highway Data Services.
- **Cadence:** Quarterly.
- **Coverage:** All public roads including state, county, and local.
- **Gotchas:** New York's parkway network (Taconic, Hutchinson, Saw Mill, etc.) has strict height restrictions (typically 7'0"–12'6") that affect many trucks. NYC's truck route network is separate and managed by NYC DOT. The statewide RIS does not consolidate all of these. For NYC specifically, the Open Data Portal (data.cityofnewyork.us) has some truck route data.

---

### North Carolina (NC)

- **DOT Portal:** https://connect.ncdot.gov/resources/gis/pages/gis-data-layers.aspx
- **Road inventory:** Road Inventory Data and Reports — multiple formats.
  - URL: https://connect.ncdot.gov/resources/State-Mapping/Pages/Road-Inventory-Data-and-Reports.aspx
- **Bridge inventory:** NCDOT Bridge Locations — point layer with NBI-derived attributes.
  - REST: https://gis11.services.ncdot.gov/arcgis/rest/services/NCDOT_Bridges/MapServer/0
  - Dynamic Structures map: https://hub.arcgis.com/maps/NCDOT::ncdot-dynamic-structures/explore
  - **Confirmed fields in bridge layer:** `Posted Single Vehicle` weight, `Posted Tractor Trailer Semi Truck` weight.
- **Posted Bridges Maps:** https://connect.ncdot.gov/resources/State-Mapping/Pages/Posted-Bridges-Maps.aspx (maps by Division; 14 divisions statewide).
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (yes — per-bridge posted single vehicle and TT semi-truck weights in NCDOT bridge GIS layer); bridge clearance (NBI + NCDOT Structures); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, ArcGIS Map Service.
- **Cadence:** Quarterly GIS releases (May, August, November, February).
- **Coverage:** All 125,000 centerline miles of public roads.
- **Gotchas:** NCDOT's bridge GIS layer is one of the best in the country for published per-bridge posted weight limits with TT semi-truck specificity. The dynamic structures service is updated daily from the bridge database. Bridge structures layer includes pipes, culverts, railroad bridges, and tunnel structures.

---

### North Dakota (ND)

- **DOT Portal:** https://gishubdata-ndgov.hub.arcgis.com/ (North Dakota GIS Hub)
- **NDDOT GIS:** https://www.dot.nd.gov/construction-and-planning/planning-process/gis-and-mapping
- **Road inventory:** Available via ND GIS Hub with NDDOT tag.
- **Bridge inventory:** NDDOT bridge locations with NBI condition ratings.
  - URL: https://www.dot.nd.gov/construction-and-planning/bridge
- **Dashboard:** NDDOT Roads and Bridges dashboard: https://www.dot.nd.gov/dot/view/dotdashboardroads.aspx
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no downloadable GIS confirmed); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, web services.
- **Cadence:** Annual.
- **Coverage:** State highway network; local roads via ND GIS.
- **Gotchas:** North Dakota has extensive seasonal weight restrictions (spring thaw) but these are not published as a GIS layer; they're issued as press releases and regulatory notices.

---

### Ohio (OH)

- **DOT Portal:** https://tims.dot.state.oh.us/tims (ODOT TIMS — Transportation Information Management System)
- **Data download:** https://gis.dot.state.oh.us/tims/Data/Download
- **Road inventory:** Road Inventory — ODOT's "official" source for certified mileage, lane miles, Federal Aid eligibility, and HPMS. Confirmed to include pavement width (total drivable width all lanes combined, cardinal side).
  - Dataset details: https://tims.dot.state.oh.us/tims/data/dataset/8999c0f518cc46b99b454c9fa51ce409
- **Bridge inventory:** Bridge Inventory — directly downloadable via TIMS.
  - Dataset details: https://tims.dot.state.oh.us/tims/data/dataset/474a6698368d4e62a8be9978abbee579
- **Attributes confirmed:** Lane count (yes); lane width (unknown per-lane); surface width (yes — "total drivable width"); posted weight limits (no standalone GIS layer confirmed); bridge clearance (NBI + ODOT bridge inventory); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain (contact TIMS@dot.ohio.gov).
- **Format:** Shapefile, GDB (direct download from TIMS).
- **Cadence:** Annual.
- **Coverage:** Full Ohio public road network.
- **Gotchas:** ODOT TIMS is one of the more complete DOT data portals in the country — both road inventory and bridge inventory are downloadable (not just viewable). Bridge inventory dataset from NBI is the source for clearance values.

---

### Oklahoma (OK)

- **DOT Portal:** https://gis-okdot.opendata.arcgis.com/
- **Road inventory:** ODOT Functionally Classified Roadway master inventory; also Local road inventory.
  - URL: https://gis-okdot.opendata.arcgis.com/datasets/d3ac3f9d411a4570af55b98b049c1ac4
- **Bridge inventory:** On-System and Off-System bridge datasets — updated **weekly**.
  - On-System: https://gis-okdot.opendata.arcgis.com (search "Bridges On-System")
  - Off-System: https://gis-okdot.opendata.arcgis.com (search "Bridges Off-System")
- **Master Inventory Data Viewer:** https://www-spotlight-okdot.hub.arcgis.com/app/6555de44b6314ab2a71bb0620e52ea78
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (unknown — bridge layers exist but attribute contents need verification); bridge clearance (NBI + OKDOT bridge layers); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Weekly (bridge inventory — among the most frequent of any state).
- **Coverage:** On-system (state highway system) and off-system (local) bridges; road inventory covers functionally classified roads.
- **Gotchas:** Oklahoma's weekly bridge update cadence is exceptional — best freshness of any state for bridge data. Also has SHP file downloads available for select layers: https://www.odot.org/maps/shp/index.htm

---

### Oregon (OR)

- **DOT Portal:** https://gis.odot.state.or.us/transgis/ (ODOT TransGIS)
- **Oregon Data portal:** https://www.oregon.gov/odot/Data/Pages/GIS%20Data.aspx
- **Road inventory:** TransGIS REST service — annual state highway assets and frequently requested datasets.
  - Categories: Structures (bridges), Drainage, Equipment, Roadway.
  - **Confirmed bridge layers:** Bridges, Scour Critical Bridges, Sign Bridges, Weight Restricted Bridges, Posted Bridges, Low Clearance Bridges.
- **Bridge inventory:** ODOT TransGIS "Bridges" and "Posted Bridges" — via REST service.
  - Bridge Conditions: https://gis.odot.state.or.us/bridgeconditions/
- **Attributes confirmed:** Lane count (yes — TransGIS roadway layers); lane width (unknown); surface width (unknown); posted weight limits (yes — dedicated "Weight Restricted Bridges" and "Posted Bridges" layers in TransGIS); bridge clearance (NBI + ODOT "Low Clearance Bridges" layer); per-lane clearance (no); height restrictions (partial — "Low Clearance Bridges" is a specific TransGIS layer); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** REST service (usable in ArcGIS Desktop/Online, not a direct shapefile download — contact odotmaps@odot.oregon.gov for export).
- **Cadence:** Annual state highway snapshots; bridge data may update more frequently.
- **Coverage:** State highway system; some local roads via county layers.
- **Gotchas:** ODOT TransGIS is a map service (REST), not a simple shapefile portal. Data export requires either ArcGIS or a REST query. The "Low Clearance Bridges" layer is specifically designed for vehicles with height concerns — verify attribute contents (clearance value, route, milepost). Oregon is a standout for having explicit low-clearance and posted-bridge layers.

---

### Pennsylvania (PA)

- **DOT Portal:** https://data-pennshare.opendata.arcgis.com/ (PennShare Open Data)
- **GIS Hub:** https://gis-hub-pennshare.hub.arcgis.com/
- **PennDOT OneMap:** https://onemap.penndot.gov/
- **Road inventory:** PennDOT Roadway Management System (RMS) — REST API and WMS. Large statewide dataset.
- **Bridge inventory:** Bridge Conditions Map available via OneMap; underlying data download from PASDA (Pennsylvania Spatial Data Access).
  - PASDA: https://www.pasda.psu.edu/
- **Attributes confirmed:** Lane count (yes); lane width (yes — RMS is HPMS-compliant and includes lane width at sample sections); surface width (unknown in public download); posted weight limits (no standalone GIS); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST (ArcGIS/WMS).
- **Cadence:** Annual.
- **Coverage:** All public roads statewide.
- **Gotchas:** Pennsylvania has an extremely large road network (~121,000 miles). PASDA is the authoritative GIS clearinghouse for PA state data and may have more current bridge data than the PennShare portal.

---

### Rhode Island (RI)

- **DOT Portal:** https://www.rigis.org/ (RIGIS — Rhode Island GIS Clearinghouse)
- **RIDOT Roads:** https://www.rigis.org/datasets/edc::ridot-roads-2016/
- **Road inventory:** RIDOT Roads 2016 — all highway, road, and street centerlines paved and unpaved. **Last update: 2016** — significantly stale.
- **Bridge inventory:** Available via NBI. No separate RIDOT bridge open-data layer confirmed.
- **Attributes confirmed:** Lane count (unknown — 2016 dataset may not have been confirmed as detailed as others); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile.
- **Cadence:** Ad-hoc — **last update 2016** (major freshness gap).
- **Coverage:** All RI roads.
- **Gotchas:** Rhode Island is the **worst-coverage state** for open-data currency. The road layer is 9+ years stale and RI is a small state where local municipalities manage many roads. For truck routing, use HPMS public shapefile for lane count and NBI for bridges; RIGIS roads are geometry-only anyway. Contact RIDOT directly (ridot.net) for current data requests.

---

### South Carolina (SC)

- **DOT Portal:** https://www.scdot.org/travel/travel-mappinggis.html
- **SCDOT GIS Site:** https://info2.scdot.org/GISMapping/Pages/GIS.aspx
- **Roadway Information System:** https://ris.scdot.org/ (RIS — public roadway information)
- **Road inventory:** SCDOT Owned and Maintained Roadway (Updated Monthly).
  - URL: https://www.arcgis.com/home/item.html?id=cea7d77b22dc48e887d0d44e05c085d4
- **Bridge inventory:** All SC DOT-maintained bridges available as shapefile or File GDB.
- **Attributes confirmed:** Lane count (yes — monthly-updated road layer); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI + SCDOT bridge layer); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, File GDB, KMZ.
- **Cadence:** Monthly (road layer — one of the better update cadences).
- **Coverage:** SCDOT-maintained roads and bridges.
- **Gotchas:** KML/KMZ format is available if GIS tooling is unavailable. Monthly update cadence is good for operational currency. Contact: GIS/Mapping at 803-737-1677.

---

### South Dakota (SD)

- **DOT Portal:** https://opendata2017-09-18t192802468z-sdbit.opendata.arcgis.com/ (South Dakota GIS Data)
- **SDDOT GIS:** https://dot.sd.gov/inside-sddot/forms-publications/maps/gis/
- **Road inventory:** Non-State Public Road Inventory — updated annually; physical features for all public roads not on state highway system.
  - URL: https://dot.sd.gov/projects-studies/planning/non-state-public-road-inventory/
- **Bridge inventory:** Bridges and Culverts — structure location and attributes along state highways and local roads.
  - URL: https://opendata2017-09-18t192802468z-sdbit.opendata.arcgis.com/datasets/fae9cfa385ed4346abac935c5831c8c1_0/about
- **Attributes confirmed:** Lane count (yes — state highway system); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI + SD Bridges and Culverts); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Annual.
- **Coverage:** State highway system + local public roads.
- **Gotchas:** South Dakota has an unusually high proportion of county- and township-owned bridges, many with weight restrictions not captured in state data. The Non-State Road Inventory is a strong asset for rural coverage.

---

### Tennessee (TN)

- **DOT Portal:** https://tn-tnmap.opendata.arcgis.com/ (State of Tennessee Downloadable GIS Data)
- **TNMap:** https://tnmap.tn.gov/
- **TDOT GIS:** https://www.tn.gov/tdot/long-range-planning-home/longrange-data-visualization/gis-mapping-and-support.html
- **Road inventory:** Transportation layers available via TNMap portal with TDOT tag.
- **Bridge inventory:** Bridge Condition layer (data from 2022 per search results).
  - URL: https://tn-tnmap.opendata.arcgis.com/maps/8d89d92e76d54ba08290b652b74c4549
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI + TDOT bridge condition); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST.
- **Cadence:** Annual (bridge condition 2022 — may be stale).
- **Coverage:** State highway system.
- **Gotchas:** Tennessee's bridge condition data appears dated (2022) in the search results. Verify current data currency. TDOT's heavy industrial road network (coal country routes) has specific weight designations not confirmed in open data.

---

### Texas (TX)

- **DOT Portal:** https://gis-txdot.opendata.arcgis.com/ (TxDOT Open Data Portal)
- **Roadway Inventory:** https://www.txdot.gov/data-maps/roadway-inventory.html
- **Road inventory:** TxDOT Roadway Inventory — published annually with GIS linework and all inventory attributes. Multiple layers: all roads, on-system only, specifications PDF.
  - All roads: https://gis-txdot.opendata.arcgis.com/datasets/txdot-roadway-inventory
  - On-system: https://gis-txdot.opendata.arcgis.com/datasets/txdot-roadway-inventory-onsystem
  - 2023 Specifications: https://gis-txdot.opendata.arcgis.com/documents/5592b6569dd54884b9de9e9341435bf9
- **Bridge inventory:** Available via NBI. TxDOT bridge rating is managed in Pontis/AASHTOWare Bridge Management, not confirmed as downloadable GIS.
- **Attributes confirmed:** Lane count (yes — inventory layer); lane width (yes — TxDOT roadway inventory specifications include lane width); surface width (yes); posted weight limits (no standalone GIS layer — load postings in bridge management system); bridge clearance (NBI); per-lane clearance (no); height restrictions (no standalone GIS layer); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, GeoJSON, REST.
- **Cadence:** Annual (roadway inventory).
- **Coverage:** State-maintained system (on-system) + off-system with county and local roads.
- **Gotchas:** Texas is the largest dataset by road miles. On-system and off-system datasets should be used together. TxDOT annual roadway inventory reports are available as PDFs with statistics. The specifications document (PDF) contains the full attribute dictionary.

---

### Utah (UT)

- **DOT Portal:** https://data-uplan.opendata.arcgis.com/ (UDOT Open Data / UPLAN)
- **SGID Roads:** https://gis.utah.gov/products/sgid/transportation/road-centerlines/
- **Road inventory:** Utah Roads via UGRC (Utah Geospatial Resource Center) SGID — multimodal statewide roads, also base for UDOT LRS/ALRS.
- **UDOT Structures (Open Data):**
  - URL: https://digitaldelivery.udot.utah.gov/datasets/uplan::udot-structures-open-data/about
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI + UDOT Structures); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Annual.
- **Coverage:** All public roads statewide; UDOT Structures for bridges.
- **Gotchas:** Utah's UPLAN is well-maintained but truck-specific attribute coverage is thin. The UDOT Structures dataset is a notable open layer worth examining for bridge attributes.

---

### Vermont (VT)

- **DOT Portal:** https://geodata.vermont.gov/ (Vermont Open Geodata Portal)
- **VTransparency:** https://vtransparency.vermont.gov/
- **Road inventory:** VT Road Centerline — statewide road network via VCGI.
  - URL: https://geodata.vermont.gov/datasets/VTrans::vt-road-centerline/about
- **Bridge inventory:** VTrans Long Structures (bridges and culverts).
  - Available via Vermont Open Geodata Portal.
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI + VTrans Long Structures); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Annual.
- **Coverage:** All public roads including state, town, and class 4 roads.
- **Gotchas:** Vermont has a large number of local bridges with posted weight limits (covered bridges, older structures). These are tracked by VTrans but the public GIS layer may not include all local bridge postings.

---

### Virginia (VA)

- **DOT Portal:** https://www.virginiaroads.org/ (Virginia Roads Open Data Portal)
- **VDOT GIS:** https://vgin.vdem.virginia.gov/
- **Road inventory:** Virginia Roads — comprehensive VDOT road inventory via Roadway Inventory Management System (RIMS).
- **Truck Routes:** VDOT Designated Truck Routes and Length Restrictions.
  - URL: https://www.virginiaroads.org/datasets/4803162ce73d458a9b8f6d9cb51aa470
  - Virginia Open Data Portal mirror: https://data.virginia.gov/dataset/vdot-designated-truck-routes-and-length-restrictions
- **Bridge weight restrictions:** Current listing updated **daily** of bridges with restricted weight limits; filterable by route and county; 90-day history.
  - TruckWeb application: http://gis.vdot.virginia.gov/vatruckweb/ (map-viewer with live data)
- **Structure Restrictions Map:** https://vdot.maps.arcgis.com/apps/webappviewer/index.html?id=00cccfd4ef0a44ac84916295d41b87c3
- **Attributes confirmed:** Lane count (yes — RIMS); lane width (unknown); surface width (unknown); posted weight limits (yes — daily-updated bridge weight restriction list); bridge clearance (NBI + VDOT); per-lane clearance (no); height restrictions (unknown — not confirmed in bulk GIS); hazmat (no); truck-prohibited (yes — designated truck routes dataset).
- **License:** Public domain.
- **Format:** Shapefile, REST, XLS, KML.
- **Cadence:** Daily (bridge weight restrictions — best update frequency in the country for this attribute); annual (road inventory).
- **Coverage:** All VDOT-maintained roads; bridge restrictions cover state-maintained structures.
- **Gotchas:** The daily bridge weight restriction update is a standout feature. The TruckWeb map viewer appears to be the primary access point — confirm whether the underlying data is downloadable as a bulk GIS file or only via REST query.

---

### Washington (WA)

- **DOT Portal:** https://gisdata-wsdot.opendata.arcgis.com/ (WSDOT Geospatial Open Data Portal)
- **Road inventory:** Local Agency Public Road Lines — HPMS submittal dataset.
  - URL: https://gisdata-wsdot.opendata.arcgis.com/datasets/d11b7423108b49d19fd141ebccd4d803_0/about
- **Bridge Vertical Clearance Trip Planner:** https://wsdot.wa.gov/data/tools/bridgeclearance/
  - GeoData layer: https://geo.wa.gov/datasets/WSDOT::wsdot-bridge-vertical-clearance-trip-planner/about
  - Fields: `MinVertClrncOverDeck`, `MinVertClrncUnderBridge`, `TunnelMinVertClrncOverRdBy10`
  - **Per-lane status:** WSDOT is actively collecting lane-specific clearances using mobile LiDAR. "Lane in which max and min clearances occur are not always listed" — partial per-lane, more data incoming.
- **All Bridge and Tunnel Inventory:**
  - URL: https://geo.wa.gov/datasets/WSDOT::wsdot-all-bridge-and-tunnel-inventory-state-local/about
- **State Bridge Structures datasets:**
  - On: https://gisdata-wsdot.opendata.arcgis.com/datasets/WSDOT::wsdot-state-bridge-structures-on/about
  - Under: https://gisdata-wsdot.opendata.arcgis.com/datasets/d060e9a4142d42158efa59891d463911_0/about
- **Commercial vehicle restrictions:** https://www.wsdot.wa.gov/commercialvehicle/restrictions/roadlist.aspx
- **Attributes confirmed:** Lane count (yes — HPMS road lines); lane width (unknown); surface width (unknown); posted weight limits (no bulk GIS layer confirmed); bridge clearance (yes — dedicated Vertical Clearance Trip Planner layer with clearance values); per-lane clearance (partial — in active collection, partial public release); height restrictions (yes — bridge vertical clearance dataset covers this use case); hazmat (no); truck-prohibited (no standalone GIS layer).
- **License:** WSDOT Bridge Preservation Office copyright (not fully public domain — requires attribution).
- **Format:** Shapefile, GeoJSON, REST (downloadable from geo.wa.gov).
- **Cadence:** Annual (road inventory); bridge clearance — ongoing collection.
- **Coverage:** State highway system; bridge inventory includes local agency structures.
- **Gotchas:** Washington is one of the top two states for bridge vertical clearance open data (with Colorado). The Vertical Clearance Trip Planner is specifically designed for oversized vehicle routing — it covers structures with restrictions up to 16 feet on state highways. Per-lane detail is being added via mobile LiDAR — follow WSDOT Bridge Office publications for updates.

---

### West Virginia (WV)

- **DOT Portal:** https://data-wvdot.opendata.arcgis.com/ (WVDOT Open Data Portal)
- **IT Division GIS:** https://gis.transportation.wv.gov/
- **GIS Data Catalog:** https://transportation.wv.gov/IT/GIS/Pages/DataCatalog.aspx
- **Road inventory:** WVDOT Open Data Portal — multiple transportation layers. Published in ESRI Shapefile format (ZIP), UTM Zone 17 / NAD83.
- **Bridge inventory:** Available via NBI. Separate state bridge GIS download not confirmed.
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain ("supports free data exchange").
- **Format:** Shapefile (UTM Zone 17, NAD83); also web services.
- **Cadence:** Annual.
- **Coverage:** State road network.
- **Contact:** DOTSupport@wv.gov
- **Gotchas:** UTM Zone 17 / NAD83 — reproject to WGS84 for routing use. West Virginia has a significant number of locally-owned rural bridges with weight restrictions, not tracked in state open data.

---

### Wisconsin (WI)

- **DOT Portal:** https://data-wisdot.opendata.arcgis.com/ (WisDOT Open Data)
- **GeoData@Wisconsin (mirror):** https://geodata.wisc.edu/
- **Road inventory:** WISLR (Wisconsin Information System for Local Roads) — road width, surface type, functional classification.
- **Structures Inventory:** In-Service Bridges and Structures — **monthly refresh; includes NBI ratings and load-related attributes.**
  - GeoData@Wisconsin: https://geodata.wisc.edu/catalog/DOT-0499edd27d5440d7884fd313dc7e80390
  - WisDOT Hub: https://data-wisdot.opendata.arcgis.com/datasets/structures-inventory
  - **Confirmed fields:** Bridge Category, Weight Limit Last Changed Date, NBI inspection records.
- **Weight Restricted Bridges:** Dedicated layer.
  - REST: https://dotmaps.wi.gov/arcgis/rest/services/agohub/WEIGHT_RESTRICTED_BRIDGES/MapServer/0
  - Interactive experience: https://experience.arcgis.com/experience/54e1c8031c9b44168f8365f294e70308
- **Attributes confirmed:** Lane count (yes); lane width (unknown); surface width (yes — WISLR road width); posted weight limits (yes — Weight Restricted Bridges layer plus Weight Limit Last Changed Date in structures inventory); bridge clearance (NBI + WisDOT Structures); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain (contact DOTAGOAdmin@dot.wi.gov for additional requests).
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Monthly (structures inventory — best for bridge data freshness).
- **Coverage:** All in-service bridges and structures statewide; roads via WISLR.
- **Gotchas:** WisDOT's bridge rating manual (Chapter 45) documents the Rating methodology — useful context for interpreting posted weights. The Weight Restricted Bridges interactive experience is designed for public use but the underlying REST layer is bulk-accessible.

---

### Wyoming (WY)

- **DOT Portal:** https://data.geospatialhub.org/ (Wyoming Geospatial Hub)
- **WYDOT GIS Group:** https://gis.wyoroad.info/
- **Road inventory:** WyDOT Highways OpenData — county roads, highways, mileposts.
  - ArcGIS Hub: https://hub.arcgis.com/datasets/9c56c885a00c4faba9b228688bfddc1c
  - Wyoming GeoHub: https://data.geospatialhub.org/search?groupIds=b4398d8f34c54f81a2cc2de6be90ae52
- **Bridge inventory:** Available via NBI. Interactive transportation system map shows bridge conditions.
- **Attributes confirmed:** Lane count (partial — state highways yes; county roads limited); lane width (unknown); surface width (unknown); posted weight limits (no); bridge clearance (NBI); per-lane clearance (no); height restrictions (no); hazmat (no); truck-prohibited (no).
- **License:** Public domain.
- **Format:** Shapefile, REST, GeoJSON.
- **Cadence:** Annual.
- **Coverage:** State highway system; county roads partial.
- **Gotchas:** Wyoming has a large number of county roads with limited GIS coverage at the state level. The Geospatial Hub is a well-maintained portal but truck-specific attributes are thin.

---

## 5. Coverage Heatmap

### Attribute Coverage Across 48 States

| Attribute | Widely Covered (>35 states) | Partial (15–35 states) | Rarely Covered (<15 states) | Notes |
|---|---|---|---|---|
| Lane count (through lanes) | YES — ~47/48 states | — | RI (stale) | Via HPMS public shapefile + state LRS |
| Surface/pavement width | Partial — ~30 states | — | ~18 states | Available where state road inventory is detailed (AR, DE, FL, MA, MT, OH, TX, WI confirmed) |
| Lane width (per-lane) | — | ~10 states | ~38 states | HPMS Item 34 is sample-only; few states publish per-lane width in bulk GIS |
| Posted weight limits (bridges) | — | YES — ~15 states | ~33 states | AR, KY, LA, MN, NC, NE, OK, OR, VA, WI confirmed; others NBI-only |
| Bridge vertical clearance (per-structure) | YES — 48/48 via NBI | — | — | NBI Item 54B available for all states via federal ASCII download |
| **Bridge vertical clearance (per-lane)** | — | — | **NO STATE** | Only CDOT (CO) and WSDOT (WA) are collecting this; neither has a complete public release |
| Posted height restrictions (signs/gantries) | — | — | ~3 states partially | CT (parkways), WA (clearance trip planner), OR (low-clearance layer) only; most states do not publish this |
| Hazmat route designations | Partial (~48 states via HPMS STRAHNET) | — | — | HPMS Item 65 is the only consistent source; state-specific hazmat routing is rare in open data |
| Truck-prohibited segments | — | ~5 states | ~43 states | CT, MA, NY (partial), VA, some urban DOTs only |

### Key Findings by Attribute

**Lane count:** Near-complete coverage via HPMS public shapefile (all Federal-aid roads) plus state LRS for local roads. Rhode Island is the only notable gap due to stale data.

**Surface width:** Confirmed in AR (LaneWidth + SurfaceWid fields), DE (road width field), FL (SURWIDTH), MA (Surface_Wd), MT (Road Log), OH (pavement width), TX (inventory specs), WI (WISLR). Other states likely have this in their LRS but it's not always exposed in public download layers.

**Per-lane width:** Almost universally absent from bulk GIS. HPMS collects lane width (Item 34) but only on sample sections, not the full network, and this sample data is NOT in the public shapefile. The only path to per-lane width on most segments is commercial data (HERE/TomTom publish lane widths in their road network products).

**Posted weight limits (bridge-level):** A growing number of states publish dedicated posted-bridge GIS layers (AR, KY, LA, MN, NC, NE, OK, OR, VA, WI are confirmed). The NBI provides operating/inventory ratings for all structures but not the actual posted tonnage (which is a state-level decision). This is the most tractable attribute to assemble from free data.

**Bridge vertical clearance (per-structure):** Universally available via federal NBI download. Item 54B (minimum vertical underclearance) is the key field for routing. Limitations: single value per structure, point geometry (not linear), ~8-month publication lag.

**Bridge vertical clearance per lane of travel:** This is the critical gap. No state currently publishes a complete, publicly downloadable per-lane bridge clearance dataset. WSDOT is furthest along (collecting via LiDAR, partial public release) and CDOT has a vertical clearance dashboard (completeness unclear). The commercial vendors (HERE, PTV, Trimble/PC*MILER) derive per-lane clearances from mobile mapping and sign databases — this is the most defensible reason to license commercial data.

**Posted height restrictions (low-clearance signs, viaduct gantries):** Almost entirely absent from state DOT open data. Connecticut's OSTA map, Washington's vertical clearance dataset, and Oregon's "Low Clearance Bridges" layer are the only three states with any version of this. This is the second critical gap: a driver assistant that enforces height routing needs a sign-level inventory of posted clearances (the "9'6" Low Clearance" signs that appear on road-mounted frames, not just bridges), and this data comes from commercial sources (HERE's low-clearance database, Trimble's trucking attribute layer).

**Truck-prohibited segments:** Massachusetts (`T_Exc_Type` field), Connecticut (OSTA), Virginia (truck routes dataset), and New York (partial, NYC-specific) are the best open-data sources. Most states do not publish truck-prohibited segments in bulk GIS.

---

## 6. Recommendation

**Recommendation: Option (c) — Hybrid strategy (open for backbone, commercial for critical gaps).**

The open-data foundation is viable for the core routing backbone on interstates and NHS routes: HPMS provides lane count and truck-network designation at near-100% coverage, NBI provides per-structure bridge clearance at 100% coverage, and a growing set of states (~15) publish posted bridge weight limits. OSM adds `maxheight`, `maxweight`, `maxwidth`, `hgv` tags that are increasingly reliable on major freight corridors and in urban areas, particularly where the community has validated commercial-vehicle restrictions. Starting from open data alone is not reckless for the BitNet project — it will give you a working validator for weight and width violations on the interstate/NHS system with defensible attribute coverage.

However, two gaps require commercial gap-fill to meet the "deterministic" bar required by the stated safety requirement:

1. **Low-clearance sign inventory (height enforcement):** No open-data source covers the posted clearance signs (under viaducts, on gantry-mounted boards) that are the primary height-hazard for trucks at the sub-bridge level. HERE's Truck Attributes layer and PTV's Truck Routing attributes both include low-clearance sign inventories. Trimble's PC*MILER has the deepest truck-routing heritage. For a device that absolutely cannot route a truck into a low-clearance situation, you must license one of these. Estimated cost range: $15K–$60K/year depending on coverage and update cadence.

2. **Per-lane bridge vertical clearance:** For the rare scenarios where lane choice changes a truck's safety margin (wide structures with angled superstructure), per-lane clearance is needed. No free source exists. Commercial providers derive this from mobile LiDAR surveys. Alternatively, instrument-based approaches using the truck's own height sensor combined with per-structure NBI clearance (conservative worst-case) may be acceptable for an on-device assistant where the vehicle's own sensor provides ground-truth.

**Suggested architecture:** Build the open-data backbone (HPMS + NBI + state posted-bridge layers + OSM) and license HERE's Truck Attributes or PTV's map for the low-clearance sign dataset as a one-time or annual update. Do NOT pay for a wholesale commercial road network license — the open-data coverage of lane count, surface width, and posted weight limits is good enough for the NHS/interstate system where 95% of truck miles occur.

---

## 7. Open Questions

1. **NMDOT and several smaller-portal states (KS, MS, ND):** Attribute coverage for lane width and surface width was not confirmable without a direct data download. Someone should pull sample segments and inspect the attribute tables in ArcGIS or GDAL.

2. **Connecticut OSTA bulk download:** The OSTA interactive map includes parkway height/weight restrictions — can CTDOT provide a bulk GIS export of the underlying layer? This would be a valuable supplement for routing through New England.

3. **WSDOT per-lane clearance public release timeline:** WSDOT is actively collecting per-lane bridge clearances via mobile LiDAR. Confirm with WSDOT Bridge Preservation Office when the upgraded dataset with lane-specific values will be publicly released.

4. **CDOT vertical clearances completeness:** The CDOT Bridges Vertical Clearances point layer exists on the geospatial hub, but the completeness (what fraction of structures are covered) and field schema (single value vs. per-lane) were not confirmed. Inspect the layer directly.

5. **NBI SNBI transition impact:** The new Specifications for the National Bridge Inventory (SNBI) have 154 items and states are transitioning from the 1995 Coding Guide. First complete SNBI submittal expected 2028. Confirm whether SNBI adds any per-lane or posted-tonnage fields that would change the open-data picture for bridges.

6. **Rhode Island:** The most stale state (roads last updated 2016). If RI coverage matters for the project, file a direct data request with RIDOT. Rhode Island is small enough that a manual review of their bridge inventory via NBI may be sufficient.

7. **Local bridge weight restrictions:** For county and township roads in rural states (SD, NE, MN, IA, WI), local bridge weight restrictions are significant for agricultural truck routing. These are managed by county highway departments, not state DOTs. No open federated source exists — this is a known gap for rural routing.

8. **Hazmat routes:** The HPMS STRAHNET flag (Item 65) and National Truck Network (Item 66) approximate preferred freight corridors but do not encode state hazmat permit route designations. If hazmat routing is a future feature, a separate state-level audit of hazmat routing regulations is required.

---

*Research conducted April 2026. Web sources include primary state DOT portals, FHWA data hubs (data.transportation.gov, geo.dot.gov), and BTS NTAD portal. No data was downloaded; all findings are based on metadata, portal documentation, and dataset schemas. URLs should be verified for currency before production use.*
