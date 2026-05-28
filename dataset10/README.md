# 🚀 Global Space Mission Database (1957–2026)

## Overview

This dataset contains **6,230 space launch records** spanning **70 years** of human spaceflight — from Sputnik's first beep in October 1957 to the Starship mega-launches of 2026. It covers every major orbital and suborbital mission across **13 spacefaring nations**, **46 rocket families**, and **29 launch sites**, with engineered intelligence columns not found in existing public space datasets.

---

## 🌟 What Makes This Dataset Unique

Most public space launch databases (Jonathan's Space Report, Gunter's Orbit Page exports) are flat text files without ML-ready features. This dataset adds:

- **Estimated launch cost (million USD)** per mission — enabling cost-per-kg analysis
- **Booster recovery flag** — quantify the SpaceX reusability revolution
- **Orbital parameters** — apoapsis, periapsis, and inclination per mission
- **Crew size** — human spaceflight missions flagged with exact crew counts
- **Mission type classification** — 19 categories from Military/Reconnaissance to Space Tourism
- **Destination field** — ISS, Moon Surface, Mars, Jupiter, L2 Halo, etc.
- **Launch site geolocation** — lat/lon for all 29 sites enabling map visualisations
- **Historical era encoding** — decade column for Cold War vs NewSpace segmentation

---

## 📁 Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `space_missions.csv` | 6,230 | Individual launch records with all features |
| `annual_summary.csv` | 70 | Year-by-year global aggregates (1957–2026) |
| `rocket_summary.csv` | 46 | Rocket family profiles — launches, success rate, cost |
| `country_summary.csv` | 13 | Country-level all-time statistics |

---

## 📊 Coverage Highlights

- **70 years**: 1957–2026 — the complete space age
- **13 countries**: USA, Soviet Union, Russia, China, Europe, Japan, India, Israel, South Korea, New Zealand, Iran, North Korea, Brazil
- **46 rocket families**: From Sputnik/R-7 and Saturn V to Falcon 9, Starship, and New Glenn
- **19 mission types**: Earth Observation, Human Spaceflight, Military, Interplanetary, Space Tourism, Rideshare, and more
- **29 launch sites**: Baikonur, Cape Canaveral, Jiuquan, Satish Dhawan, Guiana Space Centre, SpaceX Starbase, Rocket Lab LC-1, and more
- **583 crewed missions** launching thousands of humans into space
- **362 booster recoveries** — the commercial reusability revolution captured

---

## 🔬 Column Descriptions

### `space_missions.csv`

| Column | Type | Description |
|--------|------|-------------|
| `mission_id` | string | Unique mission identifier (M000001–M006230) |
| `launch_date` | date | Launch date (YYYY-MM-DD) |
| `year` | int | Launch year (1957–2026) |
| `decade` | string | Decade label (1950s, 1960s, ... 2020s) |
| `country` | string | Launching country/agency nation |
| `rocket_name` | string | Rocket family name |
| `rocket_operator` | string | Organisation operating the rocket |
| `rocket_height_m` | int | Rocket height in metres |
| `rocket_payload_leo_kg` | int | Rocket's LEO payload capacity (kg) |
| `rocket_reusable` | int | Rocket has reusable booster (0/1) |
| `rocket_status` | string | Active / Retired / Development |
| `launch_site` | string | Launch facility name |
| `launch_site_country` | string | Country hosting the launch site |
| `launch_site_lat` | float | Launch site latitude |
| `launch_site_lon` | float | Launch site longitude |
| `mission_type` | string | Mission category (19 types) |
| `orbit_type` | string | Target orbit (LEO/GEO/GTO/SSO/Lunar Transfer/etc.) |
| `destination` | string | Specific destination (Earth Orbit/ISS/Moon/Mars/etc.) |
| `payload_mass_kg` | int | Primary payload mass in kg |
| `num_payloads` | int | Number of individual satellites/payloads |
| `launch_mass_tonnes` | float | Total launch vehicle mass (tonnes) |
| `apoapsis_km` | int | Orbit apoapsis altitude (km), null for deep space |
| `periapsis_km` | int | Orbit periapsis altitude (km) |
| `inclination_deg` | float | Orbital inclination (degrees) |
| `outcome` | string | Success / Partial Failure / Launch Failure / Mission Failure |
| `mission_duration_days` | int | Mission duration in days (null if ongoing/unspecified) |
| `crew_size` | int | Number of crew members (0 for uncrewed) |
| `booster_recovered` | int | First stage successfully recovered (0/1) |
| `estimated_cost_million_usd` | float | Estimated launch cost in million USD |

### `annual_summary.csv`

| Column | Description |
|--------|-------------|
| `year` | Year |
| `total_launches` | Total launch attempts |
| `successful` | Successful launches |
| `launch_failures` | Launch failures |
| `human_missions` | Crewed missions |
| `total_crew_launched` | Total astronauts/cosmonauts launched |
| `booster_recoveries` | Successful booster landings |
| `avg_payload_kg` | Average payload mass |
| `total_cost_million_usd` | Total estimated launch costs |
| `unique_rockets` | Number of distinct rocket families used |
| `rideshare_launches` | Rideshare/multi-manifest launches |
| `success_rate_pct` | Launch success percentage |

### `rocket_summary.csv`

| Column | Description |
|--------|-------------|
| `rocket_name` | Rocket family name |
| `total_launches` | All-time launch count |
| `successes` / `failures` | Outcome counts |
| `first_launch` / `last_launch` | Career year range |
| `avg_payload_kg` | Average payload mass carried |
| `total_crew_launched` | Total humans flown |
| `booster_recoveries` | Total booster recoveries |
| `avg_cost_m` | Average launch cost ($M) |
| `success_rate_pct` | Career success rate (%) |

---

## 💡 Suggested Use Cases

1. **Cost revolution analysis** — Plot cost per kg to LEO from Saturn V ($54K/kg) to Starship target (<$100/kg)
2. **Cold War vs Commercial era** — How did launch rates, mission types, and failure rates change?
3. **Booster recovery impact** — Quantify cost savings and launch cadence changes post-2015
4. **Human spaceflight timeline** — Every crewed mission from Vostok 1 to commercial crew
5. **Country power shift** — Soviet dominance → US dominance → China surge
6. **Launch site geography** — Choropleth maps of global launch activity
7. **Orbital mechanics exploration** — Apoapsis/periapsis/inclination distributions by mission type
8. **Failure mode analysis** — Early career failure rates, which missions fail most, decade trends
9. **Rideshare megaconstellation** — The Starlink/OneWeb era captured 2019–2026
10. **Launch success predictor** — ML classifier using year, rocket maturity, mission type

---

## ⚠️ Disclaimer

This is a **synthetically generated dataset** calibrated to real-world space launch statistics, rocket specifications, country launch rates, and mission type distributions. Launch counts by year, country, and decade are modeled on published historical records (Jonathan's Space Report, NASA Launch Log, ESA statistics). Specific mission names, exact dates, and individual payload details are simulated. Not sourced from the FIA, NASA, ESA, CNSA, or any official space agency database. Suitable for educational, research, and ML practice purposes.

---

## 🔭 Key Historical Milestones Captured

| Year | Event |
|------|-------|
| 1957 | Sputnik 1 — first artificial satellite |
| 1961 | Vostok 1 — first human in space (Gagarin) |
| 1969 | Apollo 11 — first Moon landing |
| 1981 | Space Shuttle first flight |
| 1986 | Challenger disaster — captured in failure data |
| 1998 | ISS construction begins |
| 2011 | Space Shuttle retirement |
| 2015 | First Falcon 9 booster recovery — reusability era begins |
| 2020 | SpaceX Crew Dragon — US commercial human spaceflight |
| 2021 | Blue Origin & Virgin Galactic suborbital tourism flights |
| 2022 | Artemis I — return to the Moon (uncrewed) |
| 2023 | First Starship integrated test flights |
| 2024 | Starship reaches orbit; New Glenn first flight |
| 2026 | Multi-national lunar crewed missions, commercial station era |

---

## 📜 License

CC BY-SA 4.0 — Free to use, share, and adapt with attribution.

---

*Dataset covers 1957–2026 | 6,230 launches | 13 countries | 46 rockets*
