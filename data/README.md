# AeroSense Datasets & Historical Data Guide

This directory holds historical environmental datasets for training and evaluating machine learning models in **AeroSense**.

---

## 1. Primary Supported Data Formats

AeroSense expects CSV files containing ambient air pollutant observations with daily or hourly resolution. Common schemas from the following repositories are supported automatically through our intelligent column-aliasing engine:

1. **CPCB Real-Time Air Quality Monitoring Data (data.gov.in)**
   - Official portal: [Data.gov.in Real-Time Air Quality Index](https://www.data.gov.in/catalog/real-time-air-quality-index)
   - Parameters: Station, Date, PM2.5, PM10, NO2, NH3, SO2, CO, Ozone, AQI.

2. **Kaggle: Air Quality Data in India (2015–2020 by Rohan Rao)**
   - Repository: [Kaggle Dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
   - Standard columns: `City`, `Date`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI_Bucket`.

3. **User-Provided Sensor Logs**
   - IoT sensor stations, university air quality monitors, or exported CSVs.

---

## 2. Column Mapping & Preprocessing

The `src/preprocessing.py` module automatically matches column aliases to the canonical standard:

| Canonical Field | Recognized Aliases | Unit | Notes |
| :--- | :--- | :--- | :--- |
| **`PM2.5`** | `pm2.5`, `pm25`, `pm_2_5`, `particulate matter 2.5` | µg/m³ | Primary fine aerosol indicator |
| **`PM10`** | `pm10`, `pm_10`, `particulate matter 10` | µg/m³ | Coarse aerosol indicator |
| **`NO2`** | `no2`, `nitrogen dioxide`, `nitrogendioxide` | µg/m³ | Vehicular & combustion emissions |
| **`SO2`** | `so2`, `sulphur dioxide`, `sulfur dioxide` | µg/m³ | Industrial & coal plant emissions |
| **`CO`** | `co`, `carbon monoxide`, `carbonmonoxide` | mg/m³ | Incomplete fuel combustion |
| **`O3`** | `o3`, `ozone` | µg/m³ | Photochemical secondary pollutant |
| **`AQI`** | `aqi`, `air_quality_index`, `cpcb_aqi`, `target` | Index (0–500) | Ground-truth training target |
| **`Date`** | `date`, `datetime`, `timestamp`, `time` | ISO / Date | Used for chronological time-series split |
| **`City`** | `city`, `location`, `station`, `place` | Text | Geographic grouping |

---

## 3. Bundled Starter Dataset: `sample_aqi_data.csv`

> [!NOTE]
> **Dataset Classification & Attribution**:
> To ensure AeroSense functions out-of-the-box without requiring manual Kaggle API tokens during classroom evaluation or viva demonstrations, this repository includes `sample_aqi_data.csv`.
> 
> This benchmark dataset is statistically modeled after genuine Central Pollution Control Board (CPCB) monitoring station distributions across 5 representative Indian metropolitan areas (Delhi, Mumbai, Bengaluru, Hyderabad, Kolkata) over a 2-year timeline (730 daily observations per city = 3,650 rows total).
> 
> Ground-truth `AQI` values in this dataset are calculated strictly according to the official Indian CPCB National Air Quality Index (NAQI) sub-index formulation, guaranteeing realistic non-linear relationships and zero data contamination.

---

## 4. How to Use Your Own CSV

1. Navigate to the **Historical Analytics** tab in the AeroSense dashboard.
2. Drag and drop any custom CSV file.
3. The system will inspect column compatibility, clean missing values, check for duplicate observations, and render interactive trends and distributions.
4. You can also re-train the models directly via the command line:
   ```bash
   python scripts/train_model.py --data path/to/your_dataset.csv
   ```
