# TNSAIML-Training

## AeroSense: AI-Powered Air Quality Prediction and Environmental Monitoring System

**Air Quality Prediction and Monitoring using Machine Learning**

AeroSense is a Streamlit-based project for air-quality analysis and prediction using machine learning. It is designed to combine pollutant data, AQI prediction, interactive visualizations, and live atmospheric data.

### Features

- Interactive Streamlit dashboard
- AQI prediction using machine learning
- Live air-quality data integration using the Open-Meteo Air Quality API
- Historical data analysis and visualization
- Forecast and trend visualization
- Model evaluation and comparison
- CPCB AQI categories and health guidance

### Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Requests
- Joblib

### Project Structure

- `app.py` — Streamlit application
- `src/` — data processing, prediction, model, and API modules
- `tests/` — automated tests
- `requirements.txt` — Python dependencies

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/pranathi-nai/TNSAIML-Training.git
   cd TNSAIML-Training
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Important Notes

- Live data retrieval requires an internet connection.
- Model performance depends on the quality and representativeness of the training dataset.
- Verify the data sources, evaluation metrics, and model outputs before using them for research or public-health decisions.
- This project is intended for educational and informational purposes, not medical diagnosis.

### Future Enhancements

- Interactive city-wise pollution maps
- Improved historical datasets
- More robust time-series forecasting
- Additional model evaluation and explainability

### License

Add the appropriate license information if a license file is included in the repository.
