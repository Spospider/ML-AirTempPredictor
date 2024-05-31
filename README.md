# Outdoor Air Temperature Prediction Model

## Model Used
XG-Boost Regressor with a gustom gain function (to handle extreme temp. traiding data points)
## Dataset Used 
[Climate Weather Surface of Brazil - Hourly (North)](https://www.kaggle.com/datasets/PROPPG-PPG/hourly-weather-surface-brazil-southeast-region/data)

## Try it out
https://temppredictor.streamlit.app

## LocalSetup
1. Create a venv (1st time only)
```bash
python -m venv venv
```

2. Activate the virtual environment

> Unix:
> ```bash
> source venv/bin/activate
> ```

> Windows:
> ```
> venv\Scripts\activate
> ```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Run application
```bash
streamlit run app.py
```
