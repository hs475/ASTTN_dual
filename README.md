Final Project For ECE228

# ASTTN_pytorch with Weather Integration

A lightweight fork of ASTTN_pytorch that augments the original spatial-temporal transformer for traffic forecasting with weather features (e.g., temperature, precipitation, humidity).  the reference
 https://github.com/yokifly/ASTTN_pytorch
- **Integration**: Align weather time series with traffic data and concatenate or embed as extra node features.  
- **Usage**:  
  1. Prepare traffic and weather CSVs with matching timestamps.  
  2. Preprocess to merge into a single input (e.g., via a small script or utility function).  
  3. Run training with a flag to enable weather, e.g.:  
     ```bash
     python main.py --data_path ./data/processed/ --use_weather True --weather_features temperature,precipitation,humidity
     ```  
- **Configuration**: Add command-line or config options: `--use_weather`, `--weather_path`, `--weather_features`.  
- **Expected Benefit**: Improved forecasting by injecting external context. Test with/without weather to compare metrics.  

Refer to the original ASTTN repo for core details; this fork only adds weather loading, preprocessing, and model input fusion.  
