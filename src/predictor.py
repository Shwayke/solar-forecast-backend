from src.predictors.autoformer_predictor import predict_autoformer
from src.predictors.gru_predictor import predict_gru


def generate_predictions(models, data, scalers, climatology):
    """Generate predictions using both GRU and Autoformer models
    
    Args:
        models: dict with 'gru' and 'autoformer' models
        data: DataFrame with columns ['timestamp', 'temperature', 'humidity', 
              'solar_radiation', 'wind_speed', 'wind_direction']
              Must have at least 336 hours for Autoformer
        scalers: dict with scalers for both models
        climatology: dict with 'clim_table', 'clim_valid', 'clim_global_mean'
    
    Returns:
        dict with predictions from both models
    """
    # Generate GRU prediction (24 hours)
    gru_result = predict_gru(
        model=models['gru'],
        data=data,
        weather_scaler=scalers['gru_weather_scaler'],
        power_scaler=scalers['gru_power_scaler']
    )
    
    # Generate Autoformer prediction (96 hours)
    autoformer_result = predict_autoformer(
        model=models['autoformer'],
        data=data,
        weather_scaler=scalers['autoformer_weather_scaler'],
        power_scaler=scalers['autoformer_power_scaler'],
        climatology=climatology
    )
    
    # Combine results
    results = {
        'gru': gru_result,
        'autoformer': autoformer_result
    }
    
    return results