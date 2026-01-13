import numpy as np
from src.predictors.autoformer_predictor import predict_autoformer
from src.predictors.gru_predictor import predict_gru


def generate_predictions(models, data, scalers, climatology):
    """Generate predictions using both GRU and Autoformer models
    
    Args:
        models: dict with 'gru' and 'autoformer' models
        data: DataFrame with columns ['timestamp', 'temperature', 'humidity', 
              'solar_radiation', 'wind_speed', 'wind_direction', 'pressure']
              Must have at least 336 hours for Autoformer
        scalers: dict with scalers for both models
        climatology: dict with 'clim_table', 'clim_valid', 'clim_global_mean'
    
    Returns:
        dict with predictions from both models
    """

    # Calculate maximum possible pv outputs
    solar_radiation = data['radiation'].values
    temperature = data['temperature'].values
    max_pv_outputs = calculate_max_pv_outputs(solar_radiation, temperature)

    # Generate GRU prediction (24 hours)
    gru_result = predict_gru(
        model=models['gru'],
        data=data,
        max_pv_outputs=max_pv_outputs,
        weather_scaler=scalers['gru_weather_scaler'],
        power_scaler=scalers['gru_power_scaler']
    )
    
    # Generate Autoformer prediction (96 hours)
    autoformer_result = predict_autoformer(
        model=models['autoformer'],
        data=data,
        max_pv_outputs=max_pv_outputs,
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

def calculate_max_pv_outputs(solar_radiation, temperature,
                       C_R_PV=1000, G_T_STC=1000, T_C_STC=25, alpha_p=-0.004):
    """
    Calculate photovoltaic power output based on weather parameters.

    Parameters:
    -----------
    solar_radiation : float or array-like
        Solar radiation incident on PV panel surface (W/m²)
    temperature : float or array-like
        Cell/ambient temperature (°C)
    C_R_PV : float, default=1000
        Rated capacity of PV array under standard test conditions (W)
    G_T_STC : float, default=1000
        Solar radiation under standard test conditions (W/m²)
    T_C_STC : float, default=25
        Cell temperature under standard test conditions (°C)
    alpha_p : float, default=-0.004
        Temperature coefficient of power (%/°C), expressed as decimal (-0.4% = -0.004)

    Returns:
    --------
    float or array-like
        Predicted photovoltaic power output (W)

    Formula:
    --------
    P_out_pv = C_R_PV * (G_T / G_T_STC) * [1 + alpha_p * (T_C - T_C_STC)]
    """

    # Calculate radiation ratio
    radiation_ratio = solar_radiation / G_T_STC

    # Calculate temperature effect
    temperature_effect = 1 + alpha_p * (temperature - T_C_STC)

    # Calculate PV output
    P_out_pv = C_R_PV * radiation_ratio * temperature_effect

    # Ensure non-negative output (no power generation when no radiation)
    P_out_pv = np.maximum(P_out_pv, 0)

    return P_out_pv