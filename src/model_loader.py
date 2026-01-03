import numpy as np
import joblib
import pickle
import torch
import os
from tensorflow import keras
from transformers import AutoformerConfig, AutoformerForPrediction

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRU_DIR = os.path.join(BASE_DIR, 'models', 'gru')
AUTOFORMER_DIR = os.path.join(BASE_DIR, 'models', 'autoformer')

# Autoformer configuration constants
LOOKBACK_HOURS = 336  # 14 days
FORECAST_HOURS = 96   # 4 days
NUM_TIME_FEATURES = 3  # hour, day_of_year, month
MODEL_DIMENSION = 128
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 2
NUM_ATTENTION_HEADS = 8
FEEDFORWARD_DIMENSION = 512
AUTOCORRELATION_FACTOR = 3
MOVING_AVERAGE_WINDOW = 25
DROPOUT_RATE = 0.1
LAGS_SEQUENCE = [1, 2, 3, 24, 48, 168, 336]

def load_models():
    """Load both Keras GRU and PyTorch Autoformer models"""
    models = {}
    scalers = {}
    climatology = {}

    # ===== Load GRU with Legacy Format =====
    print("Loading GRU...")
    try:
        # Import legacy loader
        from keras.saving import legacy_h5_format
        
        # Load using legacy format
        models['gru'] = legacy_h5_format.load_model_from_hdf5(
            os.path.join(GRU_DIR, 'solar_gru_model.keras')
        )
        print("✓ GRU loaded successfully (legacy format)")
    except Exception as e:
        print(f"Legacy loading failed: {e}")
        # Try with compile=False as fallback
        models['gru'] = keras.models.load_model(
            os.path.join(GRU_DIR, 'solar_gru_model.keras'),
            compile=False
        )
        print("✓ GRU loaded successfully (compile=False)")
    
    # Load the scalers that were saved during training
    with open(os.path.join(GRU_DIR, 'weather_scaler.pkl'), 'rb') as f:
        scalers['weather_scaler'] = pickle.load(f)
    
    with open(os.path.join(GRU_DIR, 'power_scaler.pkl'), 'rb') as f:
        scalers['power_scaler'] = pickle.load(f)
    

    # ===== Load Autoformer =====
    print("Loading Autoformer...")
    
    model_config = AutoformerConfig(
        prediction_length=FORECAST_HOURS,
        context_length=LOOKBACK_HOURS,
        num_time_features=NUM_TIME_FEATURES,
        num_static_categorical_features=1,
        cardinality=[1],
        embedding_dimension=[2],
        d_model=MODEL_DIMENSION,
        encoder_layers=NUM_ENCODER_LAYERS,
        decoder_layers=NUM_DECODER_LAYERS,
        encoder_attention_heads=NUM_ATTENTION_HEADS,
        decoder_attention_heads=NUM_ATTENTION_HEADS,
        encoder_ffn_dim=FEEDFORWARD_DIMENSION,
        decoder_ffn_dim=FEEDFORWARD_DIMENSION,
        autocorrelation_factor=AUTOCORRELATION_FACTOR,
        moving_average=MOVING_AVERAGE_WINDOW,
        dropout=DROPOUT_RATE,
        attention_dropout=DROPOUT_RATE,
        distribution_output="student_t",
        lags_sequence=LAGS_SEQUENCE,
        scaling=True,
    )
    
    autoformer = AutoformerForPrediction(model_config)
    autoformer.load_state_dict(torch.load(
        os.path.join(AUTOFORMER_DIR, 'autoformer.pth'),
        map_location='cpu'
    ))
    autoformer.eval()
    
    models['autoformer'] = autoformer
    print("✓ Autoformer loaded successfully")

    # Load Autoformer scalers
    scalers['autoformer_weather_scaler'] = joblib.load(
        os.path.join(AUTOFORMER_DIR, 'autoformer_weather_scaler.pkl')
    )
    scalers['autoformer_power_scaler'] = joblib.load(
        os.path.join(AUTOFORMER_DIR, 'autoformer_power_scaler.pkl')
    )
    print("✓ Autoformer scalers loaded successfully")

    # Load climatology lookup tables
    climatology['clim_table'] = np.load(
        os.path.join(AUTOFORMER_DIR, 'clim_table.npy')
    )
    climatology['clim_valid'] = np.load(
        os.path.join(AUTOFORMER_DIR, 'clim_valid.npy')
    )
    climatology['clim_global_mean'] = np.load(
        os.path.join(AUTOFORMER_DIR, 'clim_global_mean.npy')
    )
    print("✓ Climatology data loaded successfully")
    
    return models, scalers, climatology