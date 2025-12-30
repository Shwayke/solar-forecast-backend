import torch
from tensorflow import keras
from transformers import AutoformerConfig, AutoformerForPrediction

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
    
    # ===== Load GRU =====
    print("Loading GRU...")
    models['gru'] = keras.models.load_model('models/gru.keras')
    print("✓ GRU loaded successfully")
    
    # ===== Load Autoformer =====
    print("Loading Autoformer...")
    
    # Create the configuration
    model_config = AutoformerConfig(
        prediction_length=FORECAST_HOURS,
        context_length=LOOKBACK_HOURS,
        
        # Features
        num_time_features=NUM_TIME_FEATURES,
        num_static_categorical_features=1,
        cardinality=[1],
        embedding_dimension=[2],
        
        # Architecture
        d_model=MODEL_DIMENSION,
        encoder_layers=NUM_ENCODER_LAYERS,
        decoder_layers=NUM_DECODER_LAYERS,
        encoder_attention_heads=NUM_ATTENTION_HEADS,
        decoder_attention_heads=NUM_ATTENTION_HEADS,
        encoder_ffn_dim=FEEDFORWARD_DIMENSION,
        decoder_ffn_dim=FEEDFORWARD_DIMENSION,
        
        # Autoformer-specific
        autocorrelation_factor=AUTOCORRELATION_FACTOR,
        moving_average=MOVING_AVERAGE_WINDOW,
        
        # Regularization
        dropout=DROPOUT_RATE,
        attention_dropout=DROPOUT_RATE,
        
        # Distribution for probabilistic forecasting
        distribution_output="student_t",
        
        # Important lags
        lags_sequence=LAGS_SEQUENCE,
        
        scaling=True,
    )
    
    # Initialize model with config
    autoformer = AutoformerForPrediction(model_config)
    
    # Load trained weights
    autoformer.load_state_dict(torch.load('models/autoformer.pth', map_location='cpu'))
    autoformer.eval()
    
    models['autoformer'] = autoformer
    print("✓ Autoformer loaded successfully")
    
    return models