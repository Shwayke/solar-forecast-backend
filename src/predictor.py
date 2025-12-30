import torch
import numpy as np

def generate_predictions(models, data):
    """Generate predictions using loaded models and fetched data"""
    
    # Preprocess data for your models
    input_tensor = preprocess_data(data)
    
    with torch.no_grad():  # Don't calculate gradients
        prediction1 = models['model1'](input_tensor)
        prediction2 = models['model2'](input_tensor)
    
    # Post-process predictions
    results = {
        'prediction1': prediction1.tolist(),
        'prediction2': prediction2.tolist()
    }
    
    return results

def preprocess_data(data):
    """Convert raw data to tensor format for models"""
    # Your preprocessing logic
    pass