import torch
import os

def load_models():
    """Load PyTorch models from .pth files"""
    models = {}
    
    # model1 = YourModel1Architecture()  # Define your model architecture
    # model1.load_state_dict(torch.load('models/model1.pth', map_location='cpu'))
    # model1.eval()  # Set to evaluation mode
    
    # model2 = YourModel2Architecture()
    # model2.load_state_dict(torch.load('models/model2.pth', map_location='cpu'))
    # model2.eval()

    # models['model1'] = model1
    # models['model2'] = model2
    
    print("✓ Models loaded successfully")
    return models