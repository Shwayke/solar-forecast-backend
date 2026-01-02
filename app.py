from flask import Flask, jsonify, request
from flask_cors import CORS
from src.data_fetcher import fetch_data
from src.model_loader import load_models
from src.predictor import generate_predictions
import os

app = Flask(__name__)
CORS(app)  # Allow requests from your React frontend

models, scalers, climatology = load_models()

@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        # 1. Fetch fresh data from external API
        data = fetch_data()
        
        # 2. Generate predictions using loaded models
        predictions = generate_predictions(models, data, scalers, climatology)
        
        # 3. Return predictions
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)