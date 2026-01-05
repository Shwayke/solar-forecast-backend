from flask import Flask, jsonify, request
from flask_cors import CORS
from src.data_fetcher import fetch_data
from src.model_loader import load_models
from src.predictor import generate_predictions
import os

app = Flask(__name__)
# Configure CORS for Vercel frontend
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",  # Local development
            "https://*.vercel.app",    # All Vercel preview deployments
            "https://your-actual-domain.vercel.app"  # Your production domain
        ]
    }
})

models, scalers, climatology = load_models()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models is not None
    })

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
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)