import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, jsonify
from flask_cors import CORS
from src.data_fetcher import fetch_data
from src.model_loader import load_models, get_analytics
from src.predictor import generate_predictions
from src.weather_proxy_generator import get_proxy

models, scalers, climatology = load_models()

app = Flask(__name__)
# CORS for Vercel frontend
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "https://*.vercel.app",
            "https://solar-forecast-frontend.vercel.app"
        ]
    }
})

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    print("Health check requested")
    return jsonify({
        'status': 'healthy',
        'models_loaded': models is not None
    })

@app.route('/api/predict', methods=['GET'])
def predict():

    print("Prediction requested...")
    try:
        # Fetch fresh data from external API
        data = fetch_data()

        # Generate future weather proxy
        weather_proxy = get_proxy(data)

        # Generate predictions using loaded models
        predictions = generate_predictions(models, data, scalers, climatology, weather_proxy)
        
        print("Predictions generated successfully.")

        # Return predictions
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Returns all model evaluation results as JSON"""

    print("Analytics requested...")
    try:
        analytics = get_analytics()
        print("Analytics retrieved successfully.")
        return jsonify(analytics)
    except Exception as e:
        print(f"Error retrieving analytics: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)