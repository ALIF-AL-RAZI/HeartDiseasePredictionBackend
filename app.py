from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["https://toggleit-heart-disease-prediction.vercel.app", "http://localhost:3000"])  # Enable CORS for frontend communication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store loaded models
model = None
scaler = None
label_encoders = None
feature_names = None

def load_model_artifacts():
    """Load all saved model artifacts from the 'models' folder with detailed error reporting"""
    os.makedirs('models', exist_ok=True)
    
    global model, scaler, label_encoders, feature_names
    
    # models_dir = os.path.join(os.path.dirname(__file__), 'models')  # Absolute path to models folder

    required_files = {
        'heart_disease_model.pkl': 'model',
        'scaler.pkl': 'scaler',
        'label_encoders.pkl': 'label_encoders',
        'feature_names.pkl': 'feature_names'
    }

    missing_files = [fname for fname in required_files if not os.path.exists(os.path.join('models', fname))]
    if missing_files:
        logger.error(f"Missing required files in models directory: {missing_files}")
        return False

    try:
        logger.info("Loading model artifacts from 'models' folder...")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Models directory: {'models'}")
        logger.info(f"Files in models directory: {os.listdir('models')}")

        with open(os.path.join('models', 'heart_disease_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded: {type(model).__name__}")
        
        with open(os.path.join('models', 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded: {type(scaler).__name__}")
        
        with open(os.path.join('models', 'label_encoders.pkl'), 'rb') as f:
            label_encoders = pickle.load(f)
        logger.info(f"Label encoders loaded: {list(label_encoders.keys()) if label_encoders else 'None'}") 
        
        with open(os.path.join('models', 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        logger.info(f"Feature names loaded: {feature_names}")
        
        # Final validation
        if any(x is None for x in [model, scaler, label_encoders, feature_names]):
            logger.error("One or more artifacts are None after loading.")
            return False
        
        logger.info("All model artifacts loaded and validated successfully!")
        return True

    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def validate_input_data(data):
    """Validate input data format and values"""
    if not data:
        return False, "No input data provided"
        
    required_fields = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    try:
        age = float(data['Age'])
        if not (0 <= age <= 120):
            return False, "Age must be between 0 and 120"
        
        resting_bp = float(data['RestingBP'])
        if not (0 <= resting_bp <= 300):
            return False, "RestingBP must be between 0 and 300"
        
        cholesterol = float(data['Cholesterol'])
        if not (0 <= cholesterol <= 1000):
            return False, "Cholesterol must be between 0 and 1000"
        
        fasting_bs = int(data['FastingBS'])
        if fasting_bs not in [0, 1]:
            return False, "FastingBS must be 0 or 1"
        
        max_hr = float(data['MaxHR'])
        if not (0 <= max_hr <= 250):
            return False, "MaxHR must be between 0 and 250"
        
        oldpeak = float(data['Oldpeak'])
        if not (-10 <= oldpeak <= 10):
            return False, "Oldpeak must be between -10 and 10"
        
        categorical_validations = {
            'Sex': ['M', 'F'],
            'ChestPainType': ['ATA', 'NAP', 'ASY', 'TA'],
            'RestingECG': ['Normal', 'ST', 'LVH'],
            'ExerciseAngina': ['Y', 'N'],
            'ST_Slope': ['Up', 'Flat', 'Down']
        }
        
        for field, valid_values in categorical_validations.items():
            if data[field] not in valid_values:
                return False, f"{field} must be one of: {', '.join(valid_values)}"
        
        return True, "Validation successful"
    
    except (ValueError, TypeError) as e:
        return False, f"Invalid data type: {str(e)}"

def preprocess_input(data):
    """Preprocess input data for prediction with detailed error handling"""
    try:
        # Check if required components are loaded
        if label_encoders is None:
            raise Exception("Label encoders not loaded")
        if scaler is None:
            raise Exception("Scaler not loaded")
        if feature_names is None:
            raise Exception("Feature names not loaded")
            
        logger.info(f"Input data: {data}")
        logger.info(f"Available label encoders: {list(label_encoders.keys()) if label_encoders else 'None'}")
        
        processed_data = data.copy()
        
        # Convert numeric fields
        processed_data['Age'] = float(processed_data['Age'])
        processed_data['RestingBP'] = float(processed_data['RestingBP'])
        processed_data['Cholesterol'] = float(processed_data['Cholesterol'])
        processed_data['FastingBS'] = int(processed_data['FastingBS'])
        processed_data['MaxHR'] = float(processed_data['MaxHR'])
        processed_data['Oldpeak'] = float(processed_data['Oldpeak'])

        # Process categorical fields
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        for col in categorical_cols:
            if col not in label_encoders:
                raise Exception(f"Label encoder for column '{col}' not found. Available encoders: {list(label_encoders.keys())}")
            
            encoder = label_encoders[col]
            if encoder is None:
                raise Exception(f"Label encoder for column '{col}' is None")
                
            try:
                processed_data[col] = encoder.transform([processed_data[col]])[0]
                logger.info(f"Encoded {col}: {data[col]} -> {processed_data[col]}")
            except ValueError as e:
                raise Exception(f"Error encoding {col} with value '{processed_data[col]}': {str(e)}")
        
        # Create feature array
        logger.info(f"Feature names: {feature_names}")
        feature_array = np.array([processed_data[feature] for feature in feature_names]).reshape(1, -1)
        logger.info(f"Feature array shape: {feature_array.shape}")
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        logger.info(f"Scaled features shape: {scaled_features.shape}")
        
        return scaled_features
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise Exception(f"Error in preprocessing: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'success',
        'message': 'Heart Disease Prediction API is running!',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if models are loaded
        if any(x is None for x in [model, scaler, label_encoders, feature_names]):
            return jsonify({
                'status': 'error', 
                'message': 'Model artifacts not properly loaded. Please check server logs.'
            }), 500
        
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
        
        # Validate input
        is_valid, validation_message = validate_input_data(data)
        if not is_valid:
            logger.warning(f"Validation failed: {validation_message}")
            return jsonify({'status': 'error', 'message': validation_message}), 400
        
        # Preprocess and predict
        processed_input = preprocess_input(data)
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]
        
        result = {
            'status': 'success',
            'prediction': int(prediction),
            'prediction_text': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
            'probability': {
                'no_heart_disease': float(prediction_proba[0]),
                'heart_disease': float(prediction_proba[1])
            },
            'confidence': float(max(prediction_proba)),
            'input_data': data,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Prediction made: {result['prediction_text']} (confidence: {result['confidence']:.3f})")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Prediction failed: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    try:
        info = {
            'status': 'success',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'encoders_loaded': label_encoders is not None,
            'feature_names_loaded': feature_names is not None
        }
        
        if model is not None:
            info['model_type'] = str(type(model).__name__)
        if feature_names is not None:
            info['feature_names'] = feature_names
        if label_encoders is not None:
            info['categorical_features'] = list(label_encoders.keys())
            info['categorical_options'] = {
                col: encoder.classes_.tolist() if encoder is not None else []
                for col, encoder in label_encoders.items()
            }
        
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error getting model info: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        model_loaded = model is not None
        scaler_loaded = scaler is not None
        encoders_loaded = label_encoders is not None
        features_loaded = feature_names is not None
        
        # Check file existence
        file_status = {}
        required_files = ['heart_disease_model.pkl', 'scaler.pkl', 'label_encoders.pkl', 'feature_names.pkl']
        for file in required_files:
            file_status[file] = os.path.exists(file)
        
        return jsonify({
            'status': 'success' if all([model_loaded, scaler_loaded, encoders_loaded, features_loaded]) else 'error',
            'components': {
                'model_loaded': model_loaded,
                'scaler_loaded': scaler_loaded,
                'encoders_loaded': encoders_loaded,
                'features_loaded': features_loaded
            },
            'files': file_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Attempting to load model artifacts...")
    if load_model_artifacts():
        logger.info("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model artifacts. Please ensure all .pkl files are present.")
        print("Required files:")
        print("- heart_disease_model.pkl")
        print("- scaler.pkl") 
        print("- label_encoders.pkl")
        print("- feature_names.pkl")
        print("Run the ML training script first to generate these files.")