from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Float, Integer, String, Date
from sqlalchemy.orm import declarative_base, sessionmaker
import joblib
from flask_cors import CORS
import socket
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Flask app
app = Flask(__name__)
CORS(app)

# Load trained model with error handling
try:
    model = joblib.load("model.pkl")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# SQLAlchemy setup
Base = declarative_base()

class EnergyPrediction(Base):
    __tablename__ = 'energy_prediction'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)
    city = Column(String(50))
    temperature = Column(Float)
    month = Column(Integer)
    day = Column(Integer)
    dayofweek = Column(Integer)
    dayofyear = Column(Integer)
    predicted_consumption = Column(Float)
    actual_consumption = Column(Float, nullable=True)

# Database connection
try:
    engine = create_engine('sqlite:///energy_prediction.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    logging.info("Database connection established")
except Exception as e:
    logging.error(f"Database connection error: {str(e)}")
    raise RuntimeError(f"Database connection failed: {str(e)}")

# List of trained cities
TRAINED_CITIES = [col.split('_')[1] for col in model.feature_names_in_ if col.startswith('City_')]

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received prediction request: {data}")
        
        # Validate input
        if not data:
            logging.warning("No data provided")
            return jsonify({'error': 'No data provided'}), 400
            
        required_fields = ['city', 'temperature', 'date']
        if not all(field in data for field in required_fields):
            logging.warning("Missing required fields")
            return jsonify({'error': 'Missing required fields'}), 400

        city = data['city']
        try:
            temperature = float(data['temperature'])
        except ValueError:
            logging.warning("Invalid temperature format")
            return jsonify({'error': 'Temperature must be a number'}), 400

        date_str = data['date']
        try:
            date_obj = pd.to_datetime(date_str)
        except ValueError:
            logging.warning("Invalid date format")
            return jsonify({'error': 'Invalid date format'}), 400

        # Prepare features
        city_features = {f'City_{c}': 0 for c in TRAINED_CITIES}
        city_key = f'City_{city}'
        if city_key not in city_features:
            logging.warning(f"City '{city}' not in trained list")
            return jsonify({'error': f"City '{city}' not in trained city list", 'valid_cities': TRAINED_CITIES}), 400
        
        city_features[city_key] = 1

        features = {
            'month': date_obj.month,
            'day': date_obj.day,
            'dayofweek': date_obj.dayofweek,
            'dayofyear': date_obj.dayofyear,
            'Predicted Temperature (°C)': temperature,
            **city_features
        }

        # Create DataFrame
        features_df = pd.DataFrame([features])

        # Handle missing features
        for col in model.feature_names_in_:
            if col not in features_df.columns:
                features_df[col] = 0

        # Ensure correct column order
        features_df = features_df[model.feature_names_in_]

        # Make prediction
        try:
            predicted_consumption = model.predict(features_df)[0]
            logging.info(f"Prediction successful: {predicted_consumption} kWh")
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

        # Save to database
        try:
            new_prediction = EnergyPrediction(
                date=date_obj.date(),
                city=city,
                temperature=temperature,
                month=date_obj.month,
                day=date_obj.day,
                dayofweek=date_obj.dayofweek,
                dayofyear=date_obj.dayofyear,
                predicted_consumption=float(predicted_consumption),
                actual_consumption=None
            )
            session.add(new_prediction)
            session.commit()
            logging.info("Prediction saved to database")
        except Exception as e:
            session.rollback()
            logging.error(f"Database error: {str(e)}")
            return jsonify({'error': f"Database error: {str(e)}"}), 500

        # Return response
        return jsonify({
            'date': date_str,
            'city': city,
            'temperature': temperature,
            'predicted_consumption_kWh': round(float(predicted_consumption), 2),
            'status': "Prediction successful and saved to database"
        })

    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': f"Unexpected error: {str(e)}",
            'message': "An error occurred during processing"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    logging.info("Health check called")
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'database_connected': engine is not None
    })

def find_available_port(start_port=5000, max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")

if __name__ == '__main__':
    try:
        port = int(os.environ.get("PORT", find_available_port()))
        ip = socket.gethostbyname(socket.gethostname())
        logging.info(f"Starting server on port {port}")
        logging.info(f"Access app at: http://localhost:{port} or http://{ip}:{port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except RuntimeError as e:
        logging.error(f"{str(e)}")
        print("Try closing other applications using these ports or specify a different port range")