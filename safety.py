from flask import Blueprint, request, jsonify
from src.models.user import db
from src.models.safety_data import LocationData, ThreatAlert, EmergencyContact, CrimeData
from datetime import datetime
import json

safety_bp = Blueprint('safety', __name__)

@safety_bp.route('/location', methods=['POST'])
def update_location():
    """Update user's current location"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        speed = data.get('speed', 0.0)
        
        if not all([user_id, latitude, longitude]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        location = LocationData(
            user_id=user_id,
            latitude=latitude,
            longitude=longitude,
            speed=speed
        )
        
        db.session.add(location)
        db.session.commit()
        
        return jsonify({'message': 'Location updated successfully', 'location': location.to_dict()}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@safety_bp.route('/alert', methods=['POST'])
def create_alert():
    """Create a new threat alert"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        alert_type = data.get('alert_type', 'manual')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        threat_level = data.get('threat_level', 'medium')
        
        if not all([user_id, latitude, longitude]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        alert = ThreatAlert(
            user_id=user_id,
            alert_type=alert_type,
            latitude=latitude,
            longitude=longitude,
            threat_level=threat_level
        )
        
        db.session.add(alert)
        db.session.commit()
        
        return jsonify({'message': 'Alert created successfully', 'alert': alert.to_dict()}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@safety_bp.route('/emergency-contacts/<int:user_id>', methods=['GET'])
def get_emergency_contacts(user_id):
    """Get emergency contacts for a user"""
    try:
        contacts = EmergencyContact.query.filter_by(user_id=user_id).all()
        return jsonify({'contacts': [contact.to_dict() for contact in contacts]}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@safety_bp.route('/emergency-contacts', methods=['POST'])
def add_emergency_contact():
    """Add a new emergency contact"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        name = data.get('name')
        phone_number = data.get('phone_number')
        relationship = data.get('relationship')
        
        if not all([user_id, name, phone_number, relationship]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        contact = EmergencyContact(
            user_id=user_id,
            name=name,
            phone_number=phone_number,
            relationship=relationship
        )
        
        db.session.add(contact)
        db.session.commit()
        
        return jsonify({'message': 'Emergency contact added successfully', 'contact': contact.to_dict()}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@safety_bp.route('/crime-data', methods=['GET'])
def get_crime_data():
    """Get crime data for threat analysis"""
    try:
        latitude = request.args.get('latitude', type=float)
        longitude = request.args.get('longitude', type=float)
        radius = request.args.get('radius', 5.0, type=float)  # km
        
        if latitude is None or longitude is None:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        # Simple distance calculation (for demo purposes)
        # In production, use proper geospatial queries
        crime_data = CrimeData.query.all()
        nearby_crimes = []
        
        for crime in crime_data:
            # Simple distance calculation (approximate)
            lat_diff = abs(crime.latitude - latitude)
            lon_diff = abs(crime.longitude - longitude)
            if lat_diff <= radius/111 and lon_diff <= radius/111:  # Rough conversion
                nearby_crimes.append(crime.to_dict())
        
        return jsonify({'crime_data': nearby_crimes}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@safety_bp.route('/predict-threat', methods=['POST'])
def predict_threat():
    """Predict threat level based on current conditions"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        current_time = datetime.now().hour
        
        if not all([user_id, latitude, longitude]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Simple threat prediction logic (for demo)
        threat_score = 0
        
        # Time-based risk (higher at night)
        if current_time >= 22 or current_time <= 6:
            threat_score += 30
        elif current_time >= 18 or current_time <= 8:
            threat_score += 15
        
        # Location-based risk (check nearby crime data)
        nearby_crimes = CrimeData.query.filter(
            CrimeData.latitude.between(latitude - 0.01, latitude + 0.01),
            CrimeData.longitude.between(longitude - 0.01, longitude + 0.01)
        ).count()
        
        threat_score += min(nearby_crimes * 10, 50)
        
        # Determine threat level
        if threat_score >= 70:
            threat_level = 'high'
        elif threat_score >= 40:
            threat_level = 'medium'
        else:
            threat_level = 'low'
        
        return jsonify({
            'threat_level': threat_level,
            'threat_score': threat_score,
            'recommendations': get_safety_recommendations(threat_level)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_safety_recommendations(threat_level):
    """Get safety recommendations based on threat level"""
    recommendations = {
        'low': [
            'Stay aware of your surroundings',
            'Keep your phone charged',
            'Share your location with trusted contacts'
        ],
        'medium': [
            'Avoid isolated areas',
            'Stay in well-lit areas',
            'Consider alternative routes',
            'Keep emergency contacts ready'
        ],
        'high': [
            'Seek immediate safe location',
            'Contact emergency services if needed',
            'Alert trusted contacts',
            'Avoid the area if possible'
        ]
    }
    return recommendations.get(threat_level, [])

