# Aadhaar-Linked Multi-Factor Authentication System with Behavioral Biometrics
# Complete implementation with Flask backend, ML models, and security features

from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import hashlib
import secrets
import time
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re
import random
import os

# 1. --- Configuration and Initialization ---

# Configure logging to show debug messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aadhaar_mfa.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Database
db = SQLAlchemy(app)


# 2. --- Database Models ---

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    aadhaar_hash = db.Column(db.String(64), unique=True, nullable=False)
    phone_number = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    behavioral_profiles = db.relationship('BehavioralProfile', backref='user', lazy=True)

class BehavioralProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    avg_dwell_time = db.Column(db.Float)
    avg_flight_time = db.Column(db.Float)
    avg_mouse_speed = db.Column(db.Float)
    confidence_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AuthSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_token = db.Column(db.String(64), unique=True, nullable=False)
    otp_code = db.Column(db.String(6))
    otp_expires_at = db.Column(db.DateTime)
    behavioral_score = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@dataclass
class BehavioralData:
    keystroke_dynamics: Dict
    mouse_dynamics: Dict


# 3. --- Core Logic Classes ---

class BehavioralBiometrics:
    """Handles ML model training and prediction for behavioral patterns."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def create_feature_vector(self, behavioral_data: BehavioralData) -> np.array:
        keystrokes = behavioral_data.keystroke_dynamics.get('keystrokes', [])
        dwell_times = [k.get('dwell_time', 0) for k in keystrokes]
        flight_times = [k.get('flight_time', 0) for k in keystrokes if k.get('flight_time', 0) > 0]
        
        keystroke_features = np.array([
            np.mean(dwell_times) if dwell_times else 0,
            np.std(dwell_times) if len(dwell_times) > 1 else 0,
            np.mean(flight_times) if flight_times else 0,
            np.std(flight_times) if len(flight_times) > 1 else 0
        ])

        mouse_events = behavioral_data.mouse_dynamics.get('mouse_events', [])
        movements = [m for m in mouse_events if m.get('type') == 'move']
        speeds = []
        if len(movements) > 1:
            for i in range(1, len(movements)):
                prev, curr = movements[i-1], movements[i]
                dx, dy = curr.get('x', 0) - prev.get('x', 0), curr.get('y', 0) - prev.get('y', 0)
                dt = curr.get('timestamp', 0) - prev.get('timestamp', 0)
                if dt > 5: speeds.append(np.sqrt(dx**2 + dy**2) / dt)
        
        mouse_features = np.array([
            np.mean(speeds) if speeds else 0,
            np.std(speeds) if len(speeds) > 1 else 0
        ])
        
        return np.concatenate([keystroke_features, mouse_features])

    def train_model(self, training_data: List[Tuple[BehavioralData, int]]):
        if len(training_data) < 10:
            logger.warning("Insufficient data for model training.")
            return False
        
        X = [self.create_feature_vector(data) for data, label in training_data]
        y = [label for data, label in training_data]
        X, y = np.array(X), np.array(y)
        
        X_scaled = self.scaler.fit_transform(np.nan_to_num(X))
        
        self.anomaly_detector.fit(X_scaled[y == 1])
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        return True

    def predict(self, behavioral_data: BehavioralData) -> Tuple[float, float]:
        if not self.is_trained: return (0.5, 0.5)

        features = np.nan_to_num(self.create_feature_vector(behavioral_data)).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        anomaly_normalized = max(0, min(1, (anomaly_score + 0.5)))
        
        proba = self.classifier.predict_proba(features_scaled)[0]
        classification_score = proba[1]
        
        behavioral_score = (anomaly_normalized + classification_score) / 2
        risk_score = 1 - behavioral_score
        
        return behavioral_score, risk_score

class AadhaarValidator:
    """Handles secure hashing of Aadhaar numbers."""
    @staticmethod
    def hash_aadhaar(aadhaar: str) -> str:
        salt = "a_very_secure_and_unique_salt_for_this_project_42"
        return hashlib.sha256((aadhaar + salt).encode()).hexdigest()

class OTPService:
    """Generates, sends (simulated), and validates OTPs."""
    @staticmethod
    def generate_otp() -> str:
        return str(random.randint(100000, 999999))

    @staticmethod
    def send_otp(phone_number: str, otp: str):
        logger.info(f"SIMULATING OTP: Sending OTP {otp} to {phone_number}")

    @staticmethod
    def is_otp_valid(stored_otp: str, input_otp: str, expires_at: datetime) -> bool:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        
        if datetime.now(timezone.utc) > expires_at:
            logger.warning("Attempt to use expired OTP.")
            return False
            
        return stored_otp == input_otp

class SecurityManager:
    """Calculates risk based on various factors."""
    @staticmethod
    def calculate_risk_score(session_data: Dict) -> float:
        risk_factors = {
            'behavioral_risk': session_data.get('behavioral_risk', 0) * 0.7,
            'new_device': 0.3 if session_data.get('new_device', False) else 0,
        }
        return min(1.0, sum(risk_factors.values()))

# Initialize the ML engine
biometric_engine = BehavioralBiometrics()


# 4. --- API Routes ---

@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        aadhaar = data.get('aadhaar')
        phone_number = data.get('phone_number')
        email = data.get('email')

        if not re.match(r'^\d{12}$', aadhaar):
            return jsonify({'error': 'Invalid Aadhaar number format'}), 400
        if not re.match(r'^\+?\d{10,15}$', phone_number):
            return jsonify({'error': 'Invalid phone number format'}), 400
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return jsonify({'error': 'Invalid email format'}), 400

        hashed_aadhaar = AadhaarValidator.hash_aadhaar(aadhaar)
        if User.query.filter_by(aadhaar_hash=hashed_aadhaar).first():
            return jsonify({'error': 'User with this Aadhaar number already exists'}), 409

        new_user = User(aadhaar_hash=hashed_aadhaar, phone_number=phone_number, email=email)
        db.session.add(new_user)
        db.session.commit()

        logger.info(f"New user registered with Aadhaar: {aadhaar}")
        return jsonify({'message': 'Registration successful!'}), 201

    except Exception as e:
        logger.error(f"Registration Error: {e}", exc_info=True)
        return jsonify({'error': 'Server error during registration'}), 500


@app.route('/api/authenticate/initiate', methods=['POST'])
def initiate_authentication():
    try:
        data = request.get_json()
        aadhaar = data.get('aadhaar')
        if not re.match(r'^\d{12}$', aadhaar):
            return jsonify({'error': 'Invalid Aadhaar number format'}), 400
            
        user = User.query.filter_by(aadhaar_hash=AadhaarValidator.hash_aadhaar(aadhaar)).first()
        if not user:
            return jsonify({'error': 'User not found. Please register.'}), 404
            
        otp = OTPService.generate_otp()
        session_token = secrets.token_hex(32)
        
        auth_session = AuthSession(
            user_id=user.id,
            session_token=session_token,
            otp_code=otp,
            otp_expires_at=datetime.now(timezone.utc) + timedelta(minutes=2)
        )
        db.session.add(auth_session)
        db.session.commit()
        
        OTPService.send_otp(user.phone_number, otp)
        
        return jsonify({'session_token': session_token, 'message': 'OTP sent successfully'}), 200
    except Exception as e:
        logger.error(f"Initiation Error: {e}", exc_info=True)
        return jsonify({'error': 'Server error during initiation'}), 500

@app.route('/api/authenticate/verify', methods=['POST'])
def verify_authentication():
    try:
        data = request.get_json()
        session_token = data.get('session_token')
        otp = data.get('otp')
        
        auth_session = AuthSession.query.filter(
            AuthSession.session_token == session_token,
            AuthSession.otp_expires_at > datetime.now(timezone.utc)
        ).first()

        if not auth_session:
            return jsonify({'error': 'Invalid or expired session token'}), 404
        
        if not OTPService.is_otp_valid(auth_session.otp_code, otp, auth_session.otp_expires_at):
            return jsonify({'error': 'Invalid or expired OTP'}), 401
        
        behavioral_data = BehavioralData(
            keystroke_dynamics=data.get('behavioral_data', {}).get('keystroke_dynamics', {}),
            mouse_dynamics=data.get('behavioral_data', {}).get('mouse_dynamics', {})
        )
        
        behavioral_score, risk_score = biometric_engine.predict(behavioral_data)
        user_profile = BehavioralProfile.query.filter_by(user_id=auth_session.user_id).first()
        
        overall_risk = SecurityManager.calculate_risk_score({
            'behavioral_risk': risk_score, 
            'new_device': not user_profile
        })
        
        auth_session.behavioral_score = behavioral_score
        auth_session.risk_score = overall_risk
        auth_session.is_verified = True
        
        if overall_risk < 0.4:
            auth_result = 'APPROVED'
        elif overall_risk < 0.75:
            auth_result = 'REVIEW'
        else:
            auth_result = 'DENIED'
        
        db.session.commit()
        
        return jsonify({
            'result': auth_result,
            'behavioral_score': float(behavioral_score),
            'risk_score': float(overall_risk),
        }), 200

    except Exception as e:
        logger.error(f"Verification Error: {e}", exc_info=True)
        return jsonify({'error': 'Server error during verification'}), 500


# 5. --- Frontend Route ---

@app.route('/')
def index():
    return render_template('index.html')


# 6. --- Initialization and Startup ---

def setup_application(app_context):
    """Initializes DB, creates test user, and trains model."""
    with app_context:
        db.create_all()
        logger.info("Database initialized.")

        # Create a test user if one doesn't exist - REMOVED for dynamic registration demo
        # test_aadhaar = "123456789012"
        # aadhaar_hash = AadhaarValidator.hash_aadhaar(test_aadhaar)
        # if not User.query.filter_by(aadhaar_hash=aadhaar_hash).first():
        #     user = User(aadhaar_hash=aadhaar_hash, phone_number="+919876543210", email="test@example.com")
        #     db.session.add(user)
        #     db.session.commit()
        #     logger.info(f"Test user created. Aadhaar: {test_aadhaar}")
        # else:
        #     logger.info("Test user already exists.")

        # Create and train the ML model with synthetic data
        sample_data = []
        for _ in range(50): # Legitimate data
            ks = [{'dwell_time': np.random.normal(120, 20), 'flight_time': np.random.normal(80, 15)} for _ in range(25)]
            ms = [{'type': 'move', 'x':0, 'y':0, 'timestamp': time.time()*1000 + j*20} for j in range(25)]
            sample_data.append((BehavioralData(keystroke_dynamics={'keystrokes': ks}, mouse_dynamics={'mouse_events': ms}), 1))
        
        for _ in range(20): # Attack data
            ks = [{'dwell_time': 100, 'flight_time': 50} for _ in range(25)]
            ms = [{'type': 'move', 'x':0, 'y':0, 'timestamp': time.time()*1000 + j*5} for j in range(25)]
            sample_data.append((BehavioralData(keystroke_dynamics={'keystrokes': ks}, mouse_dynamics={'mouse_events': ms}), 0))
            
        if biometric_engine.train_model(sample_data):
            logger.info("Behavioral biometrics model trained successfully.")

def create_template_file():
    """Creates the index.html file in a templates directory."""
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MFA System</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 500px; margin: 40px auto; padding: 20px; background-color: #f7f7f7; color: #333; }
            .container { background: #fff; padding: 25px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 20px; }
            h1, h3 { text-align: center; color: #222; }
            input, button { width: 100%; box-sizing: border-box; padding: 12px; margin-bottom: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; }
            button { background: #007aff; color: white; cursor: pointer; border: none; font-weight: 600; transition: background-color 0.2s; }
            button:hover { background: #0056b3; }
            button:disabled { background: #cce4ff; cursor: not-allowed; }
            .result { padding: 15px; margin-top: 20px; border-radius: 8px; border-left: 5px solid; font-weight: 500; }
            .success { background: #e9f7ef; color: #1d643b; border-color: #48c774; }
            .error { background: #feecf0; color: #cc0f35; border-color: #f14668; }
            .review { background: #fffbeb; color: #947600; border-color: #ffdd57; }
        </style>
    </head>
    <body>
        <h1>Multi-Factor Authentication</h1>

        <div class="container" id="register-container">
            <h3>New User Registration</h3>
            <input type="text" id="reg-aadhaar" placeholder="Enter 12-digit Aadhaar" maxlength="12">
            <input type="tel" id="reg-phone" placeholder="Enter Phone Number (e.g., +919876543210)" maxlength="15">
            <input type="email" id="reg-email" placeholder="Enter Email Address">
            <button id="reg-btn" onclick="registerUser()">Register</button>
            <p style="text-align:center; margin-top:15px;"><a href="#" onclick="showAuthSection()">Already registered? Authenticate here.</a></p>
        </div>

        <div class="container" id="init-container" style="display:none;">
            <h3>Step 1: Initiate Authentication</h3>
            <input type="text" id="auth-aadhaar" placeholder="Enter 12-digit Aadhaar" maxlength="12">
            <button id="init-btn" onclick="initiateAuth()">Send OTP</button>
            <p style="text-align:center; margin-top:15px;"><a href="#" onclick="showRegisterSection()">New user? Register here.</a></p>
        </div>
        <div class="container" id="verify-container" style="display:none;">
            <h3>Step 2: Verify Authentication</h3>
            <p style="text-align:center; color: #666; font-size: 14px;">OTP is in the console. Please type in the box below to provide behavioral data.</p>
            <input type="text" id="otp" placeholder="Enter 6-digit OTP" maxlength="6">
            <textarea id="behavior-text" placeholder="Type here..." style="height: 60px;"></textarea>
            <button id="verify-btn" onclick="verifyAuth()">Verify</button>
        </div>
        <div id="results"></div>
        <script>
            let sessionToken = '';
            let behavioralData = { keystroke_dynamics: { keystrokes: [] }, mouse_dynamics: { mouse_events: [] } };
            const initBtn = document.getElementById('init-btn');
            const verifyBtn = document.getElementById('verify-btn');
            const regBtn = document.getElementById('reg-btn');


            function attachListeners() {
                const area = document.getElementById('behavior-text');
                area.onkeydown = area.onkeyup = (e) => {
                    const ts = Date.now();
                    const lastKeystroke = behavioralData.keystroke_dynamics.keystrokes.length > 0 ? behavioralData.keystroke_dynamics.keystrokes.slice(-1)[0] : null;
                    
                    let dwellTime = 0;
                    let flightTime = 0;

                    if (e.type === 'keyup' && lastKeystroke && lastKeystroke.key === e.key && lastKeystroke.type === 'keydown') {
                        dwellTime = ts - lastKeystroke.timestamp;
                    }
                    if (lastKeystroke) {
                        flightTime = ts - lastKeystroke.timestamp;
                    }

                    behavioralData.keystroke_dynamics.keystrokes.push({
                        key: e.key, 
                        type: e.type, 
                        timestamp: ts,
                        dwell_time: dwellTime,
                        flight_time: flightTime
                    });
                };
                document.onmousemove = (e) => {
                    behavioralData.mouse_dynamics.mouse_events.push({ type: 'move', x: e.clientX, y: e.clientY, timestamp: Date.now() });
                };
            }

            function showAuthSection() {
                document.getElementById('register-container').style.display = 'none';
                document.getElementById('init-container').style.display = 'block';
                document.getElementById('verify-container').style.display = 'none';
                document.getElementById('results').innerHTML = ''; // Clear results on section switch
            }

            function showRegisterSection() {
                document.getElementById('register-container').style.display = 'block';
                document.getElementById('init-container').style.display = 'none';
                document.getElementById('verify-container').style.display = 'none';
                document.getElementById('results').innerHTML = ''; // Clear results on section switch
            }


            async function apiCall(endpoint, body, button) {
                button.disabled = true;
                button.innerText = 'Processing...';
                try {
                    const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || 'An unknown error occurred.');
                    return result;
                } catch (error) {
                    showResult(error.message, 'error');
                    throw error;
                } finally {
                    button.disabled = false;
                }
            }

            async function registerUser() {
                const aadhaar = document.getElementById('reg-aadhaar').value;
                const phone = document.getElementById('reg-phone').value;
                const email = document.getElementById('reg-email').value;

                if (!/^\\d{12}$/.test(aadhaar)) { return showResult('Please enter a valid 12-digit Aadhaar for registration.', 'error'); }
                if (!/^\\+?\\d{10,15}$/.test(phone)) { return showResult('Please enter a valid phone number for registration.', 'error'); }
                if (!/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/.test(email)) { return showResult('Please enter a valid email address for registration.', 'error'); }

                try {
                    const result = await apiCall('/api/register', { aadhaar, phone_number: phone, email }, regBtn);
                    showResult('Registration successful! You can now authenticate.', 'success');
                    // Optionally pre-fill Aadhaar for authentication or switch to auth section
                    document.getElementById('auth-aadhaar').value = aadhaar;
                    showAuthSection();
                } catch(e) { /* error handled in apiCall */ }
                regBtn.innerText = 'Register';
            }


            async function initiateAuth() {
                const aadhaar = document.getElementById('auth-aadhaar').value;
                if (!/^\\d{12}$/.test(aadhaar)) { return showResult('Please enter a valid 12-digit Aadhaar.', 'error'); }
                
                try {
                    const result = await apiCall('/api/authenticate/initiate', { aadhaar }, initBtn);
                    sessionToken = result.session_token;
                    document.getElementById('verify-container').style.display = 'block';
                    document.getElementById('init-container').style.display = 'none';
                    showResult('OTP "sent" (check console). Please type in the box and verify.', 'success');
                    attachListeners();
                } catch(e) { /* error handled in apiCall */ }
                initBtn.innerText = 'Send OTP';
            }

            async function verifyAuth() {
                const otp = document.getElementById('otp').value;
                if (!/^\\d{6}$/.test(otp)) { return showResult('Please enter a valid 6-digit OTP.', 'error'); }
                
                try {
                    const result = await apiCall('/api/authenticate/verify', { session_token: sessionToken, otp, behavioral_data: behavioralData }, verifyBtn);
                    const resultClass = result.result.toLowerCase();
                    showResult(`<h3>${result.result}</h3><p>Behavioral Score: <b>${(result.behavioral_score * 100).toFixed(1)}%</b><br>Overall Risk: <b>${(result.risk_score * 100).toFixed(1)}%</b></p>`, resultClass);
                    // Reset behavioral data for next authentication attempt
                    behavioralData = { keystroke_dynamics: { keystrokes: [] }, mouse_dynamics: { mouse_events: [] } };
                    document.getElementById('behavior-text').value = '';
                    document.getElementById('otp').value = '';

                    document.getElementById('verify-container').style.display = 'none';
                    document.getElementById('init-container').style.display = 'block';
                } catch(e) { /* error handled in apiCall */ }
                verifyBtn.innerText = 'Verify';
            }

            function showResult(message, type) {
                document.getElementById('results').innerHTML = `<div class="result ${type}">${message}</div>`;
            }
        </script>
    </body>
    </html>
    """
    with open('templates/index.html', 'w') as f:
        f.write(html_content)


if __name__ == '__main__':
    create_template_file()
    setup_application(app.app_context())
    
    logger.info("=" * 60)
    logger.info(" MFA SYSTEM READY")
    logger.info("=" * 60)
    # logger.info("-> Test user created (Aadhaar: 123456789012)") # Removed automatic test user for registration demo
    logger.info("-> Behavioral biometrics model trained.")
    logger.info(f"-> Server starting on http://127.0.0.1:5000")
    logger.info("=" * 60)
    
    app.run(debug=True, port=5000)