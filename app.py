from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import pdfkit  # For PDF generation
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Load ML Model
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
DEFAULT_FEATURE_COLUMNS = [
    'age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
    'fasting blood sugar', 'resting ecg', 'max heart rate', 'exercise angina',
    'oldpeak', 'ST slope'
]


def load_artifact(filename):
    artifact_path = MODELS_DIR / filename
    if not artifact_path.exists():
        raise FileNotFoundError(f'Missing model artifact: {artifact_path}')
    return joblib.load(artifact_path)


def load_feature_columns():
    columns_path = MODELS_DIR / "feature_names.pkl"
    if not columns_path.exists():
        joblib.dump(DEFAULT_FEATURE_COLUMNS, columns_path)
    return joblib.load(columns_path)


def prepare_model_input(form_data, columns):
    chest_pain_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
    ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
    slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}
    angina_map = {'N': 0, 'Y': 1}

    input_values = {col: 0.0 for col in columns}
    input_values['age'] = float(form_data['age'])
    input_values['sex'] = 1.0 if form_data['sex'] == 'Male' else 0.0
    input_values['chest pain type'] = float(chest_pain_map.get(form_data['chest_pain'], 0))
    input_values['resting bp s'] = float(form_data['resting_bp'])
    input_values['cholesterol'] = float(form_data['cholesterol'])
    input_values['fasting blood sugar'] = float(form_data['fasting_bs'])
    input_values['resting ecg'] = float(ecg_map.get(form_data['resting_ecg'], 0))
    input_values['max heart rate'] = float(form_data['max_hr'])
    input_values['exercise angina'] = float(angina_map.get(form_data['exercise_angina'], 0))
    input_values['oldpeak'] = float(form_data['oldpeak'])
    input_values['ST slope'] = float(slope_map.get(form_data['st_slope'], 0))

    return pd.DataFrame([input_values], columns=columns).astype(float)


def get_default_profile_data():
    return {
        'age': 50,
        'sex': 1,
        'cp': 0,
        'trestbps': 120,
        'chol': 200,
        'fbs': 0,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 0.0,
        'slope': 0,
    }


def get_profile_data_from_prediction(prediction):
    return {
        'age': prediction.age,
        'sex': 1 if prediction.sex == 'Male' else 0,
        'cp': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}.get(prediction.chest_pain_type, 0),
        'trestbps': prediction.resting_bp,
        'chol': prediction.cholesterol,
        'fbs': prediction.fasting_bs,
        'restecg': {'Normal': 0, 'ST': 1, 'LVH': 2}.get(prediction.resting_ecg, 0),
        'thalach': prediction.max_hr,
        'exang': 1 if prediction.exercise_angina == 'Y' else 0,
        'oldpeak': prediction.oldpeak,
        'slope': {'Up': 0, 'Flat': 1, 'Down': 2}.get(prediction.st_slope, 0),
    }


model = load_artifact('best_model.pkl')
scaler = load_artifact('scaler (2).pkl')
feature_columns = load_feature_columns()

# ============ Database Models ============
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(10))
    chest_pain_type = db.Column(db.String(50))
    resting_bp = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    fasting_bs = db.Column(db.Integer)
    resting_ecg = db.Column(db.String(50))
    max_hr = db.Column(db.Integer)
    exercise_angina = db.Column(db.String(10))
    oldpeak = db.Column(db.Float)
    st_slope = db.Column(db.String(20))
    prediction_result = db.Column(db.String(20))
    probability = db.Column(db.Float)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============ Routes ============

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    latest = session.get('last_prediction')

    if latest is None:
        latest_prediction = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.prediction_date.desc()).first()
        if latest_prediction is not None:
            latest = get_profile_data_from_prediction(latest_prediction)
        else:
            latest = get_default_profile_data()

    return render_template('profile.html', data=latest)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    probability = None
    
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        cp = request.form['chest_pain']
        trestbps = int(request.form['resting_bp'])
        chol = int(request.form['cholesterol'])
        fbs = int(request.form['fasting_bs'])
        restecg = request.form['resting_ecg']
        thalach = int(request.form['max_hr'])
        exang = request.form['exercise_angina']
        oldpeak = float(request.form['oldpeak'])
        slope = request.form['st_slope']
        
        input_df = prepare_model_input(request.form, feature_columns)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        result = 'Positive' if prediction == 1 else 'Negative'
        probability = round(prob * 100, 2)
        
        # Save to database
        pred = Prediction(
            user_id=current_user.id,
            age=age,
            sex=sex,
            chest_pain_type=cp,
            resting_bp=trestbps,
            cholesterol=chol,
            fasting_bs=fbs,
            resting_ecg=restecg,
            max_hr=thalach,
            exercise_angina=exang,
            oldpeak=oldpeak,
            st_slope=slope,
            prediction_result=result,
            probability=probability
        )
        db.session.add(pred)
        db.session.commit()

        session['last_prediction'] = {
            'age': age,
            'sex': 1 if sex == 'Male' else 0,
            'cp': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}.get(cp, 0),
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': {'Normal': 0, 'ST': 1, 'LVH': 2}.get(restecg, 0),
            'thalach': thalach,
            'exang': 1 if exang == 'Y' else 0,
            'oldpeak': oldpeak,
            'slope': {'Up': 0, 'Flat': 1, 'Down': 2}.get(slope, 0),
        }
        session.modified = True
        
        flash(f'Prediction: {result} (Confidence: {probability}%)', 'success')
    
    return render_template('predict.html', result=result, probability=probability)

@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.prediction_date.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/report/<int:pred_id>')
@login_required
def generate_report(pred_id):
    pred = Prediction.query.get_or_404(pred_id)

    if pred.user_id != current_user.id:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('history'))

    html_content = render_template('report_template.html', pred=pred, user=current_user)

    try:
        if os.environ.get('WKHTMLTOPDF_PATH'):
            config = pdfkit.configuration(wkhtmltopdf=os.environ['WKHTMLTOPDF_PATH'])
            pdf = pdfkit.from_string(html_content, False, configuration=config)
        else:
            pdf = pdfkit.from_string(html_content, False)

        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=heart_report_{pred.id}.pdf'
        return response
    except Exception:
        response = make_response(html_content)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=heart_report_{pred.id}.html'
        flash('PDF engine is not available on this system, so an HTML report was downloaded instead.', 'info')
        return response

@app.route('/delete_history/<int:pred_id>')
@login_required
def delete_history(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id == current_user.id:
        db.session.delete(pred)
        db.session.commit()
        flash('Prediction deleted successfully!', 'success')
    else:
        flash('Unauthorized!', 'danger')
    return redirect(url_for('history'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)