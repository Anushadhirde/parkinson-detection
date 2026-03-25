import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from predict.svm.predict import run_prediction_detailed

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max limit
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part found in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.lower().endswith('.wav'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Predict
            prediction_data = run_prediction_detailed(filepath)
            
            # Predict returns a dict with 'result' ('healthy' / 'parkinson'), 'confidence', 'details'
            return jsonify(prediction_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file format. Please upload a .wav file.'}), 400

if __name__ == '__main__':
    # Using port 5000 by default. Waitress or Gunicorn can be used for production.
    app.run(host='0.0.0.0', port=5000, debug=True)
