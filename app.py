from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize classifier - try to load, but make it optional
classifier = None
classifier_error = None
try:
    from src.classify import RoadDamageClassifier
    classifier = RoadDamageClassifier()
    print("Classifier loaded successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FULL ERROR:")
    print(str(e))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/upload')
def upload():
    return render_template('upload-page.html')

@app.route('/result')
def result():
    return render_template('result-page.html')

@app.route('/info')
def info():
    return render_template('info-page.html')

@app.route('/api/classify', methods=['POST'])
def classify_video():
    # Check if classifier is available
    if classifier is None:
        error_msg = classifier_error if classifier_error else 'Model not loaded'
        return jsonify({'error': f'Model tidak dapat dimuat: {error_msg}'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            print(f"Processing video: {filepath}")
            # Process the video
            result = classifier.process_video(filepath)
            print(f"Classification result: {result}")

            result['filename'] = filename
            return jsonify(result)

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Gagal memproses video: {str(e)}'}), 500

    return jsonify({'error': 'Jenis file tidak valid. Gunakan MP4, AVI, MOV, atau MKV'}), 400

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

@app.route('/asset/<path:filename>')
def asset_files(filename):
    return send_from_directory('asset', filename)

@app.route('/uploads/<path:filename>')
def uploaded_files(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    # Disable debug mode's reloader to avoid threading issues with TensorFlow
    app.run(debug=False, host='0.0.0.0', port=8000, use_reloader=False)
