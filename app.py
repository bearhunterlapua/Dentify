from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
from db_models import db, PatientRecord, BoundingBox

import os
from werkzeug.utils import secure_filename

# Import the detection function
from cavity_detection import run_cavity_detection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize the db with the app
# db.init_app(app)

# Create tables (simplistic approach—use migrations in a real project)
# with app.app_context():
#     db.create_all()

# Example route
# @app.route('/')
# def index():
#     return 'Hello, world!'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Call our detection function
    try:
        detections = run_cavity_detection(filepath)
        # Example: detections might look like [[100, 100, 200, 200, 2], [50, 50, 80, 80, 3], ...]
        status = "Detection done."
    except Exception as e:
        detections = []
        status = f"Error: {e}"

    # Return a template that can display these detections 
    return render_template(
        'annotate.html',
        image_url=url_for('uploaded_file', filename=filename),
        detections=detections,
        status=status
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)