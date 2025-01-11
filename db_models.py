# models.py

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PatientRecord(db.Model):
    __tablename__ = 'patient_records'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(120), nullable=False)
    patient_age = db.Column(db.Integer, nullable=False)
    patient_gender = db.Column(db.String(50), nullable=False)
    medical_history = db.Column(db.Text, nullable=True)
    xray_filename = db.Column(db.String(255), nullable=True)

    # Relationship to bounding boxes
    boxes = db.relationship("BoundingBox", backref="patient_record", lazy=True)

class BoundingBox(db.Model):
    __tablename__ = 'bounding_boxes'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_record_id = db.Column(db.Integer, db.ForeignKey('patient_records.id'), nullable=False)

    x1 = db.Column(db.Float, nullable=False)
    y1 = db.Column(db.Float, nullable=False)
    x2 = db.Column(db.Float, nullable=False)
    y2 = db.Column(db.Float, nullable=False)
    cls = db.Column(db.Integer, nullable=False)