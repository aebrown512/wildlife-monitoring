import os
import uuid
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import json
from shapely import wkt
from shapely.geometry import mapping
from coyote_tracker import Coyote_Tracker
from config import *
app = Flask(__name__)
def allow(filename,extensions={'csv','wkt','geojson'}):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    ses_id = str(uuid.uuid4())
    ses_dir=app.config['UPLOAD_FOLDER'] / ses_id
    ses_dir.mkdir(exist_ok=True)
    gps_f = request.files.get('gps_f')
    if not gps_f or not allow(gps_f.filename, {'csv'}):
        return jsonify({'error': 'No file uploaded or invalid file type'}), 400
    gps_path = ses_dir / secure_filename(gps_f.filename)
    gps_f.save(gps_path)

    urban_p=None
    urban_f=request.files.get('urban_f')
    if urban_f and allow(urban_f.filename, {'wkt','geojson'}):
        urban_p=ses_dir / secure_filename(urban_f.filename)
        urban_f.save(urban_p)

    road_p=None
    road_f=request.files.get('road_f')
    if road_f and allow(road_f.filename, {'wkt','geojson'}):
        road_p=ses_dir / secure_filename(road_f.filename)
        road_f.save(road_p)

    try:
        urban_p=None
        if urban_p:
            with open(urban_p,'r') as f:
                urban_p=wkt.loads(f.read())
        road_n=None
        from shapely.geometry import shape
        if road_p:
            with open(road_p,'r') as f:
                d=json.load(f)
            road_n=[]
            for f in d.get('features',[]):
                if f['geometry']['type']=='LineString':
                    road_n.append(shape(f['geometry']))
        
        tracker=Coyote_Tracker(gps_path,urban_p,road_n)
        result=tracker.pipeline()

    
    
