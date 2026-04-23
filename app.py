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
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
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

        moutput=ses_dir/'map.html'
        tracker.imap(str(moutput))
        collect=result['collect'].reset_index().to_dict(orient='records')
        alert=result['alert'].to_dict(orient='records') if not result['alert'].empty else []

        try:
            pre=tracker.predict_linear(aheadmin=60)
            predict={
                'timestamp': str(pre['timestamp']),
                'latitude': pre['latitude'],
                'longitude': pre['longitude'],
                'cradius_m': pre['cradius_m']
            }
        except Exception:
            predict=None

        process_csv=ses_dir/'process.csv'
        result['process'].to_csv(process_csv, index=False)
        return jsonify({
            'success': True,
            'session_id': ses_id,
            'map_url': url_for('get_file', session_id=ses_id, filename='map.html'),
            'processed_csv_url': url_for('get_file', session_id=ses_id, filename='tracked_coyote.csv'),
            'collect': collect,
            'alerts': alert,
            'prediction': predict,
            'stats': {
                'num_fixes': len(result['trajectory']),
                'total_distance_km': result['daily_summary']['total_distance_km'].sum() if not result['collect'].empty else 0,
                'home_range_area_km2': None
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_file/<ses_id>/<filename>')    
def get_file(ses_id, filename):
    file_path = app.config['UPLOAD_FOLDER'] / ses_id / filename
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path)

@app.route('/predict', methods=['POST'])
def predict():
    d=request.get_json()
    ses_id=d.get('session_id')
    aheadmin=int(d.get('aheadmin',60))
    method=d.get('method','linear')
    ses_dir=app.config['UPLOAD_FOLDER'] / ses_id
    gps_p=ses_dir/'tracked_coyote.csv'
    if not gps_p.exists():
        return jsonify({'error': 'Session not found'}), 404
    try:
        track = Coyote_Tracker(str(gps_p))
        track.preprocess()
        track.compute_movement_metrics()
        track.classify_behavior(method='gm')
        if method == 'linear':
            pred = track.predict_linear(aheadmin=aheadmin)
        else:
            pred = track.predict_k(aheadmin=aheadmin)
        return jsonify({
            'timestamp': str(pred['timestamp']),
            'latitude': pred['latitude'],
            'longitude': pred['longitude'],
            'cradius_m': pred['confidence_radius_m']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
