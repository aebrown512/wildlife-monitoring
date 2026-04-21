import pandas as pd
import numpy as np
from datetime import datetime
from geopy.distance import geodesic
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from shapely.geometry import Point as ShapelyPoint, Polygon
import warnings
from config import *

class Coyote_Tracker:
    def __init__(self,gps_csv_path, urban_pgon= None, road_nwork= None):
        self.raw_df = pd.read_csv(gps_csv_path)
        self.urban = urban_pgon
        self_roads = road_nwork
        self.df = None
        self.behavior_model = None

    def preproc(self):
        df=self.raw_df.copy()
        df=df.dropna(subset=['latitude','longitude'])
        if 'hdop' in df.columns:
            df = df[df['hdop'] <= M_HDOP]
        df=df[(df['latitude'].between(40,43)) & (df['longitude'].between(-87,-90))]
        df=df.sort_values('timestamp').reset_index(drop=True)
        df=df.drop_duplicates(subset=['timestamp'])
        df['dt_sec']=df['timestamp'].diff().dt.total_seconds()
        df=df.drop(columns=['dt_sec'],errors='ignore')
        df=df[(df['dt_sec'] >= MINTIME_DELTA) | (df['dt_sec'].isna())]
        self.df=df
        return self
    
    def behavior_classified(self, method='gm'):
        df=self.df
        if method =='threshold':
            conditions = [df['speed_ms'] <= REST_SPEED, (df['speed_ms'] > REST_SPEED) & (df['speed_ms'] <= FORAGING_MSPEED), df['speed_ms'] > FORAGING_MSPEED]
            choices = ['Resting', 'Foraging', 'Traveling']
            df['behavior'] = np.select(conditions, choices, default='Unknown')
        else:
            f=df[['step','turn_angle']].dropna().copy()
            if len(f)<10:
                df['behavior']='Unknown'
                return self
            x=np.column_stack([np.log1p(f['step']),f['turn_angle']/180.0])
            gm= GaussianMixture(n_compounds=3, random_state=42)
            label=gm.fit_predict(x)
            means=gm.means_[:,0]
            order=np.argsort(means)
            label_behavior= {order[0]:'Resting',order[1]:'Foraging',order[2]:'Traveling'}
            behavior=[label_behavior[1] for l in label]
            idx=f.index
            df.loc[idx,'behavior']=behavior
            df['behavior']=df['behavior'].fillna(method='ffill').fillna('Unknown')
        self.df=df
        return self