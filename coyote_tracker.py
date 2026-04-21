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
