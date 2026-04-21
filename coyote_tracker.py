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

def meters(lon1,lat1,lon2,lat2):
    return geodesic((lat1,lon1),(lat2,lon2)).meters

def bearing(lon1,lat1,lon2,lat2):
    from math import atan2, cos, sin, radians, degrees
    lat1,lon1,lat2,lon2=map(radians, [lat1,lon1,lat2,lon2])
    lon3=lon2-lon1
    m=sin(lon3)*cos(lat2)
    n=cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon3)
    return degrees(atan2(m,n))%360

def turning_a(b,c):
    dif=abs(b-c)%360
    return dif if dif <= 180 else 360 - dif

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
    def movement_metrics(self):
        df=self.df
        df['pre_lon']=df['longitude'].shift(1)
        df['pre_lat']=df['latitude'].shift(1)
        df['pre_time']=df['timestamp'].shift(1)
        df['step']=df.apply(lambda r: meters(r['pre_lon'],r['pre_lat'],r['longitude'],r['latitude']) 
                            if pd.notna(r['pre_lon']) else np.nan, axis=1)
        df['delta']=(df['timestamp']-df['pre_time']).dt.total_seconds()
        df['speed_ms']=df['step']/df['delta']
        df['bearing']=df.apply(lambda r: bearing(r['pre_lon'],r['pre_lat'],r['longitude'],r['latitude'])
                               if pd.notna(r['pre_lon']) else np.nan, axis=1)
        df['turn_angle']=df['bearing'].diff().abs.apply(lambda x:x if x<=180 else 360-x)
        df.loc[df['speed_ms']> MAX_CSPEED, 'speed_ms'] = np.nan
        df.loc[df['step']> 10000, 'step']=np.nan
        self.df=df
        return self
    
    def home_range(self,method='kl',levels=K_LEVELS):
        cords= self.df[['longitude','latitude']].dropna().values
        if len(cords)<5:
            print("Not enough data for range estimation.")
            return None
        if method == 'm':
            hull = ConvexHull(cords)
            hull_p = cords[hull.vertices]
            polygon = Polygon(hull_p)
            return {100: polygon}
        else:
            import matplotlib.pyplot as plt
            contr={}
            lon_g=np.linspace(cords[:,0].min()-0.01, cords[:,0].max()+0.01, K_GRID)
            lat_g=np.linspace(cords[:,0].min()-0.01, cords[:,0].max()+0.01, K_GRID)
            x,y=np.meshgrid(lon_g,lat_g)
            pos=np.vstack([x.ravel(),y.ravel()])
            ker=gaussian_kde(cords.T,b_method='scott')
            z=ker(pos).reshape(x.shape)
            for l in levels:
                plt.figure()
                thr=np.percentile(z, 100-l) 
                css= plt.contour(x,y,z, levels=[thr])   
                path=css.collections[0].get_paths()
                plt.close()
                if not path:
                    continue
                ver=path[0].vertices
                lon_val=lon_g[ver[:,0].astype(int)] 
                lat_val=lon_g[ver[:,0].astype(int)]     
                poly=Polygon(zip(lon_val,lat_val)).convex_hull
                contr[l]=poly
            return contr
    def activity(self):
        df=self.df.dropna(subset=['behavior'])
        if df.empty:
            return pd.Series(dtype=float)
        df['hour']=((df['timestamp'].dt.hour + df['longitude']/15)%24).astype(int)
        