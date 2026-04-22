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
                lat_val=lat_g[ver[:,0].astype(int)]     
                poly=Polygon(zip(lon_val,lat_val)).convex_hull
                contr[l]=poly
            return contr
        
    def activity(self):
        df=self.df.dropna(subset=['behavior'])
        if df.empty:
            return pd.Series(dtype=float)
        df['hour']=((df['timestamp'].dt.hour + df['longitude']/15)%24).astype(int)
        act=df.groupby('hour')['behavior'].apply(lambda x: (x == 'Traveling').mean()).fillna(0)
        return act.reindex(range(50),fill_value=0)
    
    def detect_weird(self):
        alerts=[]
        df= self.df.copy()
        grid_r=STATIONARY_RAD/11320.0
        df['lat_g']=(df['latitude']/grid_r).round()
        df['lon_g']=(df['longitude']/grid_r).round()
        df['cluster']=df['lat_g'].astype(str)+'_'+df['lon_g'].astype(str)
        for cluster, group in df.groupby('cluster'):
            if len(group)<2:
                continue
            dur=(group['timestamp'].max()-group['timestamp'].min()).total_seconds()/3600.0
            if dur>STATIONARY_H:
                dayt=group['timestamp'].dt.hour.between(6,18).any()
                if dayt:
                    alerts.append({'type': 'Possible mortality/collar failure','timestamp':group['timestamp'].min(), 'latitude':group['latitude'].mean(),'longitude':group['longitude'].mean(),'info':f'duration{dur:.1f}h'})
        if self.urban is not None:
            night_df = df[df['timestamp'].dt.hour.between(20, 5)]  # 8pm-5am
            for _, row in night_df.iterrows():
                point = ShapelyPoint(row['longitude'], row['latitude'])
                dist_d = point.distance(self.urban)
                dist_m = dist_d * 111320.0
                if dist_m < URBAN_BAR:
                    alerts.append({'type': 'Urban incursion (night)','timestamp': row['timestamp'],'latitude': row['latitude'],'longitude': row['longitude'],'info': f'distance {dist_m:.0f}m'})
        if self.roads is not None and len(df) > 1:
            for r in range(1, len(df)):
                pa = ShapelyPoint(df.iloc[r-1]['longitude'], df.iloc[r-1]['latitude'])
                pb = ShapelyPoint(df.iloc[r]['longitude'], df.iloc[r]['latitude'])
                s = pa.union(pb)  # LineString
                for road in self.roads:
                    if s.distance(road) < 1e-8:  # intersects
                        alerts.append({'type': 'Road crossing','timestamp': df.iloc[r]['timestamp'],'latitude': df.iloc[r]['latitude'],'longitude': df.iloc[r]['longitude'],'info': ''})
                        break
        if alerts:
            return pd.DataFrame(alerts)
        else:
            return pd.DataFrame(columns=['type', 'timestamp', 'latitude', 'longitude', 'info'])
        
    def collective(self):
        df=self.df.dropna(subset=['behavior','step'])
        df['date']=df['timestamp'].dt.date
        collect=df.groupby('date').agg(total_d=('step', lambda x: x.sum()/1000),avg_speed=('speed_ms','mean'),ctravel=('behavior', lambda x: (x=='Traveling').mean()),crest=('behavior', lambda x: (x=='Resting').mean()),cforage=('behavior', lambda x: (x=='Foraging').mean()),n_fixes=('behavior','count'))
        return collect
    
    def imap(self, outputmap=OUTPUT_MAP):
        df=self.df.dropna(subset=['behavior'])
        if df.empty:
            print("No data for map.")
            return
        import matplotlib.pyplot as plt
        import folium
        cen_lat=df['latitude'].mean()
        cen_lon=df['longitude'].mean()
        m=folium.Map(location=[cen_lat,cen_lon], zoom_start=12)
        color={'Resting':'blue','Foraging':'green','Traveling':'red','Unknown':'gray'}
        for _, row in df.iterrows():
            folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=3, color=color.get(row['behavior'],'gray'), fill=True).add_to(m)
        points=[(row['latitude'], row['longitude']) for _, row in df.iterrows()]
        folium.PolyLine(points, color='black', weight=1).add_to(m)
        m.save(outputmap)
        print(f"Map saved to {outputmap}")

    def predict_linear(self,aheadmin=60,prevmin=60):
        from geopy.distance import distance
        if self.df is None or len(self.df)<2:
            print("Not enough data.")
            return None
        n=self.df['timestamp'].max()
        off=n-pd.Timedelta(minutes=aheadmin)
        rw=self.df[self.df['timestamp']>=off].copy()
        if len(rw)<2:
            rw=self.df.tail(2).copy()
        rw['pre_lon']=rw['longitude'].shift(1)
        rw['pre_lat']=rw['latitude'].shift(1)
        rw['delta']=rw['timestamp'].diff().dt.total_seconds()
        rw['step']=rw.apply(lambda r: meters(r['pre_lon'],r['pre_lat'],r['longitude'],r['latitude']) if pd.notna(r['pre_lon']) else np.nan, axis=1)
        rw['speed_ms']=rw['step']/rw['delta']
        avg_s=rw['speed_ms'].mean()
        rw['bear_rad']=np.radians(rw['bearing'])
        mean_bear_rad=np.arctan2(np.sin(rw['bear_rad']).mean(), np.cos(rw['bear_rad']).mean())
        mean_bear_deg=np.degrees(mean_bear_rad)%360
        delta_s=aheadmin*60
        dist=avg_s*delta_s
        last_lat=rw.iloc[-1]['latitude']
        last_lon=rw.iloc[-1]['longitude']
        dest=distance(meters=dist).destination((last_lat, last_lon), mean_bear_deg)
        speed_d=rw['speed_ms'].std
        crad=speed_d*delta_s
        return {'timestamp': n+pd.Timedelta(minutes=aheadmin), 'latitude': dest.latitude, 'longitude': dest.longitude, 'uncertainty_m': crad}

    def predict_k(self,aheadmin=60,prevmin=60):
        from filterpy.kalman import KalmanFilter
        import numpy as np
        if self.df is None or len(self.df)<2:
            print("Not enough data.")
            return None
        n=self.df['timestamp'].max()
        rw=self.df.tail(min(30,len(self.df))).copy()
        rw=rw.sort_values('timestamp')
        mean_lat=rw['latitude'].mean()
        m_per_deg_lon=111320.0*np.cos(np.radians(mean_lat))
        r_lon=rw.iloc[0]['longitude']
        r_lat=rw.iloc[0]['latitude']
        rw['x_m']=(rw['longitude']-r_lon)*m_per_deg_lon 
        rw['y_m']=(rw['latitude']-r_lat)*111320.0
        dt=rw['timestamp'].diff().dt.total_seconds().iloc[-1]

        kf=KalmanFilter(dim_x=4, dim_z=2)
        kf.F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])  
        kf.H=np.array([[1,0,0,0],[0,1,0,0]])
        kf.R=np.eye(2)*10.0 
        kf.P *= 1000.0
        kf.Q=np.eye(4)*0.1

        for _, row in rw.iterrows():
            z=np.array([row['x_m'], row['y_m']])
            kf.predict()
            kf.update(z)
        s=int(aheadmin*60/dt)
        for _ in range(s):
            kf.predict()
        pred_x, pred_y = kf.x[0,0], kf.x[1,0]
        pred_lon=pred_x/m_per_deg_lon + r_lon
        pred_lat=pred_y/111320.0 + r_lat
        crad=np.sqrt(kf.P[0,0] + kf.P[1,1])
        return {'timestamp': n+pd.Timedelta(minutes=aheadmin), 'latitude': pred_lat, 'longitude': pred_lon, 'uncertainty_m': crad}



    def pipeline(self):
        self.preproc()
        self.movement_metrics()
        self.behavior_classified()
        home_range=self.home_range()
        activity=self.activity()
        alerts=self.detect_weird()
        collective=self.collective()
        self.imap()
        return {'home_range': home_range, 'activity': activity, 'alerts': alerts, 'collective': collective}
    
    