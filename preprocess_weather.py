"""
merge_weather_trainstations.py

Purpose:
- Load weather observations, weather station metadata, and train station metadata.
- Optionally remove weather stations with many missing values or interpolate missing FH, T, RH.
- Map each train station to the nearest weather station (by lat/lon) and produce a dataset that
  contains, for every datetime available in the weather dataset, the weather values at each train station
  (taken from the nearest weather station).

Assumptions & notes:
- Weather file has columns: STN, YYYYMMDD (int like 20250130), HH (1-24), FH, T, RH, M, R, S, O, Y
  HH==24 is treated as hour 0 of the next day.
- Weather metadata file has: STN, lon, lat, alt, name
- Train stations file has: id, code, uic, name_short, ..., geo_lat, geo_lng
- We operate on FH (rain float), T (temp float), RH (wind speed float) for missing checks/interp.
- The script offers two strategies for stations with many missing values: 'remove' or 'interpolate'.

Usage example:
python merge_weather_trainstations.py \
  --weather observations.csv \
  --weather_meta weather_stations.csv \
  --train_stations train_stations.csv \
  --out merged_train_weather.csv \
  --mode interpolate --missing_thresh 0.2

Output:
- CSV (or parquet if desired) with rows: train_id, train_name, datetime, FH, T, RH, STN (source), ...

"""

import argparse
import math
import numpy as np
import pandas as pd
from datetime import timedelta
import sys

weather_csv = "data/processed/weather_merged.csv"
weather_stations_csv = "data/processed/weather_stations.csv"
train_stations_csv = "data/train_stations/stations-2023-09.csv"
output_csv = "data/processed/train_weather_merged.csv"
# ---------------------- utilities ----------------------

mode = 'interpolate'   # or 'remove'
missing_thresh = 0.2   # fraction threshold
max_neighbors = 5       # for spatial fill
verbose = True

# ---------------------- utilities ----------------------

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

def parse_weather_datetime(df):
    df = df.copy()
    df['YYYYMMDD'] = df['YYYYMMDD'].astype(int)
    df['HH'] = df['HH'].astype(int)
    df['date'] = pd.to_datetime(df['YYYYMMDD'].astype(str), format='%Y%m%d')
    df['hour0'] = df['HH'] % 24
    df['datetime'] = df['date'] + pd.to_timedelta(df['hour0'], unit='h')
    df.drop(columns=['date','hour0'], inplace=True)
    return df

def compute_missing_fraction(df, cols):
    g = df.groupby('STN')[cols].agg(lambda x: x.isna().mean())
    g.columns = [c + '_missing_frac' for c in cols]
    return g

def temporal_interpolate_station(df_station):
    df = df_station.sort_values('datetime').set_index('datetime')
    vals = df[['FH','T','RH']]
    vals_interp = vals.interpolate(method='time', limit_direction='both')
    df.loc[:, ['FH','T','RH']] = vals_interp
    return df.reset_index()

def spatial_fill_remaining(df_weather, weather_meta, max_neighbors=5, eps=1e-6):
    df = df_weather.copy()
    stations = weather_meta['STN'].values
    lon = weather_meta.set_index('STN').loc[stations]['lon'].astype(float).values
    lat = weather_meta.set_index('STN').loc[stations]['lat'].astype(float).values
    dist_mat = haversine_np(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
    stn_to_idx = {stn: i for i, stn in enumerate(stations)}
    vars_to_fill = ['FH','T','RH']


    # Use a MultiIndex for proper indexing by datetime and STN
    df_indexed = df.set_index(['datetime', 'STN'])
    timestamps = df_indexed.index.get_level_values(0).unique()


    for ts in timestamps:
        slice_ts = df_indexed.loc[ts] # slice with STN as index
        for var in vars_to_fill:
            available_mask = slice_ts[var].notna()
            if available_mask.sum() == 0:
                continue
            missing_mask = slice_ts[var].isna()
            if not missing_mask.any():
                continue


            available_stns = slice_ts.index[available_mask].values
            missing_stns = slice_ts.index[missing_mask].values


            available_idx = [stn_to_idx[int(s)] for s in available_stns]
            missing_idx = [stn_to_idx[int(s)] for s in missing_stns]


            d = dist_mat[np.ix_(missing_idx, available_idx)]
            w = 1.0 / (d + eps)
            w = w / w.sum(axis=1, keepdims=True)
            available_values = slice_ts.loc[available_stns, var].astype(float).values
            filled_vals = (w * available_values[None, :]).sum(axis=1)


            for stn, val in zip(missing_stns, filled_vals):
                df_indexed.at[(ts, stn), var] = val


    return df_indexed.reset_index()

def process(weather_fp, weather_meta_fp, train_fp, out_fp,
            mode='interpolate', missing_thresh=0.2, max_neighbors=5, verbose=True):

    if verbose: print('Loading files...')
    df_w = pd.read_csv(weather_fp)
    df_wm = pd.read_csv(weather_meta_fp)
    df_train = pd.read_csv(train_fp)

    df_w.columns = df_w.columns.str.strip()
    df_wm.columns = df_wm.columns.str.strip()
    df_train.columns = df_train.columns.str.strip()

    df_w = parse_weather_datetime(df_w)

    for c in ['FH','T','RH']:
        if c in df_w.columns:
            df_w[c] = pd.to_numeric(df_w[c], errors='coerce')
        else:
            raise KeyError(f"Expected column '{c}' in weather file")

    missing_frac = compute_missing_fraction(df_w, ['FH','T','RH'])
    missing_frac['bad'] = (missing_frac['FH_missing_frac'] > missing_thresh) | \
                          (missing_frac['T_missing_frac'] > missing_thresh) | \
                          (missing_frac['RH_missing_frac'] > missing_thresh)

    bad_stns = missing_frac[missing_frac['bad']].index.astype(int).tolist()
    if verbose:
        print(f"Found {len(bad_stns)} weather stations with >{missing_thresh*100:.1f}% missing in FH/T/RH")

    if mode == 'remove':
        df_w = df_w[~df_w['STN'].isin(bad_stns)].copy()
        df_wm = df_wm[~df_wm['STN'].isin(bad_stns)].copy()
    elif mode == 'interpolate':
        if verbose: print('Interpolating temporally per-station...')
        dfs = []
        for stn, g in df_w.groupby('STN'):
            g2 = temporal_interpolate_station(g)
            dfs.append(g2)
        df_w = pd.concat(dfs, ignore_index=True)

        if verbose: print('Performing spatial fill for remaining missing values...')
        df_w = spatial_fill_remaining(df_w, df_wm, max_neighbors)
    else:
        raise ValueError("mode must be 'remove' or 'interpolate'")

    if verbose: print('Mapping train stations to nearest weather station...')
    df_wm['lon'] = pd.to_numeric(df_wm['lon'], errors='coerce')
    df_wm['lat'] = pd.to_numeric(df_wm['lat'], errors='coerce')
    df_train['geo_lng'] = pd.to_numeric(df_train['geo_lng'], errors='coerce')
    df_train['geo_lat'] = pd.to_numeric(df_train['geo_lat'], errors='coerce')
    df_wm = df_wm.dropna(subset=['lon','lat'])

    w_lons = df_wm['lon'].values
    w_lats = df_wm['lat'].values
    w_stns = df_wm['STN'].astype(int).values
    t_lons = df_train['geo_lng'].values
    t_lats = df_train['geo_lat'].values

    dist = haversine_np(t_lons[:, None], t_lats[:, None], w_lons[None, :], w_lats[None, :])
    nearest_idx = np.nanargmin(dist, axis=1)
    nearest_stn = w_stns[nearest_idx]

    df_train_map = df_train.copy()
    df_train_map['nearest_STN'] = nearest_stn
    df_train_map['nearest_dist_km'] = dist[np.arange(dist.shape[0]), nearest_idx]

    if verbose: print('Merging weather values onto train stations for each datetime...')
    cols_keep = ['STN','datetime','FH','T','RH']
    df_w_small = df_w[cols_keep].copy()
    df_final = df_train_map[['id','code','name_short','nearest_STN','nearest_dist_km']].merge(
        df_w_small, left_on='nearest_STN', right_on='STN', how='left')

    df_final = df_final.rename(columns={'id':'train_id','name_short':'train_name','STN':'weather_STN'})

    if verbose: print(f'Saving final file to {out_fp} ...')
    if out_fp.lower().endswith('.parquet'):
        df_final.to_parquet(out_fp, index=False)
    else:
        df_final.to_csv(out_fp, index=False)

    if verbose:
        print('Done.')
        print('Final dataset shape:', df_final.shape)

    return df_final

if __name__ == '__main__':
    # Uncomment this to create file yourself
    # process(weather_csv, weather_stations_csv, train_stations_csv, output_csv,
    #         mode=mode, missing_thresh=missing_thresh, max_neighbors=max_neighbors, verbose=verbose)
    chunk_size = 100000
    for chunk in pd.read_csv("data/processed/train_weather_merged.csv", chunksize=chunk_size):
        last_chunk = chunk
    print(last_chunk.tail(1))

