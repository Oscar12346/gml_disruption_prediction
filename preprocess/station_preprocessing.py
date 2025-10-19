from pathlib import Path

import pandas as pd
from datetime import timedelta
import re

from scipy.spatial import cKDTree


def process_stations():
    # Load stations
    stations = pd.read_csv("data/train_stations/stations-2023-09.csv")
    stations = stations[stations["country"] == "NL"][["id", "code", "geo_lat", "geo_lng"]].rename(
        columns={"geo_lat": "lat", "geo_lng": "lon"}
    )
    out_segments_csv = "data/processed/segment_disruptions_collapsed.csv"
    out_path = Path(out_segments_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Load disruptions
    disruptions = pd.read_csv(
        "data/train_disruptions/disruptions.csv",
        parse_dates=["start_time", "end_time"]
    )
    valid_station_codes = set(stations["code"].astype(str))
    print(f"Loaded {len(stations)} Dutch stations")


    # --- Clean timestamps ---
    disruptions["start_time"] = pd.to_datetime(disruptions["start_time"], errors="coerce")
    disruptions["end_time"] = pd.to_datetime(disruptions["end_time"], errors="coerce")

    before = len(disruptions)
    disruptions = disruptions.dropna(subset=["start_time", "end_time"])
    disruptions = disruptions[disruptions["end_time"] >= disruptions["start_time"]]
    dropped_times = before - len(disruptions)
    print(f"Dropped {dropped_times} disruptions with missing/invalid times")

    # --- Filtering: remove 'staffing problems' and selected cause_en items ---
    disruptions = disruptions[disruptions["cause_group"].str.lower() != "staffing problems"]

    cause_en_remove = [
        "Engineering works",
        "Engineering works at a station",
        "Engineering works elsewhere",
        "Engineering works on the high-speed line",
        "Over-running engineering works",
        "Planned maintenance",
        "repair works"
    ]
    mask_exact = disruptions["cause_en"].isin(cause_en_remove)
    mask_construction = disruptions["cause_en"].str.match(r"^The construction", na=False)
    disruptions = disruptions[~(mask_exact | mask_construction)]
    print(f"Remaining disruptions after cause filtering: {len(disruptions)}")

    # --- Build segment records by splitting across consecutive station codes ---
    records = []
    skipped_no_codes = 0
    skipped_single_station = 0
    skipped_invalid_duration = 0
    kept_segments = 0

    for idx, row in disruptions.iterrows():
        codes = row.get("rdt_station_codes")

        # Skip if station codes missing or NaN
        if not isinstance(codes, str) or not codes.strip():
            skipped_no_codes += 1
            continue

        # split and clean; guard against weird separators
        station_codes = [s.strip() for s in str(codes).replace(";", ",").split(",") if s.strip()]
        if len(station_codes) < 2:
            skipped_single_station += 1
            continue

        # compute duration in minutes
        total_minutes = (row["end_time"] - row["start_time"]).total_seconds() / 60.0
        if pd.isna(total_minutes) or total_minutes <= 0:
            skipped_invalid_duration += 1
            continue

        n_segments = len(station_codes) - 1
        # allocate the total segment minutes per segment (equal split)
        segment_minutes = total_minutes / n_segments

        # assign each segment a time window proportional to its position along the route
        for i in range(n_segments):
            from_st = station_codes[i]
            to_st = station_codes[i + 1]

            # only keep segments where both endpoints are in NL station list
            if (from_st not in valid_station_codes) or (to_st not in valid_station_codes):
                continue

            seg_start = row["start_time"] + timedelta(minutes=i * segment_minutes)
            seg_end = seg_start + timedelta(minutes=segment_minutes)

            # create hourly rows — round seg_start down to the hour and include hours < seg_end
            t = seg_start.replace(minute=0, second=0, microsecond=0)
            while t < seg_end:
                records.append({
                    "from_station": from_st,
                    "to_station": to_st,
                    "datetime": t,
                    "disruption_id": row.get("rdt_id"),
                    "segment_minutes": float(segment_minutes),
                    "disrupted": 1
                })
                t += timedelta(hours=1)
            kept_segments += 1

    print(f"Skipped {skipped_no_codes} disruptions with missing station lists")
    print(f"Skipped {skipped_single_station} single-station disruptions")
    print(f"Skipped {skipped_invalid_duration} disruptions with invalid duration")
    print(f"Generated hourly rows for {kept_segments} segments (pre-collapse)")

    if not records:
        print("No segment disruption records generated. Exiting.")
        return

    # --- Create DataFrame and deduplicate ---
    df_segments = pd.DataFrame.from_records(records)
    # Remove exact duplicates if any
    df_segments = df_segments.drop_duplicates(subset=["from_station", "to_station", "datetime", "disruption_id"])

    # --- Collapse consecutive hourly records for the same disruption and segment into intervals ---
    df_segments = df_segments.sort_values(["from_station", "to_station", "disruption_id", "datetime"]).reset_index(
        drop=True)

    # Create grouping key for consecutive hours
    df_segments["prev_dt"] = df_segments.groupby(["from_station", "to_station", "disruption_id"])["datetime"].shift(1)
    df_segments["gap"] = (df_segments["datetime"] - df_segments["prev_dt"]) != pd.Timedelta(hours=1)
    # First row of each group or gaps start a new block -> cumulative sum of gap flags by group
    df_segments["block"] = df_segments.groupby(["from_station", "to_station", "disruption_id"])["gap"].cumsum().fillna(
        0).astype(int)

    # Aggregate blocks into start/end
    collapsed = (
        df_segments
        .groupby(["from_station", "to_station", "disruption_id", "block"], as_index=False)
        .agg(
            start_time=("datetime", "min"),
            end_time=("datetime", "max"),
            segment_minutes=("segment_minutes", "first"),
            disrupted=("disrupted", "first")
        )
    )

    # Keep only desired columns and sort
    collapsed = collapsed[
        ["from_station", "to_station", "start_time", "end_time", "disruption_id", "segment_minutes", "disrupted"]]
    collapsed = collapsed.sort_values(["start_time", "from_station", "to_station"]).reset_index(drop=True)

    # Save CSV
    collapsed.to_csv(out_segments_csv, index=False)
    print(f"Saved {len(collapsed)} collapsed segment disruptions to: {out_segments_csv}")

# def process_segments_with_weather(
#     stations_csv="data/train_stations/stations-2023-09.csv",
#     disruptions_csv="data/disruptions/disruptions.csv",
#     weather_csv="data/processed/weather_merged.csv",
#     weather_stations_csv="data/processed/weather_stations.csv",
#     out_csv="data/processed/segment_disruptions_with_weather.csv"
# ):
#     out_path = Path(out_csv)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#
#     # --- Load train stations ---
#     stations = pd.read_csv(stations_csv, dtype=str, low_memory=False)
#     stations = stations[stations["country"] == "NL"]
#     stations = stations[["id", "code", "geo_lat", "geo_lng"]].rename(
#         columns={"geo_lat": "lat", "geo_lng": "lon"}
#     )
#     stations["lat"] = stations["lat"].astype(float)
#     stations["lon"] = stations["lon"].astype(float)
#     valid_station_codes = set(stations["code"])
#
#     # --- Load disruptions ---
#     disruptions = pd.read_csv(disruptions_csv, parse_dates=["start_time", "end_time"], low_memory=False)
#     disruptions["start_time"] = pd.to_datetime(disruptions["start_time"], errors="coerce")
#     disruptions["end_time"]   = pd.to_datetime(disruptions["end_time"], errors="coerce")
#     disruptions = disruptions.dropna(subset=["start_time", "end_time"])
#     disruptions = disruptions[disruptions["end_time"] >= disruptions["start_time"]]
#
#     # --- Filter causes ---
#     disruptions = disruptions[disruptions["cause_group"].str.lower() != "staffing problems"]
#
#     cause_en_remove = [
#         "Engineering works",
#         "Engineering works at a station",
#         "Engineering works elsewhere",
#         "Engineering works on the high-speed line",
#         "Over-running engineering works",
#         "Planned maintenance",
#         "repair works"
#     ]
#     mask_exact = disruptions["cause_en"].isin(cause_en_remove)
#     mask_construction = disruptions["cause_en"].str.match(r"^The construction", na=False)
#     disruptions = disruptions[~(mask_exact | mask_construction)]
#
#     # --- Split into segments ---
#     records = []
#     for idx, row in disruptions.iterrows():
#         codes = row.get("rdt_station_codes")
#         if not isinstance(codes, str) or not codes.strip():
#             continue
#         station_codes = [s.strip() for s in str(codes).replace(";", ",").split(",") if s.strip()]
#         if len(station_codes) < 2:
#             continue
#         total_minutes = (row["end_time"] - row["start_time"]).total_seconds() / 60.0
#         if pd.isna(total_minutes) or total_minutes <= 0:
#             continue
#         n_segments = len(station_codes) - 1
#         segment_minutes = total_minutes / n_segments
#         for i in range(n_segments):
#             from_st = station_codes[i]
#             to_st   = station_codes[i + 1]
#             if (from_st not in valid_station_codes) or (to_st not in valid_station_codes):
#                 continue
#             seg_start = row["start_time"] + timedelta(minutes=i * segment_minutes)
#             seg_end   = seg_start + timedelta(minutes=segment_minutes)
#             t = seg_start.replace(minute=0, second=0, microsecond=0)
#             while t < seg_end:
#                 records.append({
#                     "from_station": from_st,
#                     "to_station": to_st,
#                     "datetime": t,
#                     "disruption_id": row.get("rdt_id"),
#                     "segment_minutes": float(segment_minutes),
#                     "disrupted": 1
#                 })
#                 t += timedelta(hours=1)
#
#     if not records:
#         print("No segment disruptions generated. Exiting.")
#         return
#
#     # --- Create DataFrame and collapse consecutive hours ---
#     df_segments = pd.DataFrame.from_records(records)
#     df_segments = df_segments.drop_duplicates(subset=["from_station","to_station","datetime","disruption_id"])
#     df_segments = df_segments.sort_values(["from_station", "to_station", "disruption_id", "datetime"]).reset_index(drop=True)
#
#     df_segments["prev_dt"] = df_segments.groupby(["from_station", "to_station", "disruption_id"])["datetime"].shift(1)
#     df_segments["gap"] = (df_segments["datetime"] - df_segments["prev_dt"]) != pd.Timedelta(hours=1)
#     df_segments["block"] = df_segments.groupby(["from_station", "to_station", "disruption_id"])["gap"].cumsum().fillna(0).astype(int)
#
#     collapsed = (
#         df_segments
#         .groupby(["from_station", "to_station", "disruption_id", "block"], as_index=False)
#         .agg(
#             start_time=("datetime", "min"),
#             end_time=("datetime", "max"),
#             segment_minutes=("segment_minutes", "first"),
#             disrupted=("disrupted", "first")
#         )
#     )
#     # Duration in minutes
#     collapsed["duration_minutes"] = (collapsed["end_time"] - collapsed["start_time"]).dt.total_seconds() / 60 + 60
#
#     collapsed = collapsed[
#         ["from_station", "to_station", "start_time", "end_time", "disruption_id", "segment_minutes", "duration_minutes", "disrupted"]
#     ]
#
#     # --- Load weather stations and build KDTree ---
#     weather_stations = pd.read_csv(weather_stations_csv)
#     weather_stations["lat"] = weather_stations["lat"].astype(float)
#     weather_stations["lon"] = weather_stations["lon"].astype(float)
#     weather = pd.read_csv(weather_csv, parse_dates=["datetime"])
#     weather["STN"] = weather["STN"].astype(int)
#
#     tree = cKDTree(weather_stations[["lat", "lon"]].to_numpy())
#     station_coords = stations[["lat","lon"]].to_numpy()
#     distances, indices = tree.query(station_coords)
#     stations["nearest_STN"] = weather_stations.iloc[indices]["STN"].values
#
#     # --- Merge weather data to from_station ---
#     segments = collapsed.merge(
#         stations[["code", "nearest_STN"]],
#         left_on="from_station", right_on="code", how="left"
#     ).rename(columns={"nearest_STN": "from_STN"}).drop(columns=["code"])
#
#     segments = segments.merge(
#         weather, left_on=["from_STN", "start_time"], right_on=["STN", "datetime"], how="left"
#     )
#
#     # --- Save final CSV ---
#     segments.to_csv(out_csv, index=False)
#     print(f"Saved {len(segments)} segments with weather to: {out_csv}")


if __name__ == "__main__":
    process_stations()


