import io

import pandas as pd
from glob import glob
from pathlib import Path
from pandas.errors import EmptyDataError

def main():
    input_dir = Path("data/weather")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob(str(input_dir / "*.txt")))
    if not files:
        print("No weather files found.")
        return

    dfs = []

    for f in files:
        print(f"Processing {f} ...")
        with open(f, "r") as file:
            lines = file.readlines()

        # Find the last comment line that contains the CSV header
        header_line = None
        for line in reversed(lines):
            if line.startswith("# STN,"):
                header_line = line[2:].strip()  # remove "# "
                break

        if header_line is None:
            print(f"Skipping {f} — no header found")
            continue

        columns = [c.strip() for c in header_line.split(",")]

        # Read all lines after the header as CSV data
        data_start = lines.index("# " + header_line + "\n") + 1
        data_lines = lines[data_start:]

        if not data_lines:
            print(f"Skipping {f} — no data after header")
            continue

        try:
            df = pd.read_csv(
                io.StringIO("".join(data_lines)),
                names=columns,
                skip_blank_lines=True,
                dtype=str
            )
            dfs.append(df)
        except EmptyDataError:
            print(f"Skipping {f} — empty data")
            continue

    if not dfs:
        print("No valid data found.")
        return

    # Merge all files
    weather = pd.concat(dfs, ignore_index=True)

    # Convert numeric columns safely
    numeric_cols = ["STN", "FH", "T", "RH", "M", "R", "S", "O", "Y"]
    for col in numeric_cols:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    # Combine date + hour into datetime
    weather["datetime"] = pd.to_datetime(
        weather["YYYYMMDD"].astype(str).str.zfill(8)
        + weather["HH"].astype(str).str.zfill(2),
        format="%Y%m%d%H",
        errors="coerce"
    )

    # Unit conversions
    weather["FH"] = weather["FH"] / 10
    weather["T"] = weather["T"] / 10
    weather["RH"] = weather["RH"] / 10

    # Read station metadata from the top comment lines
    station_meta = []
    for line in lines:
        if line.startswith("# ") and line[2:5].strip().isdigit():
            parts = line[2:].split()
            station_meta.append({
                "STN": int(parts[0]),
                "lon": float(parts[1]),
                "lat": float(parts[2]),
                "alt": float(parts[3]),
                "name": " ".join(parts[4:])
            })
    station_meta = pd.DataFrame(station_meta)

    # Save outputs
    output_dir.mkdir(exist_ok=True, parents=True)
    weather.to_csv(output_dir / "weather_merged.csv", index=False)
    station_meta.to_csv(output_dir / "weather_stations.csv", index=False)

    print(f"Merged {len(dfs)} files successfully.")
    print("Saved weather and station CSVs.")

if __name__ == "__main__":
    main()
