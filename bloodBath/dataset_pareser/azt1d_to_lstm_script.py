from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import re

DATASET_PATH = Path("/home/desktop/Sneior_Project/datasets/azt1d/")
OUTPUT_DIR = Path("/home/desktop/Sneior_Project/bloodBender/bloodBath/bloodBank/raw/datasets")
TZ = "America/Los_Angeles"


def localize(series: pd.Series) -> pd.Series:
    return series.dt.tz_localize(
        TZ,
        ambiguous="NaT",
        nonexistent="shift_forward"
    )


def extract_subject_id(csv_file: Path) -> str:
    # tries folder name first: "Subject 3" -> "3"
    folder_name = csv_file.parent.name
    match = re.search(r"Subject\s+(\d+)", folder_name, re.IGNORECASE)
    if match:
        return match.group(1)

    # fallback to file stem: "Subject 3.csv" -> "3"
    match = re.search(r"Subject\s+(\d+)", csv_file.stem, re.IGNORECASE)
    if match:
        return match.group(1)

    return csv_file.stem


def parse_csv_to_df(csv_file: Path):
    df = pd.read_csv(csv_file)

    # --- normalize expected column names ---
    # adjust these if your AZ1TD CSV uses slightly different names
    timestamp_col = "EventDateTime"
    bg_col = "CGM"
    basal_col = "Basal"
    bolus_col = "TotalBolusInsulinDelivered"

    missing = [c for c in [timestamp_col, bg_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_file}: {missing}")

    # ----------------------------
    # 1) Timestamp + BG
    # ----------------------------
    out = pd.DataFrame()

    out["timestamp_raw"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    out = out.dropna(subset=["timestamp_raw"]).copy()
    out["timestamp_raw"] = localize(out["timestamp_raw"])

    out = out.dropna(subset=["timestamp_raw"]).copy()
    out["timestamp"] = out["timestamp_raw"].dt.floor("5min")

    out["bg_raw"] = pd.to_numeric(df.loc[out.index, bg_col], errors="coerce")

    # average if multiple rows land in same 5-minute bucket
    out = (
        out.groupby("timestamp", as_index=False)
        .agg(bg=("bg_raw", "mean"))
    )

    out["bg_observed"] = out["bg"].notna()

    # ----------------------------
    # 2) Full 5-minute grid
    # ----------------------------
    full_grid = pd.DataFrame({
        "timestamp": pd.date_range(
            start=out["timestamp"].min(),
            end=out["timestamp"].max(),
            freq="5min",
            tz=TZ
        )
    })

    out = full_grid.merge(out, on="timestamp", how="left")
    out["bg_observed"] = out["bg_observed"].fillna(False)

    # mask_bg = True when BG should be ignored
    out["mask_bg"] = ~out["bg_observed"]

    # fill BG by interpolation
    out["bg"] = out["bg"].interpolate(method="linear", limit_direction="both")

    # ----------------------------
    # 3) Basal
    # ----------------------------
    if basal_col in df.columns:
        basal_df = pd.DataFrame({
            "timestamp_raw": pd.to_datetime(df[timestamp_col], errors="coerce"),
            "basal_rate": pd.to_numeric(df[basal_col], errors="coerce")
        })

        basal_df = basal_df.dropna(subset=["timestamp_raw"]).copy()
        basal_df["timestamp_raw"] = localize(basal_df["timestamp_raw"])
        basal_df = basal_df.dropna(subset=["timestamp_raw"]).copy()
        basal_df["timestamp"] = basal_df["timestamp_raw"].dt.floor("5min")

        # keep latest non-null basal in each bucket, then carry forward
        basal_df = (
            basal_df.drop(columns=["timestamp_raw"])
            .groupby("timestamp", as_index=False)
            .agg(basal_rate=("basal_rate", "last"))
            .sort_values("timestamp")
        )

        out = out.merge(basal_df, on="timestamp", how="left")
        out["basal_rate"] = out["basal_rate"].ffill().fillna(0.0)
    else:
        out["basal_rate"] = 0.0

    # ----------------------------
    # 4) Bolus
    # ----------------------------
    out["bolus_dose"] = 0.0

    if bolus_col in df.columns:
        bolus_df = pd.DataFrame({
            "timestamp_raw": pd.to_datetime(df[timestamp_col], errors="coerce"),
            "bolus_dose": pd.to_numeric(df[bolus_col], errors="coerce").fillna(0.0)
        })

        bolus_df = bolus_df.dropna(subset=["timestamp_raw"]).copy()
        bolus_df["timestamp_raw"] = localize(bolus_df["timestamp_raw"])
        bolus_df = bolus_df.dropna(subset=["timestamp_raw"]).copy()
        bolus_df["timestamp"] = bolus_df["timestamp_raw"].dt.floor("5min")

        bolus_df = (
            bolus_df.drop(columns=["timestamp_raw"])
            .groupby("timestamp", as_index=False)
            .agg(bolus_dose=("bolus_dose", "sum"))
        )

        out = out.merge(bolus_df, on="timestamp", how="left", suffixes=("", "_new"))
        out["bolus_dose"] = out["bolus_dose_new"].fillna(out["bolus_dose"])
        out = out.drop(columns=["bolus_dose_new"])

    # ----------------------------
    # 5) Derived features
    # ----------------------------
    out["basal_delta"] = out["basal_rate"].diff().fillna(0.0)

    last_bolus_time = pd.NaT
    tslb = []
    for ts, dose in zip(out["timestamp"], out["bolus_dose"]):
        if dose > 0:
            last_bolus_time = ts

        if pd.isna(last_bolus_time):
            tslb.append(1440.0)
        else:
            tslb.append((ts - last_bolus_time).total_seconds() / 60.0)

    out["time_since_last_bolus"] = tslb

    out["bg_slope_15min"] = (out["bg"] - out["bg"].shift(3)) / 15.0
    out["bg_slope_30min"] = (out["bg"] - out["bg"].shift(6)) / 30.0
    out["bg_slope_15min"] = out["bg_slope_15min"].fillna(0.0)
    out["bg_slope_30min"] = out["bg_slope_30min"].fillna(0.0)

    minutes_of_day = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    angle = 2 * np.pi * minutes_of_day / (24 * 60)
    out["sin_time"] = np.sin(angle)
    out["cos_time"] = np.cos(angle)

    # same convention as your Ohio code
    out["mask_label"] = ~out["mask_bg"]

    subject_id = extract_subject_id(csv_file)

    out = out[
        [
            "timestamp",
            "bg",
            "basal_rate",
            "bolus_dose",
            "basal_delta",
            "time_since_last_bolus",
            "bg_slope_15min",
            "bg_slope_30min",
            "sin_time",
            "cos_time",
            "mask_bg",
            "mask_label",
        ]
    ]

    # force masks to bool
    out["mask_bg"] = out["mask_bg"].astype(bool)
    out["mask_label"] = out["mask_label"].astype(bool)

    return out, subject_id


def traverse_dataset(dataset_dir: Path, output_dir: Path) -> list[str]:
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {dataset_dir}")

    results = []

    # recursively find AZ1TD subject CSVs
    for csv_file in dataset_path.rglob("*.csv"):
        try:
            df, subject_id = parse_csv_to_df(csv_file)

            out_path = output_dir / "az1td" / f"{subject_id}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(out_path, index=False)
            print(f"Parsed {csv_file} for subject {subject_id}, saved to {out_path}")
            results.append(subject_id)

        except Exception as e:
            print(f"Failed to parse {csv_file}: {e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse AZ1TD CSV dataset and output processed dataframe"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to input AZ1TD dataset directory"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Path to output directory"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_directory = args.dataset
    output_dir = args.output

    all_data = traverse_dataset(dataset_directory, output_dir)

    for item in all_data:
        print(f"Processed subject: {item}")