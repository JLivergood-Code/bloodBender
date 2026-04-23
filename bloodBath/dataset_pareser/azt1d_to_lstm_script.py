#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def add_time_features(df):
    seconds = (
        df["timestamp"].dt.hour * 3600
        + df["timestamp"].dt.minute * 60
        + df["timestamp"].dt.second
    )
    angle = 2 * np.pi * seconds / 86400.0
    df["sin_time"] = np.sin(angle)
    df["cos_time"] = np.cos(angle)
    return df

def compute_time_since_last_bolus(df):
    last_bolus_time = None
    out = []

    for _, row in df.iterrows():
        ts = row["timestamp"]
        bolus = row["bolus_dose"]

        if last_bolus_time is None:
            out.append(1440.0)
        else:
            out.append((ts - last_bolus_time).total_seconds() / 60.0)

        if pd.notna(bolus) and bolus > 0:
            last_bolus_time = ts
            out[-1] = 0.0

    df["time_since_last_bolus"] = out
    return df

def compute_bg_slopes(df):
    bg = df["bg"]

    df["bg_slope_15min"] = ((bg - bg.shift(3)) / 15.0).fillna(0.0)
    df["bg_slope_30min"] = ((bg - bg.shift(6)) / 30.0).fillna(0.0)

    invalid15 = bg.isna() | bg.shift(3).isna()
    invalid30 = bg.isna() | bg.shift(6).isna()

    df.loc[invalid15, "bg_slope_15min"] = 0.0
    df.loc[invalid30, "bg_slope_30min"] = 0.0
    return df

def transform_subject(raw_df, timestamp_col, bg_col, basal_col, bolus_col):
    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df[timestamp_col])
    raw_df = raw_df.sort_values("timestamp")

    raw_df = raw_df.rename(columns={
        bg_col: "bg",
        basal_col: "basal_rate",
        bolus_col: "bolus_dose",
    })

    start_ts = raw_df["timestamp"].min().floor("5min")
    end_ts = raw_df["timestamp"].max().ceil("5min")

    full_index = pd.date_range(start=start_ts, end=end_ts, freq="5min")

    df = (
        raw_df.set_index("timestamp")
        .reindex(full_index)
        .rename_axis("timestamp")
        .reset_index()
    )

    df["bolus_dose"] = df.get("bolus_dose", 0.0).fillna(0.0)
    df["basal_rate"] = df.get("basal_rate", 0.0).ffill().fillna(0.0)
    df["bg"] = df.get("bg", np.nan)

    df["basal_delta"] = df["basal_rate"].diff().fillna(0.0)

    df = compute_time_since_last_bolus(df)
    df = compute_bg_slopes(df)
    df = add_time_features(df)

    df["mask_bg"] = df["bg"].isna()
    df["mask_label"] = df["bg"].notna()

    return df[
        [
            "timestamp", "bg", "basal_rate", "bolus_dose", "basal_delta",
            "time_since_last_bolus", "bg_slope_15min", "bg_slope_30min",
            "sin_time", "cos_time", "mask_bg", "mask_label"
        ]
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--bg-col", default="glucose")
    parser.add_argument("--basal-col", default="basal")
    parser.add_argument("--bolus-col", default="bolus")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    out_df = transform_subject(
        raw_df,
        args.timestamp_col,
        args.bg_col,
        args.basal_col,
        args.bolus_col
    )

    out_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
