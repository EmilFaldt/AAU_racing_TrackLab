import pandas as pd
import glob
import os
import numpy as np

def load_ecu_data(folder_path):
    """Load ECU CSVs with exactly 2 columns into a dict of DataFrames."""
    ecu_data = {}
    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        df = pd.read_csv(file, header=0)
        if df.shape[1] != 2:
            continue  # skip non-ECU data
        channel_name = os.path.splitext(os.path.basename(file))[0]
        df.columns = ["time", channel_name]
        df = df.sort_values("time").reset_index(drop=True)

        ecu_data[channel_name] = df
    return ecu_data


def load_logger_data(folder_path):
    """Load GPS/logger CSVs with more than 2 columns into a dict of DataFrames."""
    logger_data = {}
    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        df = pd.read_csv(file, header=0)
        if df.shape[1] <= 2:
            continue  # skip ECU-type data
        file_key = os.path.splitext(os.path.basename(file))[0]
        df = df.sort_values(df.columns[0]).reset_index(drop=True)

        # Assume first column is time
        df.rename(columns={df.columns[0]: "time"}, inplace=True)

        # Convert ms to seconds if needed
        if df["time"].max() > 1000:
            df["time"] = df["time"] / 1000.0

        logger_data[file_key] = df
    return logger_data


def merge_telemetry(ecu_data, logger_data, resample_rate=None):
    """
    Merge ECU and logger data on a common time base.
    If resample_rate is None, use the highest detected rate.
    """
    all_times = []
    detected_rates = []

    # Collect time ranges & rates from ECU data
    for df in ecu_data.values():
        all_times.extend(df["time"].values)
        if len(df) > 1:
            median_dt = np.median(np.diff(df["time"].values))
            if median_dt > 0:
                detected_rates.append(1.0 / median_dt)

    # Collect from logger data
    for df in logger_data.values():
        all_times.extend(df["time"].values)
        if len(df) > 1:
            median_dt = np.median(np.diff(df["time"].values))
            if median_dt > 0:
                detected_rates.append(1.0 / median_dt)

    min_time, max_time = min(all_times), max(all_times)
    if resample_rate is None:
        resample_rate = max(detected_rates) if detected_rates else 1.0

    time_step = 1.0 / resample_rate
    common_time = np.arange(min_time, max_time + time_step / 2, time_step)
    combined_df = pd.DataFrame({"time": common_time})

    # Resample ECU channels
    for channel, df in ecu_data.items():
        series = pd.Series(df[channel].values, index=df["time"].values)
        combined_df[channel] = np.interp(common_time, series.index, series.values)

    # Resample Logger/GPS columns
    for file_key, df in logger_data.items():
        for col in df.columns:
            if col == "time":
                continue
            series = pd.Series(df[col].values, index=df["time"].values)
            combined_df[col] = np.interp(common_time, series.index, series.values)

    return combined_df



folder = "C:\\Users\\Emil\\Desktop\\TrackLab\\AAU_racing_TrackLab\\data\\AiM\\20230713_102646_G9_20_AUTO_TRACK_Generic testing_a_1741_CSV"

ecu_data = load_ecu_data(folder)
logger_data = load_logger_data(folder)

print(logger_data)