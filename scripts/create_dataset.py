import os
import argparse
import pandas as pd
from scipy.signal import butter, sosfilt

def find_files(folder):
    files = {}
    for filename in os.listdir(folder):
        upper = filename.upper()
        if "EVENT" in upper:
            files["events"] = os.path.join(folder, filename)
        elif "FLOW" in upper or "NASAL" in upper:
            files["flow"] = os.path.join(folder, filename)
        elif "SPO2" in upper or "SPO₂" in upper:
            files["spo2"] = os.path.join(folder, filename)
        elif "THORAC" in upper:
            files["thorac"] = os.path.join(folder, filename)
        elif "SLEEP" in upper or "PROFIL" in upper:
            files["sleep"] = os.path.join(folder, filename)
    return files

def load_signal(filepath):
    lines = []
    data_started = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "Data:":
                data_started = True
                continue
            if data_started and line:
                parts = line.split(';')
                timestamp = parts[0].strip().replace(',', '.')
                value = parts[1].strip()
                lines.append([timestamp, value])
    
    df = pd.DataFrame(lines, columns=['timestamp', 'value'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f')
    df['value'] = pd.to_numeric(df['value'])
    df.set_index('timestamp', inplace=True)
    return df

def load_events(filepath):
    events = []
    data_started = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if "Signal Type" in line:
                data_started = True
                continue
            if not data_started or not line:
                continue
            
            date_part, rest = line.split(' ', 1)
            time_range, duration, event_type, sleep_stage = rest.split(';')
            start_time, end_time = time_range.strip().split('-')
            
            start_str = (date_part + ' ' + start_time.strip()).replace(',', '.')
            end_str = (date_part + ' ' + end_time.strip()).replace(',', '.')
            
            start = pd.to_datetime(start_str, format='%d.%m.%Y %H:%M:%S.%f')
            end = pd.to_datetime(end_str, format='%d.%m.%Y %H:%M:%S.%f')
            events.append([start, end, event_type.strip(), sleep_stage.strip()])
    
    df = pd.DataFrame(events, columns=['start', 'end', 'event_type', 'sleep_stage'])
    return df

def bandpass_filter(signal_values, lowcut, highcut, fs):
    sos = butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
    filtered = sosfilt(sos, signal_values)
    return filtered

def create_windows(df, window_size, step_size):
    windows = []
    timestamps = []
    for i in range(0, len(df) - window_size, step_size):
        window = df['value'].iloc[i:i + window_size].values
        start_ts = df.index[i]
        end_ts = df.index[i + window_size - 1]
        windows.append(window)
        timestamps.append((start_ts, end_ts))
    return windows, timestamps

def label_windows(timestamps, events_df, window_duration=30):
    breathing_events = ['Hypopnea', 'Obstructive Apnea', 'Mixed Apnea', 'Central Apnea']
    labels = []
    for start_ts, end_ts in timestamps:
        window_label = 'Normal'
        for _, event in events_df.iterrows():
            if event['event_type'] not in breathing_events:
                continue
            overlap_start = max(start_ts, event['start'])
            overlap_end = min(end_ts, event['end'])
            overlap_seconds = (overlap_end - overlap_start).total_seconds()
            if overlap_seconds > 0.5 * window_duration:
                window_label = event['event_type']
                break
        labels.append(window_label)
    return labels

parser = argparse.ArgumentParser()
parser.add_argument("-in_dir", required=True)
parser.add_argument("-out_dir", required=True)
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

all_rows = []
for participant in os.listdir(in_dir):
    participant_path = os.path.join(in_dir, participant)
    if not os.path.isdir(participant_path):
        continue
    files = find_files(participant_path)

    flow_df = load_signal(files['flow'])
    thorac_df = load_signal(files['thorac'])
    spo2_df = load_signal(files['spo2'])
    events_df = load_events(files['events'])

    flow_df['value'] = bandpass_filter(flow_df['value'], 0.17, 0.4, 32)
    thorac_df['value'] = bandpass_filter(thorac_df['value'], 0.17, 0.4, 32)
    spo2_df['value'] = bandpass_filter(spo2_df['value'], 0.17, 0.4, 4)

    window_size_flow = 30 * 32   # 30 seconds at 32 Hz = 960 samples
    step_size_flow = 15 * 32     # 15 seconds at 32 Hz = 480 samples
    
    window_size_spo2 = 30 * 4    # 30 seconds at 4 Hz = 120 samples
    step_size_spo2 = 15 * 4      # 15 seconds at 4 Hz = 60 samples

    flow_windows, timestamps = create_windows(flow_df, window_size_flow, step_size_flow)
    thorac_windows, _ = create_windows(thorac_df, window_size_flow, step_size_flow)
    spo2_windows, _ = create_windows(spo2_df, window_size_spo2, step_size_spo2)

    labels = label_windows(timestamps, events_df)
    
    for i in range(len(flow_windows)):
        row = {
            'participant': participant,
            'start': timestamps[i][0],
            'end': timestamps[i][1],
            'label': labels[i],
            'flow': flow_windows[i].tolist(),
            'thorac': thorac_windows[i].tolist(),
            'spo2': spo2_windows[i].tolist()
        }
        all_rows.append(row)
    
    print(f"{participant} done - {pd.Series(labels).value_counts().to_dict()}")

df_out = pd.DataFrame(all_rows)
df_out.to_pickle(os.path.join(out_dir, 'breathing_dataset.pkl'))
print(f"Saved filtered dataset to {out_dir}/breathing_dataset.pkl")