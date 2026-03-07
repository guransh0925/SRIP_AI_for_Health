import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

parser = argparse.ArgumentParser()
parser.add_argument("-name", required=True)
args = parser.parse_args()

folder = args.name
files = find_files(folder)
# print(files)

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

# flow_df = load_signal(files['flow'])
# print(flow_df.head())

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

# events_df = load_events(files['events'])
# print(events_df.head())

def plot_participant(files, participant_name):
    flow_df = load_signal(files['flow'])
    thorac_df = load_signal(files['thorac'])
    spo2_df = load_signal(files['spo2'])
    events_df = load_events(files['events'])

    start_time = flow_df.index[0]
    end_time = flow_df.index[-1]
    
    window = pd.Timedelta(minutes=5)
    current = start_time

    os.makedirs('Visualizations', exist_ok=True)
    pdf_path = f'Visualizations/{participant_name}_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        while current < end_time:
            window_end = current + window
            
            f_slice = flow_df[current:window_end]
            t_slice = thorac_df[current:window_end]
            s_slice = spo2_df[current:window_end]
            e_slice = events_df[(events_df['end'] >= current) & (events_df['start'] <= window_end)]
            
            fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
            
            axes[0].plot(f_slice.index, f_slice['value'], linewidth=1, color='blue')
            axes[0].set_ylabel('Nasal Flow (L/min)')
            axes[0].set_ylim(f_slice['value'].min() - 10, f_slice['value'].max() + 10)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(['Nasal Flow'], loc='upper right')
            
            axes[1].plot(t_slice.index, t_slice['value'], linewidth=1, color='orange')
            axes[1].set_ylabel('Resp. Amplitude')
            axes[1].set_ylim(t_slice['value'].min() - 10, t_slice['value'].max() + 10)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(['Thoracic/Abdominal Resp.'], loc='upper right')
            
            axes[2].plot(s_slice.index, s_slice['value'], linewidth=1, color='gray')
            axes[2].set_ylabel('SpO2 (%)')
            axes[2].set_ylim(max(0, s_slice['value'].min() - 1), min(100, s_slice['value'].max()) + 1)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(['SpO2'], loc='upper right')
            
            event_colors = {
                'Hypopnea': 'red',
                'Obstructive Apnea': 'yellow',
                'Mixed Apnea': 'purple',
                'Central Apnea': 'blue'
            }
            legend_added = set()
            for _, event in e_slice.iterrows():
                color = event_colors.get(event['event_type'], None)
                if color is None:
                    continue
                axes[0].axvspan(event['start'], event['end'], alpha=0.3, color=color)
                
                mid_time = event['start'] + (event['end'] - event['start']) / 2
                axes[0].text(
                    mid_time,                    
                    f_slice['value'].max() + 5,
                    event['event_type'],
                    ha='center',
                    va='top',
                    fontsize=8,
                    color='black',
                )
                legend_added.add(event['event_type'])
            
            axes[0].legend(['Nasal Flow'], loc='upper right')
            axes[1].legend(['Thoracic/Abdominal Resp.'], loc='upper right')
            axes[2].legend(['SpO2'], loc='upper right')
            for ax in axes:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d %H:%M:%S'))
                padding = pd.Timedelta(seconds=10)
                ax.set_xlim(f_slice.index[0] - padding, f_slice.index[-1] + padding)
                ax.set_xticks(pd.date_range(f_slice.index[0], f_slice.index[-1], freq='5s'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center')
            
            plt.suptitle(f'{participant_name} - {current.strftime("%Y-%m-%d %H:%M")} to {window_end.strftime("%Y-%m-%d %H:%M")}')
            plt.xlabel('Time')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            current = window_end
    
    print(f"Saved to {pdf_path}")

participant_name = os.path.basename(folder)
plot_participant(files, participant_name)