#%% import packages
import mne
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import hvplot.pandas
import holoviews as hv
from scipy.signal import find_peaks, iirnotch, filtfilt, butter


# %% Bestand inladen
# --- 1. Bestandspad ---
file_path = r"C:\Users\larts3\OneDrive - UMC Utrecht\Documenten\PSG\PSG_BB.edf"

# --- 2. EDF+ inladen ---
raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel=None)
print(raw.info)

# --- 3. Signaalnamen bekijken ---
print("Signaalnamen:", raw.ch_names)


#%% 
# --- 4. Data ophalen (numpy array) ---
data, times = raw.get_data(return_times=True)  # shape = (n_channels, n_times)
print("Data shape:", data.shape)


#%%
# --- 5. Eerste 5 seconden van elk kanaal plotten ---
sfreq = raw.info['sfreq']  # sample rate
seconds_to_plot = 5
samples_to_plot = int(seconds_to_plot * sfreq)

plt.figure(figsize=(12, 6))
for i, ch_name in enumerate(raw.ch_names):
    plt.plot(data[i, :samples_to_plot] + i*200, label=ch_name)  # offset per kanaal
plt.xlabel("Samples")
plt.ylabel("Amplitude (offset per kanaal)")
plt.title("Eerste 5 seconden van elk kanaal")
plt.legend(loc="upper right")
plt.show()


#%%
# DataFrame maken (tijd als index, kanaalnamen als kolommen)
df = pd.DataFrame(data.T, columns=raw.ch_names, index=times)

# Eerste paar rijen bekijken
print(df.head())


#%% RIP signalen hernoemen en RIP som maken
RIP_thor = df['Resp Thor']
RIP_abd = df['Resp Abd']
RIP_sum = RIP_thor + RIP_abd
RIP_sum


#%% Plotten
plt.figure(figsize=(12, 6))
plt.plot(RIP_thor, label='RIP Thorax')
plt.plot(RIP_abd, label='RIP Abdomen')
plt.plot(RIP_sum, label='RIP Som', linestyle='--')

plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("RIP signalen")
plt.legend()
plt.show()


#%% Kleiner deel inladen te selecteren via tmin en tmax
# --- 1. Bestand inladen (preload=False om snel te starten) ---
file_path = r"C:\Users\larts3\OneDrive - UMC Utrecht\Documenten\PSG\PSG_BB.edf"
raw = mne.io.read_raw_edf(file_path, preload=False)

# --- 2. Alleen de gewenste kanalen selecteren ---
channels = ['Resp Thor', 'Resp Abd']
raw = raw.copy().pick_channels(channels)

# --- 3. Alleen de eerste 2 minuten selecteren ---
raw_small = raw.copy().crop(tmin=18000, tmax=19200).load_data()  # preload=True na crop

# --- 4. Omzetten naar DataFrame ---
data, times = raw_small.get_data(return_times=True)
df_small = pd.DataFrame(data.T, columns=raw_small.ch_names, index=times)

# --- 5. Bekijk de eerste rijen van de DataFrame ---
print(df_small.head())

# --- 6. Plotten van de twee kanalen ---
plt.figure(figsize=(12, 5))
plt.plot(df_small.index, df_small['Resp Thor'], label='RIP Thorax')
plt.plot(df_small.index, df_small['Resp Abd'], label='RIP Abdomen')
plt.xlabel("Tijd (sec)")
plt.ylabel("Amplitude")
plt.title("RIP signalen")
plt.legend()
plt.show()


#%%
df_small['RIP_thor'] = df_small['Resp Thor']
df_small['RIP_abd'] = df_small['Resp Abd']
df_small['RIP_sum'] = df_small['RIP_thor'] + df_small['RIP_abd']
# hvplot zorgt voor een plot waarin je kan inzoomen
rip_plot = df_small.hvplot.line(
    y=['RIP_thor','RIP_abd','RIP_sum'], 
    width=1000, 
    height=400,
    title = 'RIP signals with annotations')

annotations = raw.annotations
# Alleen annotaties binnen het kleine segment (df_small)
annotations_inrange = [ann for ann in zip(annotations.onset, annotations.description) 
                       if ann[0] <= df_small.index[-1]]

# Maak verticale lijnen voor elke annotatie
vlines = [hv.VLine(onset).opts(color='red', line_width=1, line_dash='dashed') 
          for onset, desc in annotations_inrange]

# --- 4. Tekstlabels iets onder de bovenkant en verspreid --- 
y_max = df_small['RIP_sum'].max()
texts = []
for i, (onset, desc) in enumerate(annotations_inrange):
    # Verspreid labels op y-as om overlap te vermijden
    y_pos = y_max * (0.95 - (i % 5) * 0.05)  # 5 niveaus, herhaal als er meer annotaties zijn
    texts.append(hv.Text(onset, y_pos, desc).opts(
        text_color='red', fontsize=8, text_align='left', text_baseline='bottom'
    ))

# --- 5. Combineer alles ---
rip_plot = rip_plot * hv.Overlay(vlines + texts)

# --- 6. Plot tonen ---
rip_plot


#%% Een kleiner deel inladen (zelfde als hiervoor, andere tijdsmomenten)
# --- 1. Bestand inladen (preload=False om snel te starten) ---
file_path = r"C:\Users\larts3\OneDrive - UMC Utrecht\Documenten\PSG\PSG_BB.edf"
raw = mne.io.read_raw_edf(file_path, preload=False)

# --- 2. Alleen de gewenste kanalen selecteren ---
channels = ['Resp Thor', 'Resp Abd']
raw = raw.copy().pick_channels(channels)

# --- 3. Alleen de eerste 2 minuten selecteren ---
raw_small2 = raw.copy().crop(tmin=18480, tmax=18540).load_data()  # preload=True na crop

# --- 4. Omzetten naar DataFrame ---
data, times = raw_small2.get_data(return_times=True)
df_small2 = pd.DataFrame(data.T, columns=raw_small2.ch_names, index=times)

df_small2['RIP_thor'] = df_small2['Resp Thor']
df_small2['RIP_abd'] = df_small2['Resp Abd']
df_small2['RIP_sum'] = df_small2['RIP_thor'] + df_small2['RIP_abd']
# hvplot zorgt voor een plot waarin je kan inzoomen
rip_plot = df_small2.hvplot.line(
    y=['RIP_thor','RIP_abd','RIP_sum'], 
    width=1000, 
    height=400,
    title = 'RIP signals with annotations')

annotations = raw.annotations
# Alleen annotaties binnen het kleine segment (df_small2)
annotations_inrange = [ann for ann in zip(annotations.onset, annotations.description) 
                       if ann[0] <= df_small2.index[-1]]

# Maak verticale lijnen voor elke annotatie
vlines = [hv.VLine(onset).opts(color='red', line_width=1, line_dash='dashed') 
          for onset, desc in annotations_inrange]

# --- 4. Tekstlabels iets onder de bovenkant en verspreid --- 
y_max = df_small2['RIP_sum'].max()
texts = []
for i, (onset, desc) in enumerate(annotations_inrange):
    # Verspreid labels op y-as om overlap te vermijden
    y_pos = y_max * (0.95 - (i % 5) * 0.05)  # 5 niveaus, herhaal als er meer annotaties zijn
    texts.append(hv.Text(onset, y_pos, desc).opts(
        text_color='red', fontsize=8, text_align='left', text_baseline='bottom'
    ))

# --- 5. Combineer alles ---
rip_plot = rip_plot * hv.Overlay(vlines + texts)

# --- 6. Plot tonen ---
rip_plot


#%% Tabel van annotaties maken
annot_df = pd.DataFrame({
    'onset': annotations.onset,
    'duration': annotations.duration,
    'description': annotations.description
})

print(annot_df)


#%% Filtering met een 50Hz bandstop filter tegen het effect van het net en een low-pass filter waardoor alleen de
# frequenties worden doorgelaten die potentieel ademhaling zijn.
fs = 500  # samplingfrequentie

def preprocess(signal, fs=500):
    # 1. Notch filter op 50 Hz
    f0, Q = 50.0, 30.0
    b_notch, a_notch = iirnotch(f0, Q, fs)
    signal = filtfilt(b_notch, a_notch, signal)

    # 2. Low-pass filter op 1.33 Hz
    #cutoff op 1,33 uitgaande van een (ruime, maar je wil in ieder geval geen ademhaling wegfilteren) 
    # mogelijke ademhalingsfrequentie van 80/min. Order op 3. Hoe hoger de order, hoe steiler het filter (maar dat 
    # is nu overbodig, omdat we geen strakke grens hebben), maar ook hoe meer faseverschuiving. Daarom kies je vaak 3 of 4.
    cutoff, order = 1.33, 3 
    b_lp, a_lp = butter(order, cutoff/(fs/2), btype='low')
    signal = filtfilt(b_lp, a_lp, signal)
    
    return signal

df_small2['RIP_thor_filt'] = preprocess(df_small2['RIP_thor'].values, fs)
df_small2['RIP_abd_filt'] = preprocess(df_small2['RIP_abd'].values, fs)


#%% Pieken en dalen zoeken
thor = df_small2['RIP_thor_filt'].values
abd  = df_small2['RIP_abd_filt'].values

# Vind pieken (inspiratie)
# Distance is aantal samples (x-as) tussen opeenvolgende pieken dan wel dalen. Op basis van max 80 ademhalingen/min = 1,33 Hz volgt een min. distance van samplefrequency/ademhaling = 500/1.33 = 375. 
# Als we dan iets conservatiever kiezen, zit je op 350 (zowel 80 ademh./min als lagere distance gekozen)
# prominence is een maat voor hoeveel een piek uitsteekt t.o.v. omliggende pieken. Dit gaat over de y-as.
peaks_thor, _ = find_peaks(thor, distance=350, prominence=3e-5)  
peaks_abd,  _ = find_peaks(abd, distance=350, prominence=3e-5)

# Vind dalen (expiratie) door min om te draaien
troughs_thor, _ = find_peaks(-thor, distance=350, prominence=3e-5)
troughs_abd,  _ = find_peaks(-abd, distance=350, prominence=3e-5)


#%% Plot maken incl. pieken en dalen
hv.extension('bokeh')

rip_plot = df_small2.hvplot.line(
    y=['RIP_thor_filt','RIP_abd_filt'], 
    width=1000, height=400, title='RIP with extrema'
)

# Markeer thorax pieken en dalen
thor_peaks_points = hv.Scatter((df_small2.index[peaks_thor], thor[peaks_thor])).opts(color='blue', marker='^', size=6)
thor_troughs_points = hv.Scatter((df_small2.index[troughs_thor], thor[troughs_thor])).opts(color='blue', marker='v', size=6)

# Markeer abdomen pieken en dalen
abd_peaks_points = hv.Scatter((df_small2.index[peaks_abd], abd[peaks_abd])).opts(color='orange', marker='^', size=6)
abd_troughs_points = hv.Scatter((df_small2.index[troughs_abd], abd[troughs_abd])).opts(color='orange', marker='v', size=6)

rip_plot * thor_peaks_points * thor_troughs_points * abd_peaks_points * abd_troughs_points


#%% Effect van filtering bekijken
# Plot: voor en na filtering
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Thorax
axs[0].plot(df_small2['RIP_thor'].values, label='Raw RIP_thor', alpha=0.6)
axs[0].plot(df_small2['RIP_thor_filt'].values, label='Filtered RIP_thor', linewidth=2)
axs[0].set_title('RIP Thorax: raw vs filtered')
axs[0].legend()
axs[0].grid(True)

# Abdomen
axs[1].plot(df_small2['RIP_abd'].values, label='Raw RIP_abd', alpha=0.6)
axs[1].plot(df_small2['RIP_abd_filt'].values, label='Filtered RIP_abd', linewidth=2)
axs[1].set_title('RIP Abdomen: raw vs filtered')
axs[1].legend()
axs[1].grid(True)

plt.xlabel("Samples")
plt.tight_layout()
plt.show()


#%% Paradoxale ademhaling selecteren en weergeven in een figuur
time = np.arange(len(thor)) / fs   # tijd-as in seconden
paradox_cycles = []

# abdomen pieken in tijd
abd_trough_times = time[troughs_abd]

# 2. Loop over thorax-cycli (van dal naar dal)
for i in range(len(troughs_thor) - 1):
    t_start_idx = troughs_thor[i]
    t_end_idx   = troughs_thor[i+1]
    cycle_length = t_end_idx - t_start_idx

    # Samples venster 30% - 70%
    # Nu gekozen om tussen twee dalen te kijken van het thorax signaal. De x-waardes van de dalen worden bepaald en de afstand hiertussen is 100%. 
    # Als de x-waarde van het dal van het abdominale signaal tussen 30-70% van de afstand zit, wordt het als paradoxale ademhaling aangemerkt.
    cycle_30 = cycle_length * 0.3
    cycle_70 = cycle_length * 0.7

    # converteer naar echte tijd
    t_start = time[t_start_idx]
    t_end   = time[t_end_idx]
    t_30    = t_start + cycle_30/fs
    t_70    = t_start + cycle_70/fs

    # check of er een dal van abdomen in dit venster valt
    if np.any((abd_trough_times >= t_30) & (abd_trough_times <= t_70)):
        paradox_cycles.append((t_start, t_end))

# Maak curves
thor_curve = hv.Curve((time, thor), 'Tijd (s)', 'Amplitude').opts(color="blue", alpha=0.8, width=900, height=400)
abd_curve  = hv.Curve((time, abd), 'Tijd (s)', 'Amplitude').opts(color="green", alpha=0.8)

# Markeer paradoxale cycli
paradox_regions = [hv.VSpan(start, end).opts(color="red", alpha=0.2)
                   for (start, end) in paradox_cycles]

rip_plot * thor_peaks_points * thor_troughs_points * abd_peaks_points * abd_troughs_points * hv.Overlay(paradox_regions)