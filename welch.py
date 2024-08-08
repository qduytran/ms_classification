import mne
from scipy.signal import welch, hamming

def welch_method(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    channel_name = raw.info['ch_names']
    channel_index = []
    channel_data = [] 
    channel_time = []
    frequencies = []
    psd = []
    for i in range(19):
        index = raw.ch_names.index(channel_name[i])
        data = raw[index, :][0] * 1e6
        time = raw[index, :][1]

        channel_index.append(index)
        channel_data.append(data)
        channel_time.append(time)
        freqs, power = welch(channel_data[i], fs=raw.info['sfreq'], nperseg=512, window=hamming(512))
        frequencies.append(freqs)
        psd.append(power)
    return frequencies, psd