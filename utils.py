import os
import scipy.io
import pandas as pd
import numpy as np
from fooof_algorithm import fooof_tool, fooof_tool_from_mat_file
from welch import welch_method

def create_label(folder_paths):
    #read label and save Y.csv
    data = []
    set_files = read_data(folder_paths)
    for file_path in set_files:
        if 'data_new\\AD' in file_path:
            label = 1
        elif 'data_new\\FTD' in file_path:
            label = 2
        else:
            label = 0
        data.append([file_path, label])
    df_Y = pd.DataFrame(data)
    df_Y = df_Y.iloc[:, 1:]
    return set_files, df_Y

def read_data(folder_paths):
    set_files = []
    for folder_path in folder_paths:
        files = []
        for f in os.listdir(folder_path):
            if f.endswith('.set'):
                files.append(os.path.join(folder_path, f))
        set_files.extend(files)
    return set_files

def create_data(set_files):
    frequencies = []
    psd = []
    new_rows = []

    for set_file in set_files:
        frequencies, psd = welch_method(set_file)
        features = fooof_tool(frequencies, psd)
        if len(features) == 95:
            new_rows.append(features)
    df_X = pd.DataFrame(new_rows)
    return df_X

def create_features_from_mat_data(mat_path):
    eeg_data = scipy.io.loadmat(mat_path)
    psdG = eeg_data['psdG'][0]
    frequencies = []
    psd = []
    label = []
    new_rows = []
    for patience in psdG:
        frequencies.append(patience[2])
        psd.append(patience[3])
        label.append(patience[4])
    frequencies = np.array(frequencies)
    psd = np.array(psd)
    label = np.array(label)
    label = np.squeeze(label)
    for i in range(146): #146 benh nhan
        features = fooof_tool_from_mat_file(frequencies[i], psd[i])
        if len(features) == 95: 
            new_rows.append(features)
    df_X = pd.DataFrame(new_rows)
    df_Y = pd.DataFrame(label)
    return df_X, df_Y
