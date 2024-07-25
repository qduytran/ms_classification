import os
import pandas as pd
from fooof_algorithm import fooof_tool
from welch import welch_method

def create_label(folder_paths):
    #read label and save Y.csv
    data = []
    set_files = read_data(folder_paths)
    for file_path in set_files:
        if 'data\\decreased_cognition' in file_path:
            label = 1
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
    columns = []
    frequencies = []
    psd = []
    new_rows = []
    for i in range(19):
        columns.extend([
            f'ch{i+1}_peak_cf',
            f'ch{i+1}_peak_pw',
            f'ch{i+1}_peak_bw',
            f'ch{i+1}_aperiodic_offset',
            f'ch{i+1}_aperiodic_exponent'
        ])

    for set_file in set_files:
        frequencies, psd = welch_method(set_file)
        features = fooof_tool(frequencies, psd)
        if len(features) == 95:
            new_rows.append(features)
    df_X = pd.DataFrame(new_rows)
    return df_X