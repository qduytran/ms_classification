import os 
import pandas as pd

folder_paths = ['data\\decreased_cognition', 'data\\intact_cognition']
data = []
def read_data(folder_paths):
    set_files = []
    for folder_path in folder_paths:
        files = []
        for f in os.listdir(folder_path):
            if f.endswith('.set'):
                files.append(os.path.join(folder_path, f))
        set_files.extend(files)
    return set_files

set_files = read_data(folder_paths)
for file_path in set_files:
    if 'data\\decreased_cognition' in file_path:
        label = 1
    else:
        label = 0
    data.append([file_path, label])
df = pd.DataFrame(data, columns=['file_path', 'label'])
df.to_csv('Y.csv', index=False)