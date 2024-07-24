import pandas as pd
from create_label import set_files
from fooof_algorithm import fooof_tool
from welch import welch_method
columns = []
for i in range(19):
    columns.extend([
        f'ch{i+1}_peak_cf',
        f'ch{i+1}_peak_pw',
        f'ch{i+1}_peak_bw',
        f'ch{i+1}_aperiodic_offset',
        f'ch{i+1}_aperiodic_exponent'
    ])
df = pd.DataFrame(columns=columns)
for set_file in set_files:
    frequencies, psd = welch_method(set_file)
    features = fooof_tool(frequencies, psd)
    if len(features) == 95:
        # Thêm các đặc trưng vào DataFrame
        df = df.append(pd.Series(features, index=columns), ignore_index=True)
df.to_csv('X.csv', index=False)
print("done")