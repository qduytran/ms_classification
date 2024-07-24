import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('X.csv', skiprows=0)
df_label = pd.read_csv('Y.csv', skiprows=0)
df_label = df_label.iloc[:, 1:]
columns_to_drop = []
for i in range(95):
    if i%5==0 or i%5==1 or i%5==2:
        columns_to_drop.append(i)

# Loại bỏ các cột có chỉ số đã cho từ DataFrame
df_selected = df.drop(df.columns[columns_to_drop], axis=1)

X = df_selected
y = df_label
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25, random_state=43)