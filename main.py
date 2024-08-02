import pandas as pd
import pickle  
from utils import create_label, create_data, create_features_from_mat_data
from train import train, split_data, test, draw_roc_curve
#folder_paths = ['data\\decreased_cognition', 'data\\intact_cognition']

#df_X, df_Y = create_features_from_mat_data("msEEG_1")

#set_files, df_Y = create_label(folder_paths)
#df_X = create_data(set_files) 
#df_X.to_csv('X-test.csv', header=False, index=False)
#df_Y.to_csv('Y-test.csv', header=False, index=False)

df_X = pd.read_csv("X.csv")
df_Y = pd.read_csv("Y.csv")

#feature selection
columns_to_drop = []
for i in range(95):
    if i%5==0 or i%5==1 or i%5==2:
        columns_to_drop.append(i)
df_selected = df_X.drop(df_X.columns[columns_to_drop], axis=1)

X = df_selected
y = df_Y

X_train_tune, y_train_tune, X_val_tune, y_val_tune, X_test, y_test = split_data(X, y)
model = train(X, y)
test(model, X_train_tune, y_train_tune, X_val_tune, y_val_tune, X_test, y_test)
draw_roc_curve(model, X_test, y_test)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)