import optuna
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, f1_score
from utils import create_label, create_data, create_features_from_mat_data

#df_X, df_Y = create_features_from_mat_data("msEEG_1")
#folder_paths = ['data\\decreased_cognition', 'data\\intact_cognition']
columns_to_drop = []
#set_files, df_Y = create_label(folder_paths)
#df_Y.to_csv('Y.csv', header=False, index=False)
#df_X = create_data(set_files) 

df_X = pd.read_csv("X-60.csv")
df_Y = pd.read_csv("Y-60.csv")
#df_X.to_csv('X.csv',header=False, index=False)

#feature selection
for i in range(95):
    if i%5==2:
        columns_to_drop.append(i)
df_selected = df_X.drop(df_X.columns[columns_to_drop], axis=1)

X = df_selected
y = df_Y

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=43)

# Further split the training data into training and validation sets for hyperparameter tuning
X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(X_train, y_train, shuffle=True, test_size=0.1, random_state=43)

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    
    # Create a random forest classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf,
        random_state=43
    )
    
    # Train the classifier on the tuning training set
    clf.fit(X_train_tune, y_train_tune)
    
    # Predict probabilities on the validation set
    y_val_pred = clf.predict(X_val_tune)
    score = accuracy_score(y_val_tune, y_val_pred)
    
    return 1.0 - score

# Use Optuna to tune the hyperparameters
study = optuna.create_study()
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the best score
print("Best hyperparameters: ", study.best_params)
print("Best accuracy score in val set: ", 1.0 - study.best_value)

# Train the classifier with the best hyperparameters on the full training set
best_params = study.best_params
clf = RandomForestClassifier(
    n_estimators=best_params["n_estimators"], 
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"], 
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=43
)
clf.fit(X_train_tune, y_train_tune)

# Predict on the test set
y_test_pred = clf.predict(X_test)
y_test_prob = clf.predict_proba(X_test)[:, 1]

# Evaluate the tuned classifier
train_score = clf.score(X_train_tune, y_train_tune)
val_score = clf.score(X_val_tune, y_val_tune)
test_score = clf.score(X_test, y_test)
auc_score = roc_auc_score(y_test, y_test_prob)
conf_matrix = confusion_matrix(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("Train set accuracy: ", train_score)
print("Vadilation set accuracy: ",val_score)
print("Test set accuracy: ", test_score)
print("Test set AUC: ", auc_score)
print("Confusion Matrix:\n", conf_matrix)
print("F1-score: ", f1)

fpr, tpr, _ = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, marker='o', linestyle='-', color='b', label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)