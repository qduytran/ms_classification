import optuna
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import create_label, create_data

folder_paths = ['data\\decreased_cognition', 'data\\intact_cognition']
columns_to_drop = []
set_files, df_Y = create_label(folder_paths)
df_Y.to_csv('Y.csv', header=False, index=False)
df_X = create_data(set_files) 
df_X.to_csv('X.csv',header=False, index=False)

#feature selection
for i in range(95):
    if i%5==0 or i%5==1 or i%5==2:
        columns_to_drop.append(i)
df_selected = df_X.drop(df_X.columns[columns_to_drop], axis=1)

#split data
X = df_selected
y = df_Y
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25, random_state=43)

#tuning hyperparameter 
def objective(trial):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 7)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    
    # Create a random forest classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf
    )
    # Train the classifier and calculate the accuracy on the validation set
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    return 1.0 - score

# Use Optuna to tune the hyperparameters
study = optuna.create_study()
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the best score
print("Best hyperparameters: ", study.best_params)
print("Best score: ", 1.0 - study.best_value)

# Train the classifier with the best hyperparameters on the full training set
best_params = study.best_params
clf = RandomForestClassifier(
    n_estimators=best_params["n_estimators"], 
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"], 
    min_samples_leaf=best_params["min_samples_leaf"]
)
clf.fit(X_train, y_train)
# Evaluate the tuned classifier on the test set
test_score = clf.score(X_test, y_test)
train_score = clf.score(X_train, y_train)
print("Train set accuracy: ", train_score)
print("Test set accuracy: ", test_score)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)