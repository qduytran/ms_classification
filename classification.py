import optuna
import pickle 
from sklearn.ensemble import RandomForestClassifier
from preprocess_csv_data import X_train, y_train, X_test, y_test
from tuning_optuna import objective

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
score = clf.score(X_test, y_test)
print("Test set accuracy: ", score)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)