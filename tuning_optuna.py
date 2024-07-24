from sklearn.ensemble import RandomForestClassifier
from preprocess_csv_data import X_train, y_train, X_test, y_test

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