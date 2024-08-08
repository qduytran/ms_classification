import optuna
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, f1_score

def split_data(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=43)
    
    # Further split the training data into training and validation sets for hyperparameter tuning
    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(X_train, y_train, shuffle=True, test_size=0.1, random_state=43)
    return X_train_tune, y_train_tune, X_val_tune, y_val_tune, X_test, y_test

# Define the objective function for Optuna
def get_objective(X, y):
    def objective(trial):
        X_train_tune, y_train_tune, X_val_tune, y_val_tune, _, _ = split_data(X, y)
        # Define the hyperparameters to tune
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 10)
         
        # Create a random forest classifier
        clf = RandomForestClassifier(
            class_weight = 'balanced',
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
    return objective
 
def train(X, y):
    # Use Optuna to tune the hyperparameters
    X_train_tune, y_train_tune, _, _, _, _ = split_data(X, y)
    objective = get_objective(X, y)
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    
    # Print the best hyperparameters and the best score
    print("Best hyperparameters: ", study.best_params)
    print("Best accuracy score in val set: ", 1.0 - study.best_value)
    
    # Train the classifier with the best hyperparameters on the full training set
    best_params = study.best_params
    clf = RandomForestClassifier(
        class_weight = 'balanced',
        n_estimators=best_params["n_estimators"], 
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"], 
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=43
    )
    clf.fit(X_train_tune, y_train_tune)
    
    return clf 

def test(clf, X_train_tune, y_train_tune, X_val_tune, y_val_tune, X_test, y_test):
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
    
    return train_score, val_score, test_score, auc_score, conf_matrix, f1

def draw_roc_curve(clf, X_test, y_test):    
    y_test_prob = clf.predict_proba(X_test)[:, 1]
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