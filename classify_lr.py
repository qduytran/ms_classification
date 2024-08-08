import optuna
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=43)
    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(X_train, y_train, shuffle=True, test_size=0.1, random_state=43)
    return X_train_tune, y_train_tune, X_val_tune, y_val_tune, X_test, y_test

def get_objective(X, y):
    def objective(trial):
        X_train_tune, y_train_tune, X_val_tune, y_val_tune, _, _ = split_data(X, y)
        C = trial.suggest_loguniform("C", 1e-5, 1e2)
        solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
        max_iter = trial.suggest_int("max_iter", 100, 1000)

        clf = LogisticRegression(
            C=C, 
            solver=solver, 
            max_iter=max_iter,
            class_weight='balanced',
            random_state=43
        )
        
        clf.fit(X_train_tune, y_train_tune)
        
        y_val_pred = clf.predict(X_val_tune)
        score = accuracy_score(y_val_tune, y_val_pred)
        
        return 1.0 - score
    return objective

def train(X, y):
    X_train_tune, y_train_tune, _, _, _, _ = split_data(X, y)
    objective = get_objective(X, y)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    print("Best hyperparameters: ", study.best_params)
    print("Best accuracy score in val set: ", 1.0 - study.best_value)
    
    best_params = study.best_params
    clf = LogisticRegression(
        C=best_params["C"], 
        solver=best_params["solver"], 
        max_iter=best_params["max_iter"],
        class_weight='balanced',
        random_state=43
    )
    clf.fit(X_train_tune, y_train_tune)
    
    return clf

def test(clf, X_train_tune, y_train_tune, X_val_tune, y_val_tune, X_test, y_test):
    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]  # Probability for the positive class

    train_score = clf.score(X_train_tune, y_train_tune)
    val_score = clf.score(X_val_tune, y_val_tune)
    test_score = clf.score(X_test, y_test)

    auc_score = roc_auc_score(y_test, y_test_prob)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print("Train set accuracy: ", train_score)
    print("Validation set accuracy: ", val_score)
    print("Test set accuracy: ", test_score)
    print("Test set AUC: ", auc_score)
    print("Confusion Matrix:\n", conf_matrix)
    print("F1-score: ", f1)

    return train_score, val_score, test_score, auc_score, conf_matrix, f1

def draw_roc_curve(clf, X_test, y_test):    
    y_test_prob = clf.predict_proba(X_test)[:, 1]  # Probability for the positive class
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker='o', linestyle='-', label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
