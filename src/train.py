import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
import joblib

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Cleans raw text, extracts target labels, and handles the split.
    """

    
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def get_model_pool(random_state=42):
    """Defines 5 models to test"""
    return {
        "Logistic_Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)),
        "Random_Forest": RandomForestClassifier(random_state=random_state),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=random_state),
        "Support_Vector_Machine": make_pipeline(StandardScaler(), SVC(random_state=random_state))
    }


def get_hyperparameter_grids():
    """
    Defines the hyperparameter search spaces for each model.
    Note: For models inside a Pipeline, we use the prefix 'name__' to target parameters.
    """
    return {
        "Logistic_Regression": {
            "logisticregression__C": [0.1, 1.0, 10.0]
        },
        "Random_Forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        },
        "Gradient_Boosting": {
            "n_estimators": [100, 150],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5]
        },
        "Support_Vector_Machine": {
            "svc__C": [0.1, 1.0, 10.0],
            "svc__gamma": ["scale", "auto"]
        }
    }

def evaluate_model_pool(X_train, y_train, cv_folds=5, scoring='f1', random_state=42):
    """
    Step 1: Runs baseline Cross-Validation across all models to pick the winning architecture.
    """
    models = get_model_pool(random_state)
    cv_results = {}
    
    print(f"--- Step 1: Running Baseline {cv_folds}-Fold Cross-Validation ({scoring}) ---")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1)
        mean_score = np.mean(scores)
        cv_results[name] = mean_score
        print(f"{name}: Baseline Mean {scoring} = {mean_score:.4f}")
        
    return cv_results, models


def train_and_save_best_model(best_model_name, best_model_blueprint, X_train, y_train, cv_folds=3, scoring='f1', model_dir=None):
    """
    Takes the winner, performs hyperparameter tuning via GridSearchCV, 
    extracts the absolute best version, and saves it.
    """
    print(f"\n--- Step 2: Hyperparameter Tuning for Winner ({best_model_name}) ---")
    
    # Get the hyperparameter grid specific to the winning model
    param_grids = get_hyperparameter_grids()
    param_grid = param_grids.get(best_model_name, {})
    
    if param_grid:
        print(f"Running GridSearchCV over search space: {param_grid}")
        grid_search = GridSearchCV(
            estimator=best_model_blueprint,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Extract the tuned model configuration
        final_model = grid_search.best_estimator_
        print(f"Tuning Complete! Best Params: {grid_search.best_params_}")
        print(f"Optimized Training {scoring} Score: {grid_search.best_score_:.4f}")
    else:
        print(f"No param grid found for {best_model_name}. Training with default parameters...")
        final_model = best_model_blueprint
        final_model.fit(X_train, y_train)
    

    if model_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(repo_root, 'models')

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.joblib')
    joblib.dump(final_model, model_path, compress=3)
    print(f"Successfully saved compressed model pipeline to: {model_path}") 
    
    return final_model

def find_and_train_best_model(X_train, y_train, cv_folds=5, scoring='f1', random_state=42):
    """
    Orchestrator: Ties Baseline Evaluation and GridSearchCV Tuning together seamlessly.
    """
    cv_results, models = evaluate_model_pool(X_train, y_train, cv_folds, scoring, random_state)
    
    #Select the top-performing model 
    best_model_name = max(cv_results, key=cv_results.get)
    best_model_blueprint = models[best_model_name]
    print(f"\nWinner Selected: {best_model_name} is moving to the tuning phase.")
    
    #Fine-tune the winner model
    final_model = train_and_save_best_model(
        best_model_name=best_model_name,
        best_model_blueprint=best_model_blueprint,
        X_train=X_train,
        y_train=y_train,
        cv_folds=3, 
        scoring=scoring
    )
    return final_model