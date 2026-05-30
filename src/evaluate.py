import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def evaluate_saved_model(X_test, y_test, model_path=None):
    """
    Loads the best trained model architecture from disk and runs a comprehensive
    evaluation against the unseen holdout test split.
    """
    print(f"\n--- Step 3: Final Evaluation on Test Split ---")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(repo_root, 'models', 'best_model.joblib')
    elif not os.path.isabs(model_path):
        model_path = os.path.join(repo_root, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found at '{model_path}'. "
        )
        
    print(f"Loading saved model from: {model_path}...")
    model = joblib.load(model_path)
    
    #Generate predictions on test data
    predictions = model.predict(X_test)
    
    #Calculate metrics
    final_f1 = f1_score(y_test, predictions, pos_label=1)
    matrix = confusion_matrix(y_test, predictions)
    
    print("\n" + "="*50)
    print(f"Test F1 Score {final_f1:.4f}")
    print("="*50)
    
    print("\nClassification_Report")
    print(classification_report(y_test, predictions, target_names=['Human (0)', 'AI (1)']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\n\nDetailed Confusion Matrix Breakdown:")
    print(f"True Humans flagged as Human (TN): {matrix[0][0]}")
    print(f"True Humans falsely flagged as AI (FP): {matrix[0][1]}")
    print(f"True AI texts falsely flagged as Human (FN): {matrix[1][0]}")
    print(f"True AI texts caught cleanly (TP): {matrix[1][1]}")
    
    return final_f1