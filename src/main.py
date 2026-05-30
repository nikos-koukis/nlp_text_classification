import logging
from pathlib import Path
import pandas as pd
from nlp import process_data_for_pipeline
from train import prepare_data, find_and_train_best_model
from evaluate import evaluate_saved_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def run_pipeline():
    """Orchestrates the entire machine learning engineering lifecycle."""
    
    #Resolve absolute file paths reliably
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "original_data.csv"
    
    logger.info("Initializing ML pipeline execution.")
    
    #Data Ingestion Phase
    if not data_path.exists():
        logger.error(f"Data source file missing at expected location: {data_path}")
        raise FileNotFoundError(f"Missing source dataset: {data_path}")
        
    logger.info(f"Loading raw dataset from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset successfully loaded. Shape: {df.shape}")

    try:
        #NLP Feature Extraction Phase (Text to Vectors)
        logger.info("Starting text vectorization and label mapping via BGE-M3...")
        vector_df = process_data_for_pipeline(df, text_col='text_content', label_col='label')
        logger.info(f"Vectorization complete. Engine DataFrame Shape: {vector_df.shape}")

        #Data Split Phase
        logger.info("Splitting dataset into training and holdout validation sets...")
        X_train, X_test, y_train, y_test = prepare_data(vector_df)
        logger.info(f"Split completed. Train features shape: {X_train.shape} | Test features shape: {X_test.shape}")

        #Model Training & Cross-Validation Architecture Tuning Phase
        logger.info("Initiating model pool cross-validation evaluation and hyperparameter tuning...")
        final_model = find_and_train_best_model(X_train, y_train)
        logger.info("Model tuning complete and top artifact serialized to disk.")

        #Final Evaluation & Metric Generation Phase
        logger.info("Evaluating serialized production model against holdout test split...")
        evaluate_saved_model(X_test, y_test)
        logger.info("ML pipeline execution finished successfully.")

    except Exception as e:
        logger.exception(f"Pipeline execution halted due to an unhandled exception: {str(e)}")
        raise e


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        logger.warning("Pipeline execution manually interrupted by user.")