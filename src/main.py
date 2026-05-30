from nlp import process_data_for_pipeline
from train import prepare_data, find_and_train_best_model
from evaluate import evaluate_saved_model
from IPython.display import Image, display
import pandas as pd 

FILE_PATH = "data/original_data.csv"

#Get data
df = pd.read_csv(FILE_PATH)

#Data processing for ML pipeline | Text to vectors
vector_df = process_data_for_pipeline(df, text_col='text_content', label_col='label')

#Train test split
X_train, X_test, y_train, y_test = prepare_data(vector_df)

find_and_train_best_model(X_train, y_train)
evaluate_saved_model(X_test, y_test)








