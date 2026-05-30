import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def process_data_for_pipeline(df, text_col, label_col):
    """
    Maps text labels to integers, extracts BGE-M3 embeddings, 
    and returns a DataFrame containing only the numerical feature vectors and labels.
    """
    label_mapping = {"human": 0, "ai": 1}
    processed_labels = df[label_col].map(label_mapping).astype(int).values
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    model.max_seq_length = 8192
    
    print(f"Generating embeddings using {device}...")
    emb = model.encode(
        df[text_col].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print("Embeddings shape:", emb.shape)
    
    feature_names = [f"feat_{i}" for i in range(emb.shape[1])]
    pipeline_df = pd.DataFrame(emb, columns=feature_names)
    
    pipeline_df['label'] = processed_labels
    
    return pipeline_df








