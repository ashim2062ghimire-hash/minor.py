import os
import pandas as pd
import numpy as np

def preprocess_pipeline(cwru_csv, ansys_csv, output_dir):
    
    print(f"\n || Starting Preprocessing Pipeline || \n")
    
    if not (os.path.exists(cwru_csv) and os.path.exists(ansys_csv)):
        raise FileNotFoundError(f"Processed feature file not found. Please run feature extraction first.")
        
    df_cwru = pd.read_csv(cwru_csv) 
    df_ansys = pd.read_csv(ansys_csv) 

    print(f"Loaded cwru dataset with shape: {df_cwru.shape} (Rows x Columns)")
    X_cwru = df_cwru.iloc[:,:-2].values
    y_cwru = df_cwru.iloc[:,-1].values
    groups_cwru = df_cwru.iloc[:,-2].values

    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X_cwru.npy"), X_cwru)
    np.save(os.path.join(output_dir, "groups_cwru.npy"), groups_cwru)
    np.save(os.path.join(output_dir, "y_cwru.npy"), y_cwru)
    
    print(f"Isolated {X_cwru.shape[1]} engineering features.")
    print("---Saved splitted cwru files to 'data/splits/' ---")

    print(f"Loaded ansys dataset with shape: {df_ansys.shape} (Rows x Columns)")
    X_ansys = df_ansys.iloc[:,:-2].values
    y_ansys = df_ansys.iloc[:,-1].values
    groups_ansys = df_ansys.iloc[:,-2].values

    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X_ansys.npy"), X_ansys)
    np.save(os.path.join(output_dir, "groups_ansys.npy"), groups_ansys)
    np.save(os.path.join(output_dir, "y_ansys.npy"), y_ansys)
    
    print(f"Isolated {X_ansys.shape[1]} engineering features.")
    print("---Saved splitted ansy files to 'data/splits/' ---")
    print("---Pre-processing completed---")
    
if __name__ == "__main__":
    preprocess_pipeline()