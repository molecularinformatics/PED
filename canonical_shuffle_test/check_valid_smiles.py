import os
import numpy as np
import pandas as pd


root_dir = "/mnt/depts/dept01/misd/molml/embeddings/shuffle"

# loop through all the subfolders of root_dir,
# and check the length of smiles.npy and results.csv
for setting in os.listdir(root_dir):
    setting_path = os.path.join(root_dir, setting)
    if not os.path.isdir(setting_path):
        continue

    smiles_path = os.path.join(setting_path, "smiles.npy")
    results_path = os.path.join(setting_path, "results.csv")

    smiles_data = np.load(smiles_path, allow_pickle=True)
    num_smiles = len(smiles_data)

    results = pd.read_csv(results_path)
    num_results = len(results)

    print(f"{setting}: smiles.npy has {num_smiles} entries, results.csv has {num_results} entries.")