import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def smiles_to_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius, fpSize=nbits)
    return gen.GetFingerprint(mol)


def fp_to_array(fp, nbits):
    arr = np.zeros((nbits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _worker(args):
    idx, smiles, radius, nbits = args
    try:
        fp = smiles_to_fp(smiles, radius=radius, nbits=nbits)
        if fp is None:
            return idx, None
        arr = fp_to_array(fp, nbits)
        return idx, arr
    except Exception:
        return idx, None


def compute_ecfp_embeddings(smiles_list, radius=2, nbits=2048, n_workers=None, chunksize=512):
    n_workers = n_workers or cpu_count()
    n = len(smiles_list)

    tasks = [(i, smi, radius, nbits) for i, smi in enumerate(smiles_list)]

    embeddings = np.zeros((n, nbits), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=np.float32)

    with Pool(processes=n_workers) as pool:
        iterator = pool.imap(_worker, tasks, chunksize=chunksize)
        iterator = tqdm(iterator, total=n, desc="Computing ECFP", ncols=100)

        for idx, arr in iterator:
            if arr is not None:
                embeddings[idx] = arr
                valid_mask[idx] = 1.0

    masked_embeddings = embeddings * valid_mask[:, None]
    return masked_embeddings.astype(np.float32), valid_mask.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compute ECFP embeddings from a CSV file in parallel on CPU")
    parser.add_argument("--input_csv", required=True, help="Input CSV path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES column name")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--nbits", type=int, default=2048, help="Fingerprint size")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of CPU workers")
    parser.add_argument("--chunksize", type=int, default=512, help="Chunksize for multiprocessing")
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in {input_csv}")

    smiles_list = df[args.smiles_col].fillna("").astype(str).tolist()

    emb, valid_mask = compute_ecfp_embeddings(
        smiles_list=smiles_list,
        radius=args.radius,
        nbits=args.nbits,
        n_workers=args.n_workers,
        chunksize=args.chunksize,
    )

    smiles_arr = np.asarray(smiles_list, dtype=object)

    np.save(output_dir / "emb.npy", emb)
    np.save(output_dir / "smiles.npy", smiles_arr, allow_pickle=True)

    print(f"Saved embeddings to: {output_dir / 'emb.npy'}")
    print(f"Saved smiles to: {output_dir / 'smiles.npy'}")
    print(f"Total molecules: {len(smiles_list)}")
    print(f"Valid fingerprints: {int(valid_mask.sum())}")
    print(f"Invalid fingerprints: {int((valid_mask == 0).sum())}")


if __name__ == "__main__":
    main()
