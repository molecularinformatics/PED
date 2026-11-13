import os
import json
import argparse
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer


def mean_pool_last_hidden(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, L, 1)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1)
    return summed / lengths


def get_pretrained_transformer(model_name):
    if model_name == "ibm/MoLFormer-XL-both-10pct":
        model = AutoModel.from_pretrained(
            model_name,
            deterministic_eval=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    elif model_name == "DeepChem/MoLFormer-c3-1.1B":
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model.eval()
    return tokenizer, model


def _dist_info():
    """
    Works with torchrun or SLURM+torchrun.
    If not distributed, returns (False, 0, 1, 0).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def _dist_init():
    is_dist, rank, world_size, local_rank = _dist_info()
    if not is_dist:
        return False, rank, world_size, local_rank

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


@torch.no_grad()
def save_molformer_embeddings_multi_gpu(
    csv_path: str,
    cache_dir: str,
    smiles_col: str = "smiles",
    model_name: str = "ibm/MoLFormer-XL-both-10pct",
    batch_size: int = 256,
):
    is_dist, rank, world_size, local_rank = _dist_init()

    os.makedirs(cache_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = f"cuda:{local_rank}"

    # ---------- load csv (rank0 does IO), then broadcast arrays ----------
    if rank == 0:
        df = pd.read_csv(csv_path, usecols=[smiles_col])
        df = df[[smiles_col]].dropna().reset_index(drop=True)
        smiles = df[smiles_col].astype(str).to_numpy(dtype=object)

        np.save(os.path.join(cache_dir, "smiles.npy"), smiles, allow_pickle=True)

        n = int(len(smiles))
    else:
        smiles = None
        n = 0

    if is_dist:
        n_t = torch.tensor([n], dtype=torch.int64, device=device if "cuda" in device else "cpu")
        dist.broadcast(n_t, src=0)
        n = int(n_t.item())

        # Broadcast smiles via object gather (simple, OK because rank0 already saved to disk).
        # Better: just have non-rank0 load from disk after barrier (less communication).
        dist.barrier()
        if rank != 0:
            smiles = np.load(os.path.join(cache_dir, "smiles.npy"), allow_pickle=True)
        else:
            # rank0 already has smiles
            pass
    else:
        # single process: smiles already in memory
        pass

    # ---------- model ----------
    tokenizer, model = get_pretrained_transformer(model_name)
    model = model.to(device)

    d = int(model.config.hidden_size)

    # ---------- create memmap file on rank0, then barrier ----------
    emb_path = os.path.join(cache_dir, "emb.npy")
    if rank == 0:
        emb_mm0 = np.lib.format.open_memmap(
            emb_path,
            mode="w+",
            dtype=np.float32,
            shape=(n, d),
        )
        emb_mm0.flush()
        del emb_mm0

        meta = {"model_name": model_name, "embedding_dim": d, "n_samples": n, "world_size": world_size}
        with open(os.path.join(cache_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    if is_dist:
        dist.barrier()

    # open memmap in r+ mode on all ranks
    emb_mm = np.lib.format.open_memmap(
        emb_path,
        mode="r+",
        dtype=np.float32,
        shape=(n, d),
    )

    # ---------- shard indices ----------
    # each rank handles indices: rank, rank+world_size, ...
    idx_all = np.arange(n, dtype=np.int64)
    idx_rank = idx_all[rank::world_size]

    # progress bar per rank
    pbar = tqdm.tqdm(
        total=len(idx_rank),
        desc=f"rank {rank}/{world_size}",
        disable=(rank != 0 and world_size > 1),  # show only rank0 by default
    )

    # batch over idx_rank
    for start in range(0, len(idx_rank), batch_size):
        batch_idx = idx_rank[start:start + batch_size]
        batch_smiles = smiles[batch_idx].tolist()

        tok = tokenizer(batch_smiles, padding=True, return_tensors="pt")
        tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}

        out = model(**tok)
        emb = mean_pool_last_hidden(out.last_hidden_state, tok["attention_mask"])
        emb = emb.detach().cpu().numpy().astype(np.float32)

        emb_mm[batch_idx] = emb
        pbar.update(len(batch_idx))

    pbar.close()
    emb_mm.flush()

    if is_dist:
        dist.barrier()
        if rank == 0:
            print(f"Saved: {cache_dir} (n={n}, d={d}, world_size={world_size})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--smiles_col", type=str, default="smiles")
    ap.add_argument("--model_name", type=str, default="ibm/MoLFormer-XL-both-10pct")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    save_molformer_embeddings_multi_gpu(
        csv_path=args.csv_path,
        cache_dir=args.cache_dir,
        smiles_col=args.smiles_col,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
