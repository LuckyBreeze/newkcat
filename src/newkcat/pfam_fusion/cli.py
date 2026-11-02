"""Pfam × CataPro isolated experiment entrypoint."""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import (PfamDataset, audit, check_disjoint, choose_nmax, fit_vocab,
                   read_deepbgc, read_hits, read_split)
from .model import PfamCataPro

DEEPBGC_URL = "https://github.com/Merck/deepbgc/releases/download/v0.1.0/pfam2vec.csv.gz"


def config_at(path):
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("data"), dict):
        raise ValueError("Config must have data and model dictionaries")
    if cfg["model"]["source"] not in ("random", "deepbgc"):
        raise ValueError("Invalid embedding source")
    if cfg["model"]["variant"] not in ("set", "ordered"):
        raise ValueError("Invalid Pfam mode")
    return cfg


def resource(cfg, name):
    value = cfg["data"].get(name)
    if not value:
        raise ValueError(f"Missing data.{name} in YAML")
    return Path(value).expanduser().resolve()


def prepare(cfg, with_test=False):
    train = read_split(resource(cfg, "train_csv"), "train")
    val = read_split(resource(cfg, "valid_csv"), "valid")
    test = read_split(resource(cfg, "test_csv"), "test") if with_test else []
    check_disjoint(*(tuple([train,val,test]) if with_test else (train,val)))
    hits = read_hits(resource(cfg, "hits_csv"))
    vocab = fit_vocab(train, hits)
    args = cfg["model"]
    nmax = choose_nmax(train, hits, float(args.get("nmax_quantile",.99)),
                       int(args.get("nmax_cap",32)))
    return train,val,test,hits,vocab,nmax


def pretrained_vectors(cfg, vocab):
    if cfg["model"]["source"] == "random":
        return None, None
    vec = read_deepbgc(resource(cfg,"deepbgc_vectors"))
    table = torch.randn((len(vocab),100))*.02
    table[0].zero_()
    for pfam,idx in vocab.items():
        if idx>1 and pfam in vec:
            table[idx] = torch.tensor(vec[pfam])
    return table, vec


def create_model(cfg, vocab, nmax, table=None):
    return PfamCataPro(
        len(vocab), nmax, cfg["model"]["variant"] == "ordered",
        pretrained=table,
        dropout=float(cfg["model"].get("dropout", 0.1)),
        fc_dropout=float(cfg["model"].get("fc_dropout", 0.0)),
        init_scheme=cfg["model"].get("init_scheme", "legacy"),
        layers=int(cfg["model"].get("transformer_layers", 1)),
    )


def dataset(cfg, rows, hits, vocab, nmax):
    return PfamDataset(rows,hits,vocab,nmax,cfg["model"]["variant"]=="ordered",
                       resource(cfg,"features_npz"))


def device_for(cfg):
    request=cfg.get("training",{}).get("device","auto")
    if request=="auto":
        request="cuda:0" if torch.cuda.is_available() else "cpu"
    if str(request).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(request)


def batches_without_singleton(n,size,seed):
    if n<2 or size<2:
        raise ValueError("BatchNorm needs at least two training samples per batch")
    rng=np.random.default_rng(seed)
    order=rng.permutation(n).tolist()
    batches=[order[i:i+size] for i in range(0,n,size)]
    if len(batches)>1 and len(batches[-1])==1:
        batches[-2].extend(batches.pop())
    return batches


def move(batch, device):
    return {k:v.to(device) for k,v in batch.items()}


@torch.no_grad()
def inference(model,ds,device,batch_size):
    model.eval()
    result=[]
    for batch in DataLoader(ds,batch_size=batch_size):
        result.append(model(move(batch,device)).detach().cpu().numpy())
    return np.concatenate(result)


def metrics(y, p):
    """Full-dataset metrics in log10(kcat) space, never averaged across batches."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or len(y) < 1:
        raise ValueError("Expected matching 1-D nonempty target/prediction arrays")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Predictions and targets must be finite")
    mse = float(np.mean(np.square(p - y)))
    total = float(np.square(y - y.mean()).sum())
    r2 = 1 - float(np.square(p - y).sum()) / total if total > 0 else None
    pcc = None
    if len(y) > 1 and float(np.std(y)) > 0 and float(np.std(p)) > 0:
        pcc = float(np.corrcoef(y, p)[0, 1])
    return {
        "n": len(y),
        "loss": mse, "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(p - y))),
        "r2": r2, "pcc": pcc,
    }


def build_optimizer(model, lr, weight_decay):
    """AdamW decays matrices, but not scalar/bias/norm parameters."""
    if not lr > 0 or not weight_decay >= 0:
        raise ValueError("lr must be positive and weight_decay nonnegative")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def scheduled_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio):
    """Step-wise linear warmup followed by a smooth cosine ending at a fixed floor."""
    if not 1 <= step <= total_steps:
        raise ValueError("step must be in [1, total_steps]")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0 <= min_lr_ratio <= 1 or base_lr <= 0:
        raise ValueError("Invalid base_lr or min_lr_ratio")
    if warmup_steps and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


def train(cfg, output, init_catapro=None):
    """Train with epoch-level train/validation eval metrics and durable CSV history."""
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}")
    tr, va, _, hits, vocab, nmax = prepare(cfg)
    ordered = cfg["model"]["variant"] == "ordered"
    minimum = float(cfg["model"].get("min_train_pfam_fraction", 0.9))
    train_audit = audit(tr, hits, vocab, nmax, ordered)
    if train_audit["annotation_hit_fraction"] < minimum:
        raise ValueError("Pfam hit row_uid coverage insufficient; supply real annotations")
    ds_tr, ds_va = (dataset(cfg, rows, hits, vocab, nmax) for rows in (tr, va))

    pars = cfg.get("training", {})
    seed = int(pars.get("seed", 668))
    batch_size = int(pars.get("batch_size", 8))
    epochs = int(pars.get("epochs", 50))
    patience = int(pars.get("patience", 30))
    warmup_epochs = int(pars.get("warmup_epochs", 3))
    base_lr = float(pars.get("lr", 5e-5))
    decay = float(pars.get("weight_decay", 1e-3))
    min_lr_ratio = float(pars.get("min_lr_ratio", 0.1))
    min_delta = float(pars.get("min_delta", 1e-4))
    if epochs < 1 or patience < 1 or not 0 <= warmup_epochs < epochs:
        raise ValueError("Require epochs/patience >=1 and 0 <= warmup_epochs < epochs")
    if batch_size < 2:
        raise ValueError("CataPro BatchNorm requires batch_size >= 2")
    if not 0 <= min_delta < float("inf"):
        raise ValueError("min_delta must be finite and nonnegative")
    batches_per_epoch = len(batches_without_singleton(len(ds_tr), batch_size, seed))
    total_steps = epochs * batches_per_epoch
    warmup_steps = warmup_epochs * batches_per_epoch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    table, vectors = pretrained_vectors(cfg, vocab)
    model = create_model(cfg, vocab, nmax, table)
    if init_catapro:
        weights = torch.load(init_catapro, map_location="cpu", weights_only=True)
        model.initialize_catapro(weights.get("state_dict", weights))
    device = device_for(cfg)
    model.to(device)
    opt = build_optimizer(model, base_lr, decay)

    output.mkdir(parents=True, exist_ok=False)
    history_csv = output / "history.csv"
    fields = [
        "epoch", "train_online_loss", "train_loss", "train_rmse",
        "val_loss", "val_rmse", "val_mae", "val_r2", "val_pcc",
        "lr_first", "lr_last", "best_epoch_so_far", "early_stop_counter",
    ]
    history = []
    best = math.inf
    best_epoch = None
    stale = 0
    global_step = 0
    best_val_predictions = None
    sequence_hashes = [hashlib.sha256(r["sequence"].encode()).hexdigest() for r in tr + va]
    cluster_ids = sorted({r["cluster_id"] for r in tr + va if r["cluster_id"]})

    with history_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        file.flush()
        for epoch in range(1, epochs + 1):
            model.train()
            online_squared_sum = 0.0
            online_n = 0
            lr_first = None
            lr_last = None
            batches = batches_without_singleton(len(ds_tr), batch_size, seed + epoch)
            for batch in DataLoader(ds_tr, batch_sampler=batches):
                batch = move(batch, device)
                global_step += 1
                current_lr = scheduled_lr(
                    global_step, total_steps, warmup_steps, base_lr, min_lr_ratio
                )
                for group in opt.param_groups:
                    group["lr"] = current_lr
                if lr_first is None:
                    lr_first = current_lr
                lr_last = current_lr
                opt.zero_grad(set_to_none=True)
                pred = model(batch)
                if pred.shape != batch["target"].shape:
                    raise ValueError("Target shape mismatch; broadcasting forbidden")
                squared = torch.square(pred - batch["target"])
                loss = squared.mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                online_squared_sum += float(squared.detach().sum().item())
                online_n += batch["target"].numel()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(pars.get("grad_clip", 5))
                )
                opt.step()

            # Same eval() + complete-data rules for end-of-epoch train and val.
            train_pred = inference(model, ds_tr, device, batch_size)
            val_pred = inference(model, ds_va, device, batch_size)
            train_values = metrics(ds_tr.labels.numpy(), train_pred)
            val_values = metrics(ds_va.labels.numpy(), val_pred)
            improved = val_values["rmse"] < best - min_delta
            if improved:
                best = val_values["rmse"]
                best_epoch = epoch
                stale = 0
                best_val_predictions = np.array(val_pred, copy=True)
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                checkpoint = {
                    "state_dict": state, "vocab": vocab, "nmax": nmax,
                    "source": cfg["model"]["source"],
                    "variant": cfg["model"]["variant"],
                    "dropout": float(cfg["model"].get("dropout", 0.1)),
                    "fc_dropout": float(cfg["model"].get("fc_dropout", 0.0)),
                    "init_scheme": cfg["model"].get("init_scheme", "legacy"),
                    "layers": int(cfg["model"].get("transformer_layers", 1)),
                    "train_val_sequence_hashes": sequence_hashes,
                    "train_val_clusters": cluster_ids,
                    "best_epoch": best_epoch,
                    "validation_rmse": best,
                    "seed": seed,
                    "optimizer": "AdamW",
                    "training_hyperparameters": {
                        "batch_size": batch_size, "epochs": epochs,
                        "warmup_epochs": warmup_epochs, "lr": base_lr,
                        "min_lr_ratio": min_lr_ratio, "weight_decay": decay,
                        "patience": patience, "min_delta": min_delta,
                    },
                }
                torch.save(checkpoint, output / "best.pt")
            else:
                stale += 1
            row = {
                "epoch": epoch,
                "train_online_loss": online_squared_sum / online_n,
                "train_loss": train_values["loss"],
                "train_rmse": train_values["rmse"],
                "val_loss": val_values["loss"],
                "val_rmse": val_values["rmse"],
                "val_mae": val_values["mae"],
                "val_r2": val_values["r2"],
                "val_pcc": val_values["pcc"],
                "lr_first": lr_first, "lr_last": lr_last,
                "best_epoch_so_far": best_epoch,
                "early_stop_counter": stale,
            }
            history.append(row)
            writer.writerow(row)
            file.flush()
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            if stale >= patience:
                break

    if best_epoch is None or best_val_predictions is None:
        raise RuntimeError("No finite validation model")
    with (output / "best_valid_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["row_uid", "observed", "predicted"])
        for sample, pred in zip(va, best_val_predictions):
            writer.writerow([sample["row_uid"], sample["target"], float(pred)])
    report = {
        "best_epoch": best_epoch, "validation_rmse": best,
        "nmax": nmax, "vocab_size": len(vocab),
        "training_history": history,
        "train_audit": audit(tr, hits, vocab, nmax, ordered,
                             set(vectors) if vectors else None),
        "validation_audit": audit(va, hits, vocab, nmax, ordered,
                                  set(vectors) if vectors else None),
        "history_csv": str(history_csv),
        "train_loss_definition": "eval-mode full-train MSE at epoch end",
        "train_online_loss_definition": "sample-weighted train-mode MSE during updates",
        "validation_selection": "minimum complete-validation RMSE",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def evaluate(cfg,checkpoint,output):
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if cfg["model"]["variant"]!=saved["variant"] or cfg["model"]["source"]!=saved["source"]:
        raise ValueError("Wrong checkpoint for this experiment")
    te=read_split(resource(cfg,"test_csv"),"test")
    used=set(saved["train_val_sequence_hashes"])
    groups=set(saved["train_val_clusters"])
    if any(hashlib.sha256(r["sequence"].encode()).hexdigest() in used or
           (r["cluster_id"] and r["cluster_id"] in groups) for r in te):
        raise ValueError("Test leakage detected")
    hits=read_hits(resource(cfg,"hits_csv"))
    ds=dataset(cfg,te,hits,saved["vocab"],saved["nmax"])
    model=PfamCataPro(
        len(saved["vocab"]), saved["nmax"], saved["variant"]=="ordered",
        dropout=saved["dropout"], layers=saved["layers"],
        fc_dropout=saved.get("fc_dropout",0.0),
        init_scheme=saved.get("init_scheme","legacy"),
    )
    model.load_state_dict(saved["state_dict"],strict=True)
    device=device_for(cfg);model.to(device)
    predictions=inference(model,ds,device,int(cfg.get("training",{}).get("batch_size",64)))
    result=metrics(ds.labels.numpy(),predictions)
    result["audit"]=audit(te,hits,saved["vocab"],saved["nmax"],saved["variant"]=="ordered")
    output.mkdir(parents=True,exist_ok=False)
    with (output/"predictions.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["row_uid","observed","predicted"])
        for row,p in zip(te,predictions):
            writer.writerow([row["row_uid"],row["target"],float(p)])
    (output/"metrics.json").write_text(json.dumps(result,indent=2))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    for verb in ("audit","train","evaluate"):
        p=sub.add_parser(verb);p.add_argument("--config",required=True,type=Path)
        if verb!="audit":p.add_argument("--output",required=True,type=Path)
        if verb=="train":p.add_argument("--init-catapro",type=Path)
        if verb=="evaluate":p.add_argument("--checkpoint",required=True,type=Path)
    args=parser.parse_args(argv)
    cfg=config_at(args.config)
    if args.command=="audit":
        tr,va,te,hits,vocab,nmax=prepare(cfg,True)
        vec=read_deepbgc(resource(cfg,"deepbgc_vectors")) if cfg["model"]["source"]=="deepbgc" else None
        result={"nmax_train_only":nmax,"vocab_train_only":len(vocab)}
        for key,rows in (("train",tr),("valid",va),("test_descriptive_only",te)):
            result[key]=audit(rows,hits,vocab,nmax,cfg["model"]["variant"]=="ordered",
                              set(vec) if vec else None)
    elif args.command=="train":
        result=train(cfg,args.output,args.init_catapro)
    else:result=evaluate(cfg,args.checkpoint,args.output)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
