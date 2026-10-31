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
    return PfamCataPro(len(vocab),nmax,cfg["model"]["variant"]=="ordered",
                        pretrained=table,dropout=float(cfg["model"].get("dropout",.1)),
                        layers=int(cfg["model"].get("transformer_layers",1)))


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


def metrics(y,p):
    return {"n":len(y),"rmse":float(np.sqrt(np.mean((p-y)**2))),
            "mae":float(np.mean(np.abs(p-y)))}


def train(cfg,output,init_catapro=None):
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}")
    tr,va,_,hits,vocab,nmax=prepare(cfg)
    minimum=float(cfg["model"].get("min_train_pfam_fraction",.9))
    if audit(tr,hits,vocab,nmax,cfg["model"]["variant"]=="ordered")["annotation_hit_fraction"]<minimum:
        raise ValueError("Pfam hit row_uid coverage insufficient; supply real Pfam annotations")
    features=resource(cfg,"features_npz")
    ds_tr,ds_va=(dataset(cfg,rows,hits,vocab,nmax) for rows in (tr,va))
    pars=cfg.get("training",{})
    seed=int(pars.get("seed",17))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    table, vec=pretrained_vectors(cfg,vocab)
    model=create_model(cfg,vocab,nmax,table)
    if init_catapro:
        weights=torch.load(init_catapro,map_location="cpu",weights_only=True)
        model.initialize_catapro(weights.get("state_dict",weights))
    device=device_for(cfg)
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(pars.get("lr",1e-4)),
                          weight_decay=float(pars.get("weight_decay",1e-5)))
    batch_size=int(pars.get("batch_size",64))
    best=math.inf; best_state=None; best_epoch=None; stale=0; history=[]
    for epoch in range(1,int(pars.get("epochs",150))+1):
        model.train()
        losses=[]
        batches=batches_without_singleton(len(ds_tr),batch_size,seed+epoch)
        for batch in DataLoader(ds_tr,batch_sampler=batches):
            batch=move(batch,device)
            opt.zero_grad(set_to_none=True)
            pred=model(batch)
            if pred.shape != batch["target"].shape:
                raise ValueError("Target shape mismatch; broadcasting forbidden")
            loss=((pred-batch["target"])**2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(pars.get("grad_clip",5)))
            opt.step()
            losses.append(loss.item())
        predicted=inference(model,ds_va,device,batch_size)
        val=metrics(ds_va.labels.numpy(),predicted)
        row={"epoch":epoch,"training_loss":float(np.mean(losses)),**val}
        history.append(row);print(json.dumps(row),flush=True)
        if val["rmse"] < best-float(pars.get("min_delta",1e-4)):
            best=val["rmse"];best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale >= int(pars.get("patience",20)):
                break
    if best_state is None:
        raise RuntimeError("No finite validation model")
    output.mkdir(parents=True,exist_ok=False)
    training_sequences=[hashlib.sha256(r["sequence"].encode()).hexdigest() for r in tr+va]
    checkpoint={"state_dict":best_state,"vocab":vocab,"nmax":nmax,
                "source":cfg["model"]["source"],"variant":cfg["model"]["variant"],
                "dropout":float(cfg["model"].get("dropout",.1)),
                "layers":int(cfg["model"].get("transformer_layers",1)),
                "train_val_sequence_hashes":training_sequences,
                "train_val_clusters":sorted({r["cluster_id"] for r in tr+va if r["cluster_id"]}),
                "best_epoch":best_epoch,"validation_rmse":best}
    torch.save(checkpoint,output/"best.pt")
    report={"best_epoch":best_epoch,"validation_rmse":best,"nmax":nmax,
            "vocab_size":len(vocab),"training_history":history,
            "train_audit":audit(tr,hits,vocab,nmax,cfg["model"]["variant"]=="ordered",set(vec) if vec else None),
            "validation_audit":audit(va,hits,vocab,nmax,cfg["model"]["variant"]=="ordered",set(vec) if vec else None)}
    (output/"report.json").write_text(json.dumps(report,indent=2))
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
    model=PfamCataPro(len(saved["vocab"]),saved["nmax"],saved["variant"]=="ordered",
                       dropout=saved["dropout"],layers=saved["layers"])
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
