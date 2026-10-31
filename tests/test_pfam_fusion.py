"""Unit checks for the isolated four-way Pfam branch."""
import csv
import gzip
import torch
import pytest
from newkcat.pfam_fusion.data import read_hits, fit_vocab, encode, ordered_tokens, read_deepbgc
from newkcat.pfam_fusion.model import PfamEncoder, PfamCataPro
from newkcat.model import ModelConfig, NewKcat


def test_repeat_and_order(tmp_path):
    file=tmp_path/"hits.csv"
    file.write_text("row_uid,pfam_id,start,end\nx,PF00002.4,30,60\nx,PF00001,1,18\nx,PF00002.4,70,92\nx,PF00002.4,30,60\n")
    hits=read_hits(file)
    assert ordered_tokens("x",hits,False)==["PF00001","PF00002"]
    assert ordered_tokens("x",hits,True)==["PF00001","PF00002","PF00002"]
    vocab=fit_vocab([{"row_uid":"x"}],hits)
    assert encode("x",hits,vocab,4,True).tolist()==[vocab["PF00001"],vocab["PF00002"],vocab["PF00002"],0]


@pytest.mark.parametrize("ordered",[False,True])
def test_masked_padding_and_backward(ordered):
    torch.manual_seed(4)
    encoder=PfamEncoder(5,4,ordered,dropout=0)
    sample=torch.tensor([[2,3,0,0],[0,0,0,0]],dtype=torch.long)
    prediction=encoder(sample)
    assert prediction.shape==(2,256)
    assert torch.count_nonzero(prediction[1])==0
    prediction.sum().backward()
    assert torch.isfinite(encoder.embedding.weight.grad).all()


def test_pretrained_mapping_and_unknown():
    torch.manual_seed(2)
    weights=torch.randn(5,100)
    encoder=PfamEncoder(5,3,True,pretrained=weights,dropout=0)
    assert torch.allclose(encoder.embedding.weight[2:],weights[2:])
    assert torch.isfinite(encoder(torch.tensor([[1,2,0]]))).all()


def test_original_catapro_unchanged_after_transfer():
    torch.manual_seed(7)
    baseline=NewKcat(ModelConfig(atom_dim=1,bond_dim=1,mode="baseline"))
    extension=PfamCataPro(6,4,False,dropout=0)
    extension.initialize_catapro(baseline.state_dict())
    baseline.eval();extension.eval()
    batch={"protein":torch.randn(3,1024),"molt5":torch.randn(3,768),
           "maccs":torch.zeros(3,167),
           "pfam_ids":torch.tensor([[2,3,0,0],[0,0,0,0],[4,0,0,0]])}
    with torch.no_grad():
        features=torch.cat((baseline.prot_norm(batch["protein"]),
                            baseline.molt5_norm(batch["molt5"]),batch["maccs"]),dim=-1)
        expected=baseline.out(baseline.decoder(features)).squeeze(-1)
        actual=extension(batch)
    assert torch.allclose(actual,expected,atol=1e-6)
