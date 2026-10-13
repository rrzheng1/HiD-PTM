import argparse
import torch
from common import VARIANTS
from h2_reference import SiteDiseaseModel as H2
from model import SiteDiseaseModel
from early_stopping import EarlyStopping

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');args=p.parse_args();device=args.device
    torch.set_num_threads(4);text=torch.randn(8,768);edges=torch.stack([torch.arange(8),torch.arange(8)]);blosum=torch.randn(21,20)
    batch=[torch.randn(5,31,1280,device=device),torch.randn(5,1280,device=device),torch.randint(0,21,(5,31),device=device),
           torch.ones(5,31,dtype=torch.bool,device=device),torch.arange(5,device=device),torch.arange(5,device=device)]
    torch.manual_seed(42);h2=H2(text,edges,blosum,variant='H2').to(device).eval()
    torch.manual_seed(42);f0=SiteDiseaseModel(text,edges,blosum,variant='F0').to(device).eval()
    with torch.no_grad():assert torch.allclose(h2(*batch),f0(*batch),atol=1e-6),'F0 must equal H2'
    for variant in VARIANTS:
        model=SiteDiseaseModel(text,edges,blosum,variant=variant).to(device)
        inputs=list(batch)
        if variant=='F4':inputs[1]=None
        if variant=='F5':inputs[0]=None
        output=model(*inputs);output.square().mean().backward();assert torch.isfinite(output).all()
        if variant=='F1':assert (model.local_features(batch[2],batch[3])[...,:21]==0).all()
        if variant=='F2':assert (model.local_features(batch[2],batch[3])[...,21:]==0).all()
        if variant=='F3':assert model.local[0].weight.grad is None and model.local_projection.weight.grad is None
        if variant=='F4':assert model.global_projection.weight.grad is None and model.protein_disease_gate.weight.grad is None
        if variant=='F5':assert model.esm_projection.weight.grad is None
        if variant=='F6':assert model.gcn[0].weight.grad is None and model.text_projection.weight.grad is not None
        model.eval()
        with torch.no_grad():
            original=model(*batch)
            changed=list(batch)
            if variant=='F4':changed[1]=torch.full_like(batch[1],float('nan'))
            if variant=='F5':changed[0]=torch.full_like(batch[0],float('nan'))
            if variant=='F3':changed[2]=(batch[2]+1)%21
            if variant in ['F3','F4','F5']:assert torch.allclose(original,model(*changed),atol=1e-6),'Removed input affects output'
    early=EarlyStopping(2,.001);assert not early.update(.7) and not early.update(.7001)
    other=EarlyStopping(2,.001);other.load_state_dict(early.state_dict());assert other.update(.7002)
    print('PASSED: F0/H2 equivalence; F0-F6 gradients, excluded-input invariance, channel masks, early-stop restore.')

if __name__=='__main__':main()
