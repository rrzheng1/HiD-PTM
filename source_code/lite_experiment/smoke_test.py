import argparse
import torch
from common import VARIANTS
from model import SiteDiseaseModel,compact_f3_state_dict
from h2_reference import SiteDiseaseModel as H2

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');args=p.parse_args();dev=args.device;torch.set_num_threads(4)
    text=torch.randn(8,768);edges=torch.stack([torch.arange(8),torch.arange(8)]);blosum=torch.randn(21,20)
    batch=[torch.randn(5,31,1280,device=dev),torch.randn(5,1280,device=dev),torch.randint(0,21,(5,31),device=dev),torch.ones(5,31,dtype=torch.bool,device=dev),torch.arange(5,device=dev),torch.arange(5,device=dev)]
    torch.manual_seed(42);reference=H2(text,edges,blosum,variant='H2').to(dev).eval()
    reference.local_projection.weight.data.zero_();reference.local_projection.bias.data.zero_()
    torch.manual_seed(42);lite=SiteDiseaseModel(text,edges,blosum,variant='L0').to(dev).eval()
    with torch.no_grad():assert torch.allclose(reference(*batch),lite(*batch),atol=1e-6),'Lite must equal descriptor-disabled H2'
    lite.load_state_dict(compact_f3_state_dict(reference.state_dict()),strict=True)
    for variant in VARIANTS:
        model=SiteDiseaseModel(text,edges,blosum,variant=variant).to(dev)
        assert not hasattr(model,'local') and not hasattr(model,'blosum') and not hasattr(model,'local_projection')
        inputs=list(batch);inputs[2]=None
        if variant=='L2':inputs[1]=None
        logits=model(*inputs);logits.square().mean().backward();assert torch.isfinite(logits).all()
        if variant in ['L1','L2']:assert not hasattr(model,'protein_disease_gate')
        if variant=='L2':assert not hasattr(model,'global_projection')
        if variant=='L3':assert not hasattr(model,'gcn')
        else:assert model.gcn[0].weight.grad is not None
        if variant in ['L0','L3']:assert model.protein_disease_gate.weight.grad.abs().sum()>0
        model.eval()
        with torch.no_grad():
            original=model(*batch);changed=list(batch);changed[2]=(batch[2]+1)%21
            if variant=='L2':changed[1]=torch.full_like(batch[1],float('nan'))
            assert torch.allclose(original,model(*changed),atol=1e-6),'Removed features affect prediction'
    print('PASSED: compact F3-equivalent path, strict checkpoint conversion, physical module removal, L0-L3 gradients and excluded-input invariance.')

if __name__=='__main__':main()
