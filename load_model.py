from pathlib import Path
import torch
from model import SiteDiseaseModel

ROOT=Path(__file__).resolve().parent

def load_model(fold=1,device='cpu'):
    if fold not in range(1,6):raise ValueError('fold must be 1..5')
    checkpoint=torch.load(ROOT/f'models/fold_{fold}/best.pt',map_location='cpu',weights_only=False)
    state=checkpoint['model_state_dict']
    model=SiteDiseaseModel(state['text'],state['adjacency'].coalesce().indices(),variant='L0')
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()

if __name__=='__main__':
    model=load_model()
    print('HiD-PTM fold 1 loaded; parameters:',sum(p.numel() for p in model.parameters()))
