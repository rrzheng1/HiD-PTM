import argparse
import shutil
import torch
from common import *
from model import SiteDiseaseModel,compact_f3_state_dict

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--patience',type=int,default=5);p.add_argument('--min-delta',type=float,default=.0001);args=p.parse_args()
    source=HERE.parent/'h2_feature_ablation_v1/results/F3'/f'seed_{args.seed}'
    target=HERE/'results/L0'/f'seed_{args.seed}'
    require(source.exists(),'F3 source missing; run L0 individually to train it instead')
    for fold in range(1,6):
        done=load(source/f'fold_{fold}/completed.json');c=done['config']
        for k in ['seed','batch_size','epochs','patience','min_delta']:require(c[k]==getattr(args,k),'F3 settings differ: '+k)
        require(c['lr']==.0003 and c['threshold']==.5,'F3 training protocol differs')
        require(c['train_sha256']==sha(DATA/f'folds/fold_{fold}/train_1to1.csv') and c['validation_sha256']==sha(DATA/f'folds/fold_{fold}/validation_1to1.csv'),'F3 data changed')
        for name,h in done['files_sha256'].items():require(sha(source/f'fold_{fold}'/name)==h,'Source F3 file changed')
    if not target.exists():shutil.copytree(source,target)
    original={}
    for path in source.rglob('*'):
        if path.is_file():
            name=path.relative_to(source).as_posix();require((target/name).exists() and sha(target/name)==sha(path),'L0 copy differs from F3')
            original[name]=sha(path)
    provenance=target/'reused_from_F3.json';prior=load(provenance) if provenance.exists() else None;compact={}
    for fold in range(1,6):
        filename=f'fold_{fold}/compact_best.pt';path=target/filename
        if prior is not None:
            require(sha(path)==prior['compact_checkpoint_sha256'][filename],'Compact checkpoint changed')
        else:
            state=torch.load(source/f'fold_{fold}/best.pt',map_location='cpu',weights_only=False)
            text=state['model_state_dict']['text'];blosum=state['model_state_dict']['blosum']
            adjacency=state['model_state_dict']['adjacency'].coalesce()
            model=SiteDiseaseModel(text,adjacency.indices(),blosum,variant='L0')
            compact_state=compact_f3_state_dict(state['model_state_dict']);model.load_state_dict(compact_state,strict=True)
            torch.save(dict(model_state_dict=compact_state,variant='L0',architecture='H2-Lite',threshold=.5,
                            source_config=state['config'],source_checkpoint_sha256=sha(source/f'fold_{fold}/best.pt'),
                            epoch=state['epoch'],removed_keys=['blosum','local.*','local_projection.*']),path)
        compact[filename]=sha(path)
    record=dict(source=str(source.resolve()),retrained=False,files_sha256=original,compact_checkpoint_sha256=compact,
                conversion='Only discarded unused F3 descriptor parameters; active weights unchanged. Original raw checkpoints also retained.')
    if prior is not None:require(prior==record,'Reuse provenance changed')
    else:dump(provenance,record)
    print('L0 reused F3; compact_best.pt saved with descriptor CNN physically removed.')

if __name__=='__main__':main()
