"""One fixed protein-cluster CV fold, with atomic best/last checkpoints."""
import argparse
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from common import *
from model import SiteDiseaseModel
from early_stopping import EarlyStopping

class Features:
    def __init__(self,cache,variant='L0'):
        self.cache=cache;self.variant=variant
        self.arrays={name:np.load(cache/(name+'.npy'),mmap_mode='r') for name in ['esm_windows','protein_mean','mask','site_protein']}
        self.site_index={s:i for i,s in enumerate(read(cache/'sites.csv').Site_ID)}
        self.disease_index={s:i for i,s in enumerate(read(cache/'disease_nodes.csv').Disease_ID)}

    def encode(self,frame):
        return dict(site=np.array([self.site_index[s] for s in frame.Site_ID],dtype=np.int64),
                    ptm=np.array([PTMS.index(p) for p in frame.PTM_standard],dtype=np.int64),
                    disease=np.array([self.disease_index[d] for d in frame.Disease_ID],dtype=np.int64),
                    label=frame.Label.to_numpy(dtype=np.float32))

    def batch(self,encoded,indices,device,site_override=None):
        sites=encoded['site'][indices] if site_override is None else site_override[indices]
        protein=self.arrays['site_protein'][sites]
        def tensor(array):return torch.from_numpy(np.array(array,copy=True)).to(device)
        return [tensor(self.arrays['esm_windows'][sites]),
                None if self.variant=='L2' else tensor(self.arrays['protein_mean'][protein]),
                None,tensor(self.arrays['mask'][sites]),
                tensor(encoded['ptm'][indices]),tensor(encoded['disease'][indices])]

def atomic_save(value,path):
    temporary=Path(str(path)+'.tmp');torch.save(value,temporary);os.replace(temporary,path)

@torch.no_grad()
def predict(model,features,encoded,device,batch_size,site_override=None):
    model.eval();table=model.disease_table();result=[]
    for start in range(0,len(encoded['label']),batch_size):
        idx=np.arange(start,min(start+batch_size,len(encoded['label'])))
        probability=torch.sigmoid(model(*features.batch(encoded,idx,device,site_override),disease_table=table))
        result.append(probability.cpu().numpy())
    return np.concatenate(result)

def metric_rows(frame,probability):
    rows=[]
    for ptm in PTMS+['Overall']:
        use=np.ones(len(frame),dtype=bool) if ptm=='Overall' else frame.PTM_standard.eq(ptm).to_numpy()
        require(use.any(),f'No validation samples for {ptm}')
        rows.append(dict(PTM=ptm,**scores(frame.Label.to_numpy()[use],probability[use])))
    return pd.DataFrame(rows)

def diagnostics(train,val,probability):
    counts=train.assign(y=train.Label.astype(int)).groupby('Disease_ID').y.agg(['sum','count'])
    prior=(counts['sum']+1)/(counts['count']+2)
    baseline=val.Disease_ID.map(prior).fillna(.5).to_numpy()
    per=[]
    for ptm in PTMS+['Overall']:
        use=np.ones(len(val),dtype=bool) if ptm=='Overall' else val.PTM_standard.eq(ptm).to_numpy()
        subset=val[use].copy();subset['Probability']=probability[use]
        auc=[];covered=0
        for _,group in subset.groupby('Disease_ID'):
            y=group.Label.astype(int)
            if y.sum()>=5 and (1-y).sum()>=5:
                auc.append(scores(y,group.Probability)['roc_auc']);covered+=len(group)
        per.append(dict(PTM=ptm,Within_disease_mean_AUC=float(np.mean(auc)) if auc else None,
                        Eligible_disease_count=len(auc),Sample_coverage=covered/len(subset)))
    return metric_rows(val,baseline),pd.DataFrame(per)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,required=True,choices=range(1,6))
    parser.add_argument('--device',default='cuda');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=40);parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--min-delta',type=float,default=.0001)
    parser.add_argument('--batch-size',type=int,default=128);parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--variant',choices=VARIANTS,default='L0')
    args=parser.parse_args()
    require(args.epochs>0 and args.patience>0 and args.batch_size>0,'Positive epochs/patience/batch size required')
    torch.set_num_threads(4);device=torch.device(args.device)
    require(device.type!='cuda' or torch.cuda.is_available(),'CUDA not available')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    cache=feature_cache_path();require((cache/'manifest.json').exists(),'Run prepare.py first')
    manifest=load(cache/'manifest.json')
    require(manifest['signature']['development']==sha(DATA/'development_1to1.csv'),'Development dataset changed')
    require(manifest['signature']['data_metadata']==sha(DATA/'metadata.json'),'Dataset metadata changed')
    for name,h in manifest['files_sha256'].items():require(sha(cache/name)==h,f'Corrupted cache: {name}')
    train,val=fold_data(args.fold)
    config=dict(vars(args),data=str(DATA),threshold=.5,pr_auc_definition='sklearn average_precision_score',
                train_sha256=sha(DATA/f'folds/fold_{args.fold}/train_1to1.csv'),
                validation_sha256=sha(DATA/f'folds/fold_{args.fold}/validation_1to1.csv'),
                feature_manifest_sha256=sha(cache/'manifest.json'),implementation=code_signature())
    out=HERE/'results'/args.variant/f'seed_{args.seed}'/f'fold_{args.fold}';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        done=load(out/'completed.json');require(done['config']==config,'Completed run configuration changed; use another seed/experiment folder')
        for name,h in done['files_sha256'].items():require(sha(out/name)==h,f'Corrupted completed run: {name}')
        print(f'Complete and verified; skipped: {out}');return
    if (out/'config.json').exists():require(load(out/'config.json')==config,'Partial run configuration changed')
    dump(out/'config.json',config)
    features=Features(cache,args.variant);tr,va=features.encode(train),features.encode(val)
    model=SiteDiseaseModel(torch.from_numpy(np.load(cache/'disease_text.npy')),torch.from_numpy(np.load(cache/'ontology_edges.npy')),
                           variant=args.variant).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    stopping=EarlyStopping(args.patience,args.min_delta)
    criterion=nn.BCEWithLogitsLoss();best=-float('inf');best_epoch=0;start=0;history=[]
    last=out/'last.pt'
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False) # Locally generated trusted checkpoint.
        require(state['config']==config,'Checkpoint configuration changed')
        model.load_state_dict(state['model_state_dict']);optimizer.load_state_dict(state['optimizer_state_dict'])
        start=state['epoch'];best=state['best_validation_pr_auc'];best_epoch=state['best_epoch'];history=state['history']
        stopping.load_state_dict(state['early_stopping'])
        torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'Resumed after epoch {start}',flush=True)
    stop=start if stopping.wait>=args.patience else args.epochs
    for epoch in range(start+1,stop+1):
        begin=time.time();model.train();total=0
        order=np.random.default_rng(args.seed+epoch*1000).permutation(len(train))
        for offset in range(0,len(order),args.batch_size):
            idx=order[offset:offset+args.batch_size];optimizer.zero_grad(set_to_none=True)
            logits=model(*features.batch(tr,idx,device));target=torch.from_numpy(tr['label'][idx]).to(device)
            loss=criterion(logits,target);require(torch.isfinite(loss).item(),'Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=loss.item()*len(idx)
        probability=predict(model,features,va,device,args.batch_size)
        validation=scores(va['label'],probability)
        require(validation['pr_auc'] is not None,'Validation must contain both labels')
        row=dict(epoch=epoch,loss=total/len(train),seconds=time.time()-begin,**validation);history.append(row)
        if validation['pr_auc']>best:
            best=validation['pr_auc'];best_epoch=epoch
            atomic_save(dict(model_state_dict=model.state_dict(),config=config,epoch=epoch,
                             validation_pr_auc=best,threshold=.5),out/'best.pt')
        should_stop=stopping.update(validation['pr_auc'])
        row['early_stop_wait']=stopping.wait
        atomic_save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=config,
                         epoch=epoch,best_epoch=best_epoch,best_validation_pr_auc=best,history=history,early_stopping=stopping.state_dict(),
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None),last)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f'fold={args.fold} epoch={epoch} loss={row["loss"]:.5f} validation_pr_auc={best:.5f} seconds={row["seconds"]:.1f}',flush=True)
        if should_stop:break
    state=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict'])
    probability=predict(model,features,va,device,args.batch_size)
    prediction=val[['PDA_key','Site_ID','UniProt','Cluster_ID','PTM_standard','Disease_ID','Label','Fold']].copy()
    prediction['Probability']=probability;prediction.to_csv(out/'validation_predictions.csv',index=False)
    metric_rows(val,probability).to_csv(out/'metrics.csv',index=False)
    baseline,within=diagnostics(train,val,probability)
    baseline.to_csv(out/'disease_only_metrics.csv',index=False);within.to_csv(out/'within_disease_auc.csv',index=False)
    shuffled=va['site'].copy();rng=np.random.default_rng(args.seed+9000+args.fold)
    for ptm in range(5):
        indices=np.flatnonzero(va['ptm']==ptm);shuffled[indices]=rng.permutation(shuffled[indices])
    shuffled_probability=predict(model,features,va,device,args.batch_size,site_override=shuffled)
    metric_rows(val,shuffled_probability).to_csv(out/'site_shuffled_metrics.csv',index=False)
    outputs=['config.json','best.pt','last.pt','history.csv','validation_predictions.csv','metrics.csv',
             'disease_only_metrics.csv','within_disease_auc.csv','site_shuffled_metrics.csv']
    dump(out/'completed.json',dict(config=config,best_epoch=best_epoch,files_sha256={name:sha(out/name) for name in outputs}))
    print(f'Saved model and predictions: {out}',flush=True)

if __name__=='__main__':main()
