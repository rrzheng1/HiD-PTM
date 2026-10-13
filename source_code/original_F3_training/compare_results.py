import argparse
import numpy as np
import pandas as pd
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    frames=[];stops=[];reference={}
    for exp in VARIANTS:
        root=HERE/'results'/exp/f'seed_{args.seed}'
        for fold in range(1,6):
            done=load(root/f'fold_{fold}/completed.json');config=done['config']
            require(config['implementation']==code_signature(),'Training implementation changed')
            controls={k:v for k,v in config.items() if k!='variant'}
            require(fold not in reference or controls==reference[fold],'Different features/data/training protocol')
            reference[fold]=controls
            for name in ['metrics.csv','history.csv','validation_predictions.csv','config.json']:
                require(sha(root/f'fold_{fold}'/name)==done['files_sha256'][name],'Changed fold artifact')
            frame=pd.read_csv(root/f'fold_{fold}/metrics.csv');frame['Experiment']=exp;frame['Fold']=fold;frames.append(frame)
            history=pd.read_csv(root/f'fold_{fold}/history.csv')
            stops.append(dict(Experiment=exp,Fold=fold,Epochs=len(history),Best_epoch=done['best_epoch'],Patience=config['patience'],Min_delta=config['min_delta']))
    folds=pd.concat(frames,ignore_index=True);out=HERE/'reports';out.mkdir(exist_ok=True)
    folds.to_csv(out/'all_fold_metrics.csv',index=False);pd.DataFrame(stops).to_csv(out/'early_stopping.csv',index=False)
    mean=folds.groupby(['Experiment','PTM'])[METRICS].mean();std=folds.groupby(['Experiment','PTM'])[METRICS].std(ddof=1)
    mean.to_csv(out/'all_mean.csv');std.to_csv(out/'all_std.csv')
    markdown='# H2输入消融F0–F6\n\n各值为五折均值±样本标准差；pr_auc采用AP。F4联合移除全蛋白表示与依赖它的层次分支，F6停用图传播。\n\n'
    for exp in VARIANTS:
        markdown+=f'## {exp}\n\n| PTM | '+' | '.join(METRICS)+' |\n|---|'+'---|'*7+'\n'
        for ptm in PTMS+['Overall']:
            markdown+='| '+ptm+' | '+' | '.join(f'{mean.loc[(exp,ptm),m]:.4f} ± {std.loc[(exp,ptm),m]:.4f}' for m in METRICS)+' |\n'
        markdown+='\n'
    (out/'F0_F6_results.md').write_text(markdown,encoding='utf-8')
    rows=[]
    for ptm in PTMS+['Overall']:
        paired=folds[folds.PTM.eq(ptm)].pivot(index='Fold',columns='Experiment',values='pr_auc')
        for exp in VARIANTS[1:]:
            delta=paired[exp]-paired.F0
            rows.append(dict(Experiment=exp,PTM=ptm,Delta_AP_vs_F0=float(delta.mean()),Delta_std=float(delta.std(ddof=1)),Positive_folds=int((delta>0).sum())))
    pd.DataFrame(rows).to_csv(out/'paired_AP_differences.csv',index=False)
    print(mean.xs('Overall',level='PTM').to_string());print('Saved reports/F0_F6_results.md')

if __name__=='__main__':main()
