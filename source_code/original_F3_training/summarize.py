import argparse
import numpy as np
import pandas as pd
from common import *
from train import metric_rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--variant',choices=VARIANTS,default='F0');args=parser.parse_args()
    out=HERE/'results'/args.variant/f'seed_{args.seed}';rows=[];all_predictions=[];signature=None
    for fold in range(1,6):
        path=out/f'fold_{fold}';require((path/'completed.json').exists(),f'Incomplete fold {fold}')
        completed=load(path/'completed.json')
        for name,h in completed['files_sha256'].items():require(sha(path/name)==h,f'Corrupted fold {fold}: {name}')
        config=completed['config'];common={k:v for k,v in config.items() if k not in ['fold','train_sha256','validation_sha256']}
        require(signature is None or signature==common,'Mixed training configurations');signature=common
        require(config['implementation']==code_signature(),'Training code changed since this result')
        pred=read(path/'validation_predictions.csv');source=read(DATA/f'folds/fold_{fold}/validation_1to1.csv')
        cols=['PDA_key','Site_ID','UniProt','Cluster_ID','PTM_standard','Disease_ID','Label','Fold']
        require(pred[cols].equals(source[cols]),f'Validation samples differ in fold {fold}')
        require(config['validation_sha256']==sha(DATA/f'folds/fold_{fold}/validation_1to1.csv'),'Source changed')
        probability=pred.Probability.to_numpy(dtype=float)
        metrics=metric_rows(pred,probability);metrics['Fold']=fold;rows.append(metrics)
        all_predictions.append(pred)
    folds=pd.concat(rows,ignore_index=True);folds.to_csv(out/'fold_metrics.csv',index=False)
    pred=pd.concat(all_predictions,ignore_index=True);dev=read(DATA/'development_1to1.csv')
    require(not pred.PDA_key.duplicated().any() and set(pred.PDA_key)==set(dev.PDA_key),'OOF coverage failed')
    pred.to_csv(out/'oof_predictions.csv',index=False)
    mean=folds.groupby('PTM')[METRICS].mean().reindex(PTMS+['Overall'])
    std=folds.groupby('PTM')[METRICS].std(ddof=1).reindex(PTMS+['Overall'])
    mean.to_csv(out/'fivefold_mean.csv');std.to_csv(out/'fivefold_std.csv')
    table=mean.copy().astype(object)
    for column in METRICS:
        table[column]=[f'{m:.4f} ± {s:.4f}' for m,s in zip(mean[column],std[column])]
    table.to_csv(out/'fivefold_mean_std.csv')
    markdown='| PTM | '+' | '.join(METRICS)+' |\n|---|'+'---|'*len(METRICS)+'\n'
    for ptm,row in table.iterrows():markdown+='| '+ptm+' | '+' | '.join(row)+' |\n'
    (out/'fivefold_results.md').write_text(markdown,encoding='utf-8')
    metric_rows(pred,pred.Probability.to_numpy(dtype=float)).to_csv(out/'pooled_oof_metrics.csv',index=False)
    print(table.to_string());print(f'Five-fold summary: {out}',flush=True)

if __name__=='__main__':main()
