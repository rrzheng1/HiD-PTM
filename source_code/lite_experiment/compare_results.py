import argparse
import pandas as pd
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);args=p.parse_args();frames=[];controls={};audit=[]
    for exp in VARIANTS:
        root=HERE/'results'/exp/f'seed_{args.seed}'
        if exp=='L0' and (root/'reused_from_F3.json').exists():
            reused=load(root/'reused_from_F3.json')
            for name,h in {**reused['files_sha256'],**reused['compact_checkpoint_sha256']}.items():require(sha(root/name)==h,'Reused L0 artifact changed')
        else:reused=None
        for fold in range(1,6):
            done=load(root/f'fold_{fold}/completed.json');c=done['config']
            if reused is None:require(c['implementation']==code_signature(),'Changed training code')
            comparable={k:c[k] for k in ['seed','batch_size','epochs','patience','min_delta','lr','data','threshold','train_sha256','validation_sha256','feature_manifest_sha256']}
            require(fold not in controls or controls[fold]==comparable,'Unequal controls');controls[fold]=comparable
            for name in ['metrics.csv','validation_predictions.csv','history.csv','config.json']:
                require(sha(root/f'fold_{fold}'/name)==done['files_sha256'][name],'Changed fold result')
            f=pd.read_csv(root/f'fold_{fold}/metrics.csv');f['Experiment']=exp;f['Fold']=fold;frames.append(f)
            hist=pd.read_csv(root/f'fold_{fold}/history.csv');audit.append(dict(Experiment=exp,Fold=fold,Epochs=len(hist),Best_epoch=done['best_epoch'],Reused_F3=reused is not None))
    folds=pd.concat(frames);out=HERE/'reports';out.mkdir(exist_ok=True)
    folds.to_csv(out/'all_fold_metrics.csv',index=False);pd.DataFrame(audit).to_csv(out/'training_audit.csv',index=False)
    mean=folds.groupby(['Experiment','PTM'])[METRICS].mean();std=folds.groupby(['Experiment','PTM'])[METRICS].std(ddof=1)
    mean.to_csv(out/'all_mean.csv');std.to_csv(out/'all_std.csv')
    text='# H2-Lite验证：L0–L3\n\nL0复用F3，其他组同输入划分及训练规则；pr_auc采用AP。\n\n'
    for exp in VARIANTS:
        text+=f'## {exp}\n\n| PTM | '+' | '.join(METRICS)+' |\n|---|'+'---|'*7+'\n'
        for ptm in PTMS+['Overall']:
            text+='| '+ptm+' | '+' | '.join(f'{mean.loc[(exp,ptm),m]:.4f} ± {std.loc[(exp,ptm),m]:.4f}' for m in METRICS)+' |\n'
        text+='\n'
    (out/'L0_L3_results.md').write_text(text,encoding='utf-8')
    differences=[]
    for ptm in PTMS+['Overall']:
        paired=folds[folds.PTM.eq(ptm)].pivot(index='Fold',columns='Experiment',values='pr_auc')
        for exp in VARIANTS[1:]:
            d=paired[exp]-paired.L0
            differences.append(dict(Experiment=exp,PTM=ptm,Mean_delta_AP=float(d.mean()),Std_delta_AP=float(d.std(ddof=1)),Positive_folds=int((d>0).sum())))
    pd.DataFrame(differences).to_csv(out/'paired_AP_differences.csv',index=False)
    print(mean.xs('Overall',level='PTM').to_string())

if __name__=='__main__':main()
