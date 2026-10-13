from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DATA = HERE.parent/'data/ptmd_site_5ptm_mmseqs30_1to1_v1'
EMBED = HERE.parent/'embedding/esm2_embedding'
MONDO = PROJECT/'PTMD-data/PTMD_site_dataset_v1/reference/mondo.obo'
TEXT_MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
VARIANTS = ['L0','L1','L2','L3']
PTMS = ['Acetylation','Methylation','Phosphorylation','SUMOylation','Ubiquitination']
METRICS = ['accuracy','precision','recall','f1','mcc','roc_auc','pr_auc']
AA = 'ACDEFGHIKLMNPQRSTVWYX'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def scores(y, probability):
    y=np.asarray(y,dtype=int); p=np.asarray(probability)
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probabilities')
    prediction=p>=.5
    return dict(accuracy=float(accuracy_score(y,prediction)),precision=float(precision_score(y,prediction,zero_division=0)),
                recall=float(recall_score(y,prediction,zero_division=0)),f1=float(f1_score(y,prediction,zero_division=0)),
                mcc=float(matthews_corrcoef(y,prediction)),roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
                pr_auc=float(average_precision_score(y,p)) if len(set(y))==2 else None)

def code_signature():
    return {name:sha(HERE/name) for name in ['common.py','prepare.py','baseline.py','h2_reference.py','model.py','early_stopping.py','train.py']}

def feature_cache_path():
    ref=HERE/'cache/cache_reference.json'
    if ref.exists():
        item=load(ref);path=Path(item['path'])
        require(sha(path/'manifest.json')==item['manifest_sha256'],'Shared feature manifest changed')
        return path
    return HERE/'cache'

def fasta(path):
    result={}; key=None; seq=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('>'):
            if key is not None:
                result[key]=''.join(seq)
            key=line[1:].split()[0]; seq=[]
        else:
            seq.append(line.strip())
    if key is not None:
        result[key]=''.join(seq)
    return result

def fold_data(fold):
    tr=read(DATA/f'folds/fold_{fold}/train_1to1.csv')
    va=read(DATA/f'folds/fold_{fold}/validation_1to1.csv')
    for col in ['UniProt','Cluster_ID','Site_ID','PDA_key']:
        require(not set(tr[col]) & set(va[col]),f'Fold leakage: {col}')
    require(set(va.Fold)=={str(fold)} and str(fold) not in set(tr.Fold),'Wrong fold assignment')
    require(not tr.PDA_key.duplicated().any() and not va.PDA_key.duplicated().any(),'Duplicate keys')
    return tr,va
