from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def require(condition,message):
    if not condition:
        raise ValueError(message)

def main():
    meta=json.loads((ROOT/'metadata.json').read_text(encoding='utf-8'))
    for name,expected in meta['files_sha256'].items():
        h=hashlib.sha256()
        with (ROOT/name).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):
                h.update(block)
        require(h.hexdigest()==expected,f'Hash mismatch: {name}')
    def read(name):
        return pd.read_csv(ROOT/name,dtype=str,keep_default_na=False)
    data=read('samples_1to1.csv')
    dev=read('development_1to1.csv')
    test=read('independent_test_1to1.csv')
    known=read('known_positive_exclusion_all11.csv')
    require(len(data)==meta['total'],'Wrong total')
    require(set(data.Label)=={'0','1'},'Wrong labels')
    require(not data.PDA_key.duplicated().any(),'Duplicate sample keys')
    pos=data[data.Label.eq('1')]
    neg=data[data.Label.eq('0')]
    source=read(Path(meta['source'])/'associations.csv')
    require(set(pos.PDA_key)==set(source.PDA_key),'Positive source changed')
    require(not set(neg.PDA_key)&set(known.PDA_key),'Negative overlaps known positive')
    require((data.PDA_key==data.Site_ID+'|'+data.PTM_standard+'|'+data.Disease_ID).all(),'Key mismatch')
    assignments=read('protein_split.csv').set_index('UniProt')
    for col in ['Cluster_ID','Global_split','Fold']:
        require(data[col].equals(data.UniProt.map(assignments[col])),f'Assignment changed: {col}')
    counts=data.groupby(['Site_ID','PTM_standard','Label']).size().unstack(fill_value=0)
    require(counts['0'].equals(counts['1']),'Site/PTM balance changed')
    require(set(dev.PDA_key)|set(test.PDA_key)==set(data.PDA_key),'Wrong partition union')
    require(len(dev)+len(test)==len(data),'Wrong partition size')
    def disjoint(a,b):
        for col in ['UniProt','Cluster_ID','Site_ID','PDA_key']:
            require(not set(a[col])&set(b[col]),f'Overlap: {col}')
    def balanced(frame):
        c=frame.groupby(['PTM_standard','Label']).size().unstack(fill_value=0)
        require(c['0'].equals(c['1']),'PTM balance changed')
    disjoint(dev,test)
    balanced(dev)
    balanced(test)
    keys=[]
    for k in range(1,6):
        train=read(f'folds/fold_{k}/train_1to1.csv')
        val=read(f'folds/fold_{k}/validation_1to1.csv')
        disjoint(train,val)
        balanced(train)
        balanced(val)
        require(set(val.Fold)=={str(k)},'Wrong validation fold')
        require(set(train.PDA_key)|set(val.PDA_key)==set(dev.PDA_key),'Wrong fold union')
        require(len(train)+len(val)==len(dev),'Wrong fold size')
        keys.extend(val.PDA_key)
    require(len(keys)==len(dev) and len(set(keys))==len(dev),'Development samples not validated exactly once')
    print(f'PASSED: {len(pos)} positives + {len(neg)} unlabeled negatives; fixed 1:1; 5 folds; no protein/cluster leakage; hashes matched.')

if __name__=='__main__':
    main()
