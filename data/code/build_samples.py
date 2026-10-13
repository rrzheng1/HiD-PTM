"""Fixed site-conditioned uniform unlabeled sampling, independent of model scores."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
SOURCE = OUT.parent / 'ptmd_site_5ptm_mmseqs30_v1'
SEED = 20261010
SITE_COLS = ['UniProt','Residue','Position','Sequence_position','Sequence_version','Sequence_length','Site_ID','PTM_standard','Site_window_31aa','Cluster_ID','Global_split','Fold']
COLS = ['PDA_key'] + SITE_COLS + ['Disease_ID','Disease_standard','Evidence_level','Label','Sample_origin']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def save(data, name):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path,index=False,encoding='utf-8-sig')

def main():
    require(not (OUT / 'metadata.json').exists(), 'Release already exists; preserve the fixed samples.')
    meta = json.loads((SOURCE / 'metadata.json').read_text(encoding='utf-8'))
    for name in ['associations.csv','known_positive_exclusion_all11.csv','diseases.csv','protein_split.csv','proteins.fasta']:
        require(sha(SOURCE / name) == meta['files_sha256'][name], f'Source hash mismatch: {name}')
    positives = read(SOURCE / 'associations.csv')
    known = read(SOURCE / 'known_positive_exclusion_all11.csv')
    disease_names = read(SOURCE / 'diseases.csv')
    require(disease_names.groupby('Disease_ID').Disease_standard.nunique().le(1).all(), 'Ambiguous disease names')
    names = disease_names.drop_duplicates('Disease_ID').set_index('Disease_ID').Disease_standard.to_dict()
    diseases = np.array(sorted(names))
    exclusions = defaultdict(set)
    for row in known[['Site_ID','PTM_standard','Disease_ID']].itertuples(index=False) if 'Site_ID' in known else []:
        exclusions[(row.Site_ID,row.PTM_standard)].add(row.Disease_ID)
    # Canonical exclusion table stores coordinates rather than Site_ID.
    if 'Site_ID' not in known:
        for row in known[['UniProt','Residue','Sequence_position','PTM_standard','Disease_ID']].itertuples(index=False):
            exclusions[(f'{row.UniProt}:{row.Residue}{row.Sequence_position}',row.PTM_standard)].add(row.Disease_ID)
    rng = np.random.default_rng(SEED)
    negative_rows = []
    for (site,ptm), group in positives.groupby(['Site_ID','PTM_standard'],sort=True):
        n = len(group)
        blocked = exclusions[(site,ptm)]
        require(n <= len(diseases)-len(blocked & set(names)), f'Insufficient unique unlabeled diseases: {site}/{ptm}')
        selected = set()
        ordered = []
        # Rejection sampling is uniform over allowed diseases, without replacement.
        while len(ordered) < n:
            for disease in rng.choice(diseases,size=max(16,2*(n-len(ordered))),replace=True):
                if disease not in blocked and disease not in selected:
                    selected.add(disease)
                    ordered.append(disease)
                    if len(ordered) == n:
                        break
        base = group.iloc[0][SITE_COLS].to_dict()
        for disease in ordered:
            negative_rows.append(dict(base,PDA_key=f'{site}|{ptm}|{disease}',Disease_ID=disease,Disease_standard=names[disease],Evidence_level='Unlabeled',Label=0,Sample_origin='uniform_unlabeled'))
    neg = pd.DataFrame(negative_rows,columns=COLS)
    pos = positives[SITE_COLS + ['PDA_key','Disease_ID','Disease_standard','Evidence_level']].copy()
    pos['Label'] = 1
    pos['Sample_origin'] = 'known_positive'
    samples = pd.concat([pos[COLS],neg],ignore_index=True)
    samples = samples.iloc[rng.permutation(len(samples))].reset_index(drop=True)
    checks = {}
    def check(name,condition):
        checks[name] = bool(condition)
        require(condition, name)
    check('one_negative_per_positive',len(neg)==len(pos))
    check('no_duplicate_sample_keys',not samples.PDA_key.duplicated().any())
    check('all_gold_and_silver_positive_keys_retained',set(pos.PDA_key)==set(positives.PDA_key))
    check('no_negative_known_positive_overlap',not set(neg.PDA_key)&set(known.PDA_key))
    counts = samples.groupby(['Site_ID','PTM_standard','Label']).size().unstack(fill_value=0)
    check('site_ptm_counts_matched',counts[0].equals(counts[1]))
    dev = samples[samples.Global_split.eq('dev')]
    test = samples[samples.Global_split.eq('test')]
    for col in ['UniProt','Cluster_ID','Site_ID','PDA_key']:
        check('dev_test_disjoint_'+col,not set(dev[col])&set(test[col]))
    save(samples,'samples_1to1.csv')
    save(dev,'development_1to1.csv')
    save(test,'independent_test_1to1.csv')
    stats = []
    parts = [('all',samples),('development',dev),('independent_test',test)]
    for k in range(1,6):
        train = dev[dev.Fold.ne(str(k))]
        val = dev[dev.Fold.eq(str(k))]
        for col in ['UniProt','Cluster_ID','Site_ID','PDA_key']:
            check(f'fold_{k}_disjoint_{col}',not set(train[col])&set(val[col]))
        save(train,f'folds/fold_{k}/train_1to1.csv')
        save(val,f'folds/fold_{k}/validation_1to1.csv')
        parts.extend([(f'fold_{k}_train',train),(f'fold_{k}_validation',val)])
    for name,frame in parts:
        for ptm,group in [('ALL',frame)]+list(frame.groupby('PTM_standard',sort=True)):
            p = int(group.Label.eq(1).sum())
            n = int(group.Label.eq(0).sum())
            check(f'balanced_{name}_{ptm}',p==n)
            stats.append(dict(partition=name,PTM=ptm,positive=p,negative=n,total=len(group),gold=int(group.Evidence_level.eq('Gold').sum()),silver=int(group.Evidence_level.eq('Silver').sum()),proteins=group.UniProt.nunique(),clusters=group.Cluster_ID.nunique()))
    save(pd.DataFrame(stats),'sample_statistics.csv')
    save(pd.DataFrame([dict(check=k,passed=v) for k,v in checks.items()]),'sampling_audit.csv')
    for name in ['proteins.fasta','sites.csv','diseases.csv','protein_split.csv','cluster_mapping.csv','known_positive_exclusion_all11.csv']:
        shutil.copy2(SOURCE / name,OUT / name)
    result = dict(version='ptmd-site-5ptm-mmseqs30-1to1-v1',source=str(SOURCE),source_metadata_sha256=sha(SOURCE/'metadata.json'),seed=SEED,negative_ratio=1,sampling_unit='Site_ID x PTM',sampling_method='uniform disease without replacement per site-PTM; same site-PTM counts as positives',candidate_disease_catalog='fixed curated disease vocabulary from source diseases.csv; no test disease frequencies or model scores',negative_definition='unobserved associations, not experimentally confirmed negatives',known_positive_exclusion='all11 canonical Gold+Silver associations across every split',development_positive=len(dev)//2,independent_test_positive=len(test)//2,positive=len(pos),negative=len(neg),total=len(samples),resample_per_fold=False,label_column='Label',features='sequence and coordinate/PTM/disease identifiers; evidence and Sample_origin are not model inputs',audit_all_passed=all(checks.values()))
    result['files_sha256']={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    (OUT/'metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(stats).query("PTM == 'ALL'").to_string(index=False))
    print(f'PASSED: positives={len(pos)}, negatives={len(neg)}; checks={len(checks)}')

if __name__=='__main__':
    main()
