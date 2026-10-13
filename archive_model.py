"""Freeze HiD-PTM artifacts; retain exact source configurations and hashes."""
from pathlib import Path
import shutil
import hashlib
import json

ROOT=Path(__file__).resolve().parent
EXPERIMENT=ROOT.parent
LITE=EXPERIMENT/'h2_lite_validation_v1'
SOURCE=LITE/'results/L0/seed_42'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target):raise ValueError(f'Existing archive differs: {target}')
    else:shutil.copy2(source,target)

def tree(source,target):
    for path in source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:copy_file(path,target/path.relative_to(source))

def main():
    reuse=json.loads((SOURCE/'reused_from_F3.json').read_text(encoding='utf-8'))
    for name,expected in {**reuse['files_sha256'],**reuse['compact_checkpoint_sha256']}.items():
        if sha(SOURCE/name)!=expected:raise ValueError(f'Source artifact changed: {name}')
    tree(SOURCE,ROOT/'results/cross_validation/seed_42')
    for fold in range(1,6):copy_file(SOURCE/f'fold_{fold}/compact_best.pt',ROOT/f'models/fold_{fold}/best.pt')
    for name in ['model.py','h2_reference.py','baseline.py']:
        copy_file(LITE/name,ROOT/name)
    for path in LITE.glob('*.py'):copy_file(path,ROOT/'source_code/lite_experiment'/path.name)
    for name in ['README.md','run_all.ps1','requirements.txt']:copy_file(LITE/name,ROOT/'source_code/lite_experiment'/name)
    original=EXPERIMENT/'h2_feature_ablation_v1'
    for path in original.glob('*.py'):copy_file(path,ROOT/'source_code/original_F3_training'/path.name)
    for name in ['run_all.ps1','requirements.txt']:copy_file(original/name,ROOT/'source_code/original_F3_training'/name)
    data=EXPERIMENT/'data/ptmd_site_5ptm_mmseqs30_1to1_v1'
    for name in ['metadata.json','protein_split.csv','cluster_mapping.csv','diseases.csv','sites.csv','proteins.fasta']:
        copy_file(data/name,ROOT/'metadata/dataset'/name)
    cache=EXPERIMENT/'site_disease_cross_attention_v1/cache'
    for name in ['manifest.json','disease_nodes.csv','text_provenance.json','embedding_provenance.csv']:
        copy_file(cache/name,ROOT/'metadata/features'/name)
    record=dict(model_name='HiD-PTM',expanded_name='Hierarchical Disease-conditioned PTM Association Model',
                architecture='L0 / H2-Lite / F3-equivalent',source=str(SOURCE),seed=42,
                checkpoint_count=5,full_development_retrained_model=False,independent_test_evaluated=False,
                threshold=.5,early_stopping=dict(patience=5,min_delta=.0001,monitor='validation AP'),
                dataset=str(data),feature_cache=str(cache),esm_directory=str(EXPERIMENT/'embedding/esm2_embedding'),
                prediction_inputs='31aa ESM, protein mean ESM, valid-position mask, PTM type, disease index',
                archive_kind='five-fold best models and their original results; no retraining or relabeling',
                files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*')
                              if p.is_file() and p.name!='archive_manifest.json' and '__pycache__' not in p.parts})
    (ROOT/'archive_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('HiD-PTM archive complete:',ROOT)

if __name__=='__main__':main()
