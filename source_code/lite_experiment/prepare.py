"""Label-free feature cache: full-context residue ESM, local descriptors, MONDO/text."""
import argparse
import os
import re
import shutil
import numpy as np
import pandas as pd
import torch
from Bio.Align import substitution_matrices
from tqdm import tqdm
from common import *

def parse_mondo(path):
    terms={}; item=None
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('['):
            if item and item.get('id','').startswith('MONDO:'):
                terms[item['id']]=item
            item={'parents':[],'synonyms':[]} if line=='[Term]' else None
        elif item is not None:
            if line.startswith('id: '): item['id']=line[4:]
            elif line.startswith('name: '): item['name']=line[6:]
            elif line.startswith('def: '):
                match=re.match(r'def: "(.*)" \[',line)
                if match: item['definition']=match.group(1)
            elif line.startswith('is_a: '): item['parents'].append(line[6:].split()[0])
            elif line.startswith('synonym: ') and ' EXACT ' in line:
                match=re.match(r'synonym: "(.*?)"',line)
                if match: item['synonyms'].append(match.group(1))
            elif line=='is_obsolete: true': item['obsolete']=True
    if item and item.get('id','').startswith('MONDO:'): terms[item['id']]=item
    return terms

def disease_features(cache,args):
    terms=parse_mondo(args.mondo)
    catalog=read(DATA/'diseases.csv').drop_duplicates('Disease_ID')
    needed=set(catalog.Disease_ID); stack=list(needed)
    while stack:
        d=stack.pop()
        for p in terms.get(d,{}).get('parents',[]):
            if p not in needed: needed.add(p); stack.append(p)
    nodes=sorted(needed); index={d:i for i,d in enumerate(nodes)}
    names=catalog.set_index('Disease_ID').Disease_standard.to_dict()
    rows=[]; edges=set()
    for d in nodes:
        t=terms.get(d,{})
        name=t.get('name',names.get(d,d))
        text=' '.join([name,t.get('definition',''),' ; '.join(t.get('synonyms',[])[:3])]).strip()
        rows.append(dict(Disease_ID=d,Name=name,Text=text,Has_definition=bool(t.get('definition')),Mapped_to_MONDO=d in terms))
        edges.add((index[d],index[d]))
        for p in t.get('parents',[]):
            if p in index:
                edges.add((index[d],index[p]));edges.add((index[p],index[d]))
    from transformers import AutoTokenizer,AutoModel
    tokenizer=AutoTokenizer.from_pretrained(args.text_model,cache_dir=str(args.hf_cache),local_files_only=args.local_files_only)
    encoder=AutoModel.from_pretrained(args.text_model,cache_dir=str(args.hf_cache),local_files_only=args.local_files_only).to(args.device).eval()
    embedding=[]
    with torch.inference_mode():
        for start in tqdm(range(0,len(rows),16),desc='Disease text'):
            tokens=tokenizer([r['Text'] for r in rows[start:start+16]],padding=True,truncation=True,max_length=192,return_tensors='pt').to(args.device)
            hidden=encoder(**tokens).last_hidden_state
            mask=tokens.attention_mask.unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            embedding.append(pooled.float().cpu().numpy())
    np.save(cache/'disease_text.npy',np.concatenate(embedding))
    np.save(cache/'ontology_edges.npy',np.array(sorted(edges),dtype=np.int64).T)
    pd.DataFrame(rows).to_csv(cache/'disease_nodes.csv',index=False)
    dump(cache/'text_provenance.json',dict(model=args.text_model,config=encoder.config.to_dict(),
                                         tokenizer_class=type(tokenizer).__name__,max_tokens=192,pooling='attention-mask mean',
                                         definition_coverage=sum(r['Has_definition'] for r in rows)/len(rows)))
    del encoder
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--embedding-dir',type=Path,default=EMBED)
    parser.add_argument('--mondo',type=Path,default=MONDO)
    parser.add_argument('--text-model',default=TEXT_MODEL)
    parser.add_argument('--hf-cache',type=Path,default=HERE.parent/'embedding/huggingface_cache')
    parser.add_argument('--local-files-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    cache=HERE/'cache'; cache.mkdir(parents=True,exist_ok=True)
    signature=dict(data_metadata=sha(DATA/'metadata.json'),development=sha(DATA/'development_1to1.csv'),
                   fasta=sha(DATA/'proteins.fasta'),mondo=sha(args.mondo),text_model=args.text_model,
                   embedding_dir=str(args.embedding_dir.resolve()),prepare_code=sha(HERE/'prepare.py'))
    shared=HERE.parent/'site_disease_cross_attention_v1/cache'
    if (shared/'manifest.json').exists() and not (cache/'manifest.json').exists():
        manifest=load(shared/'manifest.json')
        compatible=all(manifest['signature'].get(k)==v for k,v in signature.items() if k!='prepare_code')
        if compatible:
            for name,h in manifest['files_sha256'].items():require(sha(shared/name)==h,f'Corrupted shared feature: {name}')
            dump(cache/'cache_reference.json',dict(path=str(shared.resolve()),manifest_sha256=sha(shared/'manifest.json')))
            print('Reusing verified first-experiment feature cache read-only; no ESM duplication.',flush=True)
            return
        require(not (cache/'cache_reference.json').exists(),'Requested features differ from shared cache; use a new experiment folder.')
    if (cache/'manifest.json').exists():
        manifest=load(cache/'manifest.json');require(manifest['signature']==signature,'Cache configuration changed; use a new experiment folder.')
        for name,h in manifest['files_sha256'].items():require(sha(cache/name)==h,f'Corrupted feature: {name}')
        print('Verified feature cache already exists; skipped.');return
    if (cache/'preparation_config.json').exists():
        require(load(cache/'preparation_config.json')==signature,'Partial feature cache configuration changed; use a new experiment folder.')
    else:
        dump(cache/'preparation_config.json',signature)
    # Download/encode text before allocating the large ESM cache. Completed text
    # survives an interrupted site preparation step.
    text_names=['disease_text.npy','ontology_edges.npy','disease_nodes.csv','text_provenance.json']
    if (cache/'text_complete.json').exists():
        for name,h in load(cache/'text_complete.json').items():require(sha(cache/name)==h,f'Corrupted disease cache: {name}')
        print('Disease cache verified; skipped.',flush=True)
    else:
        disease_features(cache,args)
        dump(cache/'text_complete.json',{name:sha(cache/name) for name in text_names})
    dev=read(DATA/'development_1to1.csv')
    columns=['Site_ID','UniProt','Residue','Position','Sequence_position','Sequence_version','Sequence_length','Site_window_31aa']
    sites=dev[columns].drop_duplicates().sort_values('Site_ID').reset_index(drop=True)
    require(not sites.Site_ID.duplicated().any(),'Conflicting canonical coordinates/window')
    proteins=sorted(sites.UniProt.unique());pi={p:i for i,p in enumerate(proteins)}
    sequences=fasta(DATA/'proteins.fasta')
    required_bytes=len(sites)*31*1280*2
    print(f'{len(sites)} sites, {len(proteins)} proteins; ESM window cache {required_bytes/1024**3:.2f} GiB',flush=True)
    missing=[p for p in proteins if not (args.embedding_dir/(p+'.pt')).exists()]
    if missing:
        dump(cache/'missing_embeddings.json',missing)
        raise ValueError(f'{len(missing)} ESM files missing; see cache/missing_embeddings.json. No sites dropped.')
    require(shutil.disk_usage(cache).free>required_bytes+1024**3,'Insufficient disk space for lossless 31aa ESM cache')
    progress_path=cache/'esm_progress.jsonl'
    completed={}
    if progress_path.exists():
        for line in progress_path.read_text(encoding='utf-8').splitlines():
            try:
                item=__import__('json').loads(line);completed[item['UniProt']]=item
            except __import__('json').JSONDecodeError:
                # A partial last journal line after interruption is ignored.
                pass
    mode='r+' if progress_path.exists() else 'w+'
    residue=np.lib.format.open_memmap(cache/'esm_windows.npy',mode=mode,dtype=np.float16,shape=(len(sites),31,1280))
    means=np.lib.format.open_memmap(cache/'protein_mean.npy',mode=mode,dtype=np.float32,shape=(len(proteins),1280))
    aa=np.lib.format.open_memmap(cache/'aa.npy',mode=mode,dtype=np.int64,shape=(len(sites),31))
    mask=np.lib.format.open_memmap(cache/'mask.npy',mode=mode,dtype=bool,shape=(len(sites),31))
    # A newline separates any incomplete journal record from subsequent entries.
    with progress_path.open('a',encoding='utf-8') as f:f.write('\n')
    embedding_manifest=[]
    for protein,group in tqdm(sites.groupby('UniProt',sort=True),desc='ESM windows'):
        path=args.embedding_dir/(protein+'.pt')
        if protein in completed:
            require(sha(path)==completed[protein]['SHA256'],f'Raw ESM changed during preparation: {protein}')
            embedding_manifest.append(completed[protein]);continue
        tensor=torch.load(path,map_location='cpu',weights_only=True)
        sequence=sequences[protein]
        require(isinstance(tensor,torch.Tensor) and tuple(tensor.shape)==(len(sequence),1280),f'ESM shape mismatch: {protein}')
        require(torch.isfinite(tensor).all().item(),f'Nonfinite ESM: {protein}')
        means[pi[protein]]=tensor.float().mean(0).numpy()
        array=tensor.float().numpy()
        for row_id,row in group.iterrows():
            center=int(row.Sequence_position)-1
            require(0<=center<len(sequence) and sequence[center]==row.Residue,f'Canonical sequence mismatch: {row.Site_ID}')
            indices=np.arange(center-15,center+16); valid=(indices>=0)&(indices<len(sequence))
            window=''.join(sequence[j] if ok else 'X' for j,ok in zip(indices,valid))
            require(window==row.Site_window_31aa,f'Window mismatch: {row.Site_ID}')
            residue[row_id]=0;residue[row_id,valid]=array[indices[valid]].astype(np.float16)
            aa[row_id]=[AA.index(c) if c in AA else 20 for c in window];mask[row_id]=valid
        record=dict(UniProt=protein,Length=len(sequence),File=str(path.resolve()),SHA256=sha(path),Sequence_SHA256=__import__('hashlib').sha256(sequence.encode()).hexdigest())
        embedding_manifest.append(record)
        for array_to_flush in [residue,means,aa,mask]:array_to_flush.flush()
        with progress_path.open('a',encoding='utf-8') as f:
            f.write(__import__('json').dumps(record)+'\n');f.flush();os.fsync(f.fileno())
    residue.flush();del residue
    for array_to_flush in [means,aa,mask]:array_to_flush.flush()
    del means,aa,mask
    np.save(cache/'site_protein.npy',sites.UniProt.map(pi).to_numpy(dtype=np.int64))
    matrix=substitution_matrices.load('BLOSUM62')
    blosum=np.zeros((21,20),dtype=np.float32)
    for i,a in enumerate(AA[:20]):
        blosum[i]=[matrix[a,b]/4.0 for b in AA[:20]]
    np.save(cache/'blosum.npy',blosum)
    sites.to_csv(cache/'sites.csv',index=False)
    pd.DataFrame(embedding_manifest).to_csv(cache/'embedding_provenance.csv',index=False)
    files={p.name:sha(p) for p in cache.iterdir() if p.is_file() and p.name not in ['manifest.json','missing_embeddings.json']}
    dump(cache/'manifest.json',dict(signature=signature,site_count=len(sites),protein_count=len(proteins),esm_dim=1280,
                                  features_use_labels=False,files_sha256=files,
                                  provenance_limit='Existing raw ESM tensors contain no sequence hash; dimensions checked and current sequence hashes recorded, but historical same-length sequence identity cannot be proved. Long proteins may use overlapping chunks.'))
    print('Features ready. No independent-test labels used.',flush=True)

if __name__=='__main__':main()
