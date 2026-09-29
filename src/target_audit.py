"""Label-free frozen audit selection. No models, predictions or test-data loading."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PHASE='final_data/versions/preprocessing_v2_e861084d1a722a21'
PHASE_HASH='594a2defef59d663ef171020b7cd1ba75aa321c30e7a7d09b57b560563c109c9'
OUT='results/versions/target_audit_v1/seed20260928_n600'
SEED=20260928
STRATA=[f'{f}_{d}' for f in ['lt100','100to999','ge1000','missing'] for d in ['observed','missing']]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def allocation(sizes,n=600):
    if sum(sizes.values())<n:raise ValueError('Insufficient population')
    base={h:min(10,sizes[h]) for h in STRATA}
    remaining=n-sum(base.values());capacity={h:sizes[h]-base[h] for h in STRATA};total=sum(capacity.values())
    if remaining<0:raise ValueError('Minimum allocation exceeds sample')
    floors={h:(remaining*capacity[h]//total if total else 0) for h in STRATA}
    residual=remaining-sum(floors.values())
    ranked=sorted(STRATA,key=lambda h:(-(remaining*capacity[h]%total if total else 0),STRATA.index(h)))
    for h in ranked[:residual]:floors[h]+=1
    return {h:base[h]+floors[h] for h in STRATA}

def population(root):
    directory=root/PHASE;manifest=directory/'manifest.json'
    if digest(manifest)!=PHASE_HASH:raise ValueError('Phase-2 manifest changed')
    m=json.loads(manifest.read_text());rel='splits/users/unlabeled_users.csv';p=directory/rel
    if digest(p)!=m['artifacts'][rel]['sha256']:raise ValueError('Population changed')
    # Deliberately parse only sampling attributes, annotation status and snapshot dates.
    frame=pd.read_csv(p,usecols=['user_key','followers_count','description','last_updated','label_raw','label_binary'],dtype={'user_key':str})
    if len(frame)!=18331 or frame.user_key.isna().any() or not frame.user_key.is_unique:raise ValueError('Invalid identities')
    if frame[['label_raw','label_binary']].notna().any().any():raise ValueError('Population already annotated')
    if not frame.user_key.eq(frame.user_key.str.strip().str.lstrip('@').str.lower()).all():raise ValueError('Noncanonical identity')
    f=frame.followers_count
    if ((f.dropna()<0)|~np.isfinite(f.dropna())).any():raise ValueError('Invalid follower attribute')
    frame['stratum']=[('missing' if pd.isna(v) else 'lt100' if v<100 else '100to999' if v<1000 else 'ge1000')+'_'+('missing' if pd.isna(d) else 'observed') for v,d in zip(f,frame.description)]
    return frame,{'phase2_manifest':PHASE_HASH,'population':digest(p)}

def select(frame):
    sizes={h:int(frame.stratum.eq(h).sum()) for h in STRATA};counts=allocation(sizes)
    rng=np.random.Generator(np.random.PCG64(SEED));keys=[];strata=[]
    for h in STRATA:
        pool=sorted(frame.loc[frame.stratum.eq(h),'user_key'])
        chosen=sorted(rng.choice(pool,size=counts[h],replace=False).tolist()) if counts[h] else []
        keys.extend(chosen)
        strata.append({'stratum':h,'population_n':sizes[h],'sample_n':counts[h],
            'inclusion_probability':counts[h]/sizes[h] if sizes[h] else None,
            'analysis_weight':sizes[h]/counts[h] if counts[h] else None})
    # Annotation IDs do not encode stratum or source ordering.
    order=np.random.Generator(np.random.PCG64(SEED+1)).permutation(sorted(keys))
    indexed=frame.set_index('user_key');lookup={r['stratum']:r for r in strata};rows=[]
    for i,key in enumerate(order,1):
        record=indexed.loc[key];s=lookup[record.stratum]
        rows.append({'annotation_id':f'TA{i:04d}','user_key':str(key),'stratum':record.stratum,
            'snapshot_date':str(record.last_updated),'inclusion_probability':s['inclusion_probability'],
            'analysis_weight':s['analysis_weight']})
    return rows,strata

def assert_not_audit(keys,root=ROOT):
    """Mandatory fail-closed guard for all future fitting/selection populations."""
    pointer=json.loads((Path(root)/'configs/target_audit_registry.json').read_text())
    path=Path(root)/pointer['registry']
    if digest(path)!=pointer['sha256']:raise ValueError('Audit exclusion registry missing/changed')
    reserved=set(json.loads(path.read_text())['user_keys'])
    overlap=reserved.intersection(map(str,keys))
    if overlap:raise ValueError(f'Forbidden audit overlap: {len(overlap)} identities')

def run(root=ROOT):
    root=Path(root);frame,hashes=population(root);rows,strata=select(frame);out=root/OUT
    if out.exists() or (root/'configs/target_audit_registry.json').exists():raise ValueError('Audit already frozen; verify instead of resampling')
    out.mkdir(parents=True)
    for folder in ['private','annotator_A','annotator_B','adjudicator']:(out/folder).mkdir()
    write(out/'private/selected_registry.json',rows);write(out/'strata.json',strata)
    registry={'schema_version':'target_audit_v1','permanent_exclusion':True,'user_keys':sorted(r['user_key'] for r in rows)}
    write(out/'private/exclusion_registry.json',registry)
    evidence=[{k:r[k] for k in ['annotation_id','user_key','snapshot_date']} for r in rows]
    write(out/'private/evidence_index.json',evidence)
    # Existing prior-training membership only; never load scores or pseudo-label values.
    prior=root/'results/versions/semi_supervised_v2/20260920T104058299599Z_da452759'
    pm=json.loads((prior/'manifest.json').read_text());rel='pseudo_labels/selected_for_student.csv'
    expected=pm['artifacts'][rel]['sha256']
    if digest(prior/rel)!=expected:raise ValueError('Prior training membership changed')
    prior_keys=set(pd.read_csv(prior/rel,usecols=['user_key'],dtype=str).user_key)
    write(out/'private/prior_exposure.json',{'scope':'Known selected student training membership only; not an exhaustive legacy exposure audit',
        'source':str(prior.relative_to(root)/rel),'source_sha256':expected,
        'records':[{'annotation_id':r['annotation_id'],'known_student_training_exposure':r['user_key'] in prior_keys} for r in rows]})
    fields=['annotator_id','evidence_access_date','supported_period','evidence_sources','evidence_types','evidence_summary',
        'automation_indicators','human_operation_indicators','limitations','temporal_mismatch','account_access_status',
        'label','confidence','rationale','organization_news_service_status']
    for reviewer in ['annotator_A','annotator_B','adjudicator']:
        forms=[{'annotation_id':r['annotation_id'],'rubric_version':'target_audit_v1',**dict.fromkeys(fields,''),
                **({'adjudication_rationale':''} if reviewer=='adjudicator' else {})} for r in rows]
        (out/reviewer/'annotation_forms.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms),encoding='utf-8')
    config={'schema_version':'target_audit_v1','sample_size':600,'population_size':18331,'seed':SEED,'rng':'NumPy PCG64',
        'annotation_order_seed':SEED+1,'stratum_order':STRATA,'minimum_per_stratum':10,
        'allocation':'minimum capped at stratum size; remainder proportional to residual capacity; integer largest remainder; fixed-order ties',
        'description_definition':'canonical raw description nonmissing, no cleaned-description substitution',
        'replacement_policy':'never replace inaccessible or uncertain accounts','input_hashes':hashes,
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'code_sha256':digest(Path(__file__)),'test_access':False,'predictions_generated':False}
    write(out/'sampling_config.json',config)
    (out/'RUBRIC_AND_WORKFLOW.md').write_bytes((root/'reports/TARGET_AUDIT_PROTOCOL.md').read_bytes())
    write(out/'manifest.json',{'complete':True,'artifacts':{p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    write(root/'configs/target_audit_registry.json',{'schema_version':'target_audit_v1','registry':OUT+'/private/exclusion_registry.json',
        'sha256':digest(out/'private/exclusion_registry.json'),'sample_manifest':OUT+'/manifest.json','sample_manifest_sha256':digest(out/'manifest.json')})
    return verify(root)

def verify(root=ROOT):
    root=Path(root);out=root/OUT;frame,hashes=population(root);rows,strata=select(frame)
    m=json.loads((out/'manifest.json').read_text())
    for p,h in m['artifacts'].items():
        if digest(out/p)!=h:raise ValueError('Artifact changed: '+p)
    if rows!=json.loads((out/'private/selected_registry.json').read_text()) or strata!=json.loads((out/'strata.json').read_text()):raise ValueError('Selection reproduction failed')
    keys=[r['user_key'] for r in rows]
    if len(set(keys))!=600:raise ValueError('Wrong sample')
    for reviewer in ['annotator_A','annotator_B','adjudicator']:
        forms=[json.loads(s) for s in (out/reviewer/'annotation_forms.jsonl').read_text().splitlines()]
        if [r['annotation_id'] for r in forms]!=[r['annotation_id'] for r in rows]:raise ValueError('Forms misaligned')
        if any(r['label'] or r['confidence'] for r in forms):raise ValueError('Template has labels')
        if any(any(k in r for k in ['user_key','stratum','prediction','probability','pseudo_label']) for r in forms):raise ValueError('Blind form contains restricted fields')
    assert_not_audit(set(frame.user_key)-set(keys),root)
    try:assert_not_audit(keys[:1],root)
    except ValueError:pass
    else:raise ValueError('Exclusion guard failed')
    return {'sample_n':600,'unique':True,'selection_exact':True,'hashes_verified':len(m['artifacts']),
        'prediction_free_forms':True,'exclusion_guard_passed':True,'test_access':False,
        'known_prior_student_exposure':sum(r['known_student_training_exposure'] for r in json.loads((out/'private/prior_exposure.json').read_text())['records']),
        'strata':strata}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');a=parser.parse_args()
    print(json.dumps(verify() if a.verify else run(),indent=2))
