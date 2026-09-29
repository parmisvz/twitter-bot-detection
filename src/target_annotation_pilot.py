"""Pilot-only label-free selection; no model imports or test-label reads."""
import json
import hashlib
from pathlib import Path
import pandas as pd
from src.target_audit import ROOT, OUT as AUDIT, population, assert_not_audit, digest, write

OUT = 'results/versions/target_annotation_pilot_v1'
SEED = 20260929

def assert_not_pilot_evaluation(keys, root=ROOT):
    """Call before any future M0/M2/M4 evaluation; fail closed on missing registry."""
    pointer = json.loads((Path(root)/'configs/target_pilot_registry.json').read_text())
    registry = Path(root)/pointer['registry']
    if digest(registry) != pointer['sha256']:
        raise ValueError('Pilot registry changed')
    reserved = set(json.loads(registry.read_text())['user_keys'])
    if reserved.intersection(map(str, keys)):
        raise ValueError('Pilot-only users cannot participate in model evaluation')

def run():
    out = ROOT/OUT
    pointer = ROOT/'configs/target_pilot_registry.json'
    if out.exists() or pointer.exists():
        raise ValueError('Pilot already exists; do not overwrite or resample')
    # Hash existing frozen packages before and after; do not parse private exposure data.
    protected = {}
    for folder in [ROOT/AUDIT, ROOT/'results/versions/target_audit_annotation_materials_v1']:
        manifest = json.loads((folder/'manifest.json').read_text())
        for name, expected in manifest['artifacts'].items():
            p = folder/name
            if digest(p) != expected: raise ValueError('Existing artifact hash mismatch')
        for p in folder.rglob('*'):
            if p.is_file(): protected[p] = digest(p)
    for p in [ROOT/'configs/target_audit_registry.json', ROOT/'analysis/06_multimodal_fusion.ipynb']:
        protected[p] = digest(p)
    frame, inputs = population(ROOT)
    reserved = set(json.loads((ROOT/AUDIT/'private/exclusion_registry.json').read_text())['user_keys'])
    # Column-restricted read: identity and split only. Never load historical test labels.
    split_path = ROOT/'configs/frozen_user_split_v1.csv'
    splits = pd.read_csv(split_path, usecols=['user_key','split'], dtype=str)
    if not splits.user_key.is_unique: raise ValueError('Duplicate split identity')
    test = set(splits.loc[splits.split.eq('test'), 'user_key'])
    if len(test) != 144: raise ValueError('Unexpected historical test identity count')
    excluded = reserved | set(splits.user_key)
    inputs['frozen_split_file'] = digest(split_path)
    # Check saved development evaluation population identities only, never scores/labels.
    evaluation_keys = set()
    paths = sorted((ROOT/'results/versions/modality_ablation_v1').glob('*/provenance/validation_population.csv'))
    if not paths: raise ValueError('Missing modality evaluation identity provenance')
    for p in paths:
        evaluation_keys.update(pd.read_csv(p, usecols=['user_key'], dtype=str).user_key)
        inputs[p.relative_to(ROOT).as_posix()] = digest(p)
    excluded |= evaluation_keys
    eligible = frame.loc[~frame.user_key.isin(excluded)].copy()
    # Three cases per nonempty follower/description group; training exercise, not a survey.
    sizes = eligible.groupby('stratum').size().to_dict()
    if len(sizes) != 6 or min(sizes.values()) < 3:
        raise ValueError('Unexpected eligible strata; review allocation rather than silently change it')
    rank = lambda key, stage: hashlib.sha256(f'{SEED}|{stage}|{key}'.encode()).hexdigest()
    selected = []
    for stratum in sorted(sizes):
        pool = eligible.loc[eligible.stratum.eq(stratum), 'user_key'].tolist()
        selected.extend(sorted(pool, key=lambda k: (rank(k,'selection'),k))[:3])
    selected.sort(key=lambda k: (rank(k,'annotation_order'),k))
    assert_not_audit(selected)
    assert len(selected) == len(set(selected)) == 18
    assert not set(selected) & (test | evaluation_keys | set(splits.user_key))
    lookup = frame.set_index('user_key')
    rows = [{'annotation_id':f'PILOT{i:03d}', 'user_key':key, 'role':'pilot_only',
             'snapshot_date':str(lookup.loc[key,'last_updated']), 'stratum':lookup.loc[key,'stratum']}
            for i,key in enumerate(selected,1)]
    out.mkdir()
    (out/'private').mkdir()
    write(out/'private/selected_registry.json', rows)
    write(out/'private/exclusion_registry.json', {'role':'pilot_only', 'excluded_from_model_evaluation':True,
          'user_keys':sorted(selected)})
    template = json.loads((ROOT/AUDIT/'annotator_A/annotation_forms.jsonl').read_text().splitlines()[0])
    for role in ['annotator_A','annotator_B','adjudicator']:
        (out/role).mkdir()
        forms = [{**template,'annotation_id':r['annotation_id'],
                 **({'adjudication_rationale':''} if role=='adjudicator' else {})} for r in rows]
        (out/role/'annotation_forms.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in forms),encoding='utf-8')
        (out/role/'ANNOTATOR_GUIDE.md').write_bytes((ROOT/'reports/annotation_materials/ANNOTATOR_GUIDE.md').read_bytes())
        (out/role/'PILOT_README.md').write_text(
            '# Assigned non-audit pilot\n\nThese 18 IDs are assigned pilot-only cases. '
            'The unassigned-template paragraph in the general guide does not apply to this package. '
            'The coordinator retains the account mapping and will supply authorized case packets. '
            'No evidence has been collected and no labels have been assigned. '
            'Work independently using the same Human / Bot / Uncertain rubric. '
            'Do not view another annotator\'s decisions until both submissions are locked.\n',encoding='utf-8')
    (out/'COORDINATOR_WORKFLOW.md').write_bytes((ROOT/'reports/annotation_materials/PILOT_WORKFLOW.md').read_bytes())
    write(out/'selection_provenance.json', {'seed':SEED,'method':'SHA256(seed|stage|user_key) ranking, lexical tie-break; 3 per nonempty group',
          'role':'pilot_only; annotation training, not model evaluation', 'eligible_group_sizes':sizes,
          'selected_n':18,'inputs':inputs,'code_sha256':digest(Path(__file__)),
          'audit_overlap':0,'historical_test_identity_overlap':0,'labeled_split_overlap':0,
          'saved_modality_validation_overlap':0,'test_labels_read':False,'predictions_read_or_generated':False,
          'protected_artifacts':{p.relative_to(ROOT).as_posix():v for p,v in protected.items()}})
    for p,v in protected.items():
        if digest(p)!=v: raise ValueError('Existing artifact changed')
    write(pointer, {'registry':OUT+'/private/exclusion_registry.json',
          'sha256':digest(out/'private/exclusion_registry.json'),'role':'pilot_only'})
    assert_not_pilot_evaluation(reserved)
    try: assert_not_pilot_evaluation(selected)
    except ValueError: pass
    else: raise ValueError('Pilot evaluation guard failed')
    print(json.dumps({'pilot_n':18,'audit_overlap':0,'historical_test_overlap':0,'validation_overlap':0,'frozen_artifacts_unchanged':True}))

if __name__ == '__main__': run()
