"""The user-authorized fixed scope, shared by scheduling and offline analysis."""
import json
from oracle_audit.io import ROOT, sha256

MANIFEST = ROOT / 'manifests/experiment-scope-20261009.json'
INTERRUPTIONS = ROOT / 'manifests/scope-reduction-execution-20261009.json'


def load():
    return json.loads(MANIFEST.read_text())


def required_complete(status, successful=False):
    required = [v for v in status.values() if v.get('required', True)]
    return bool(required) and all(v['complete'] and
        (not successful or v['transport_failures'] == 0) for v in required)


def select(frame, plan, guard=False):
    if frame.empty:
        return frame.copy(), frame.copy()
    if guard:
        spec = plan['guards']
        keep = (frame.system.isin(spec['systems']) & frame.repeat.isin(spec['repeats']) &
                frame.scenario.isin(spec['scenarios']) & frame.guard_mode.isin(spec['modes']))
    else:
        keep = frame.suite.eq('__none__')
        for suite in ['validation', 'matched_test', 'low_temperature']:
            spec = plan[suite]
            if spec['required']:
                keep |= (frame.suite.eq(suite) & frame.system.isin(spec['systems']) &
                         frame.repeat.isin(spec['repeats']))
    return frame.loc[keep].copy(), frame.loc[~keep].copy()


def provenance(folder):
    receipt = json.loads(INTERRUPTIONS.read_text())
    history = [a for a in receipt['interrupted_attempts']
               if a['active_path'] == str(folder.relative_to(ROOT))]
    return dict(experiment_scope=load()['scope_id'], scope_manifest_sha256=sha256(MANIFEST),
                scope_interrupted_attempts=len(history),
                scope_prior_attempt_paths=[a['archive_path'] for a in history])


def audit():
    plan = load()
    for record in plan['matched_test']['frozen_completion_records']:
        if sha256(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen B2 repeat changed after scope selection')
    receipt = json.loads(INTERRUPTIONS.read_text())
    for config in receipt['configurations']:
        if sha256(ROOT / config['archive_path']) != config['sha256']:
            raise ValueError('Scope amendment configuration archive changed')
    for attempt in receipt['interrupted_attempts']:
        for file in attempt['files']:
            if sha256(ROOT / attempt['archive_path'] / file['path']) != file['sha256']:
                raise ValueError('Scope interruption archive changed')
    return plan
