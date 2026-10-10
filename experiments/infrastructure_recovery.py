"""Archive infrastructure failures without filtering model-quality outcomes."""
import json
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from oracle_audit.io import ROOT, save_json, sha256

MANIFEST=ROOT/'manifests/infrastructure-recovery-20261009.json'


@lru_cache(maxsize=1)
def recovery_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'attempts':[]}


def provenance(folder):
    key=str(folder.relative_to(ROOT))
    history=[r for r in recovery_manifest()['attempts'] if r['active_path']==key]
    prior=history[-1] if history else None
    import os
    return dict(attempt_number=len(history)+1,
                infrastructure_recovery=prior is not None,
                prior_attempt_path=prior['archive_path'] if prior else None,
                serving_revision=os.environ.get('ORACLE_SERVING_REVISION','initial_no_cuda_graph'))


def archive(round_number=1):
    old=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    if old and round_number<=old.get('round_number',1):raise ValueError('Recovery round already exists')
    if old:
        snapshot=MANIFEST.with_name(MANIFEST.stem+f'-before-round{round_number}.json')
        if snapshot.exists():raise ValueError('Previous receipt snapshot already exists')
        shutil.copy2(MANIFEST,snapshot)
    destination=ROOT/f'outputs/infrastructure_attempts/20261009_recovery{round_number}'
    roots=[ROOT/'outputs/qwen_supplement'/x for x in ['validation','matched_test','low_temperature']]
    roots.append(ROOT/'outputs/guard_experiment/runs')
    attempts=[]
    for root in roots:
        for marker in sorted(root.rglob('input_messages.json')):
            folder=marker.parent;meta_path=folder/'run_meta.json'
            meta=json.loads(meta_path.read_text()) if meta_path.exists() else None
            failed=meta is None or meta['return_code']!=0 or meta['timed_out']
            if not failed:continue
            target=destination/folder.relative_to(ROOT)
            attempts.append(dict(active_path=str(folder.relative_to(ROOT)),archive_path=str(target.relative_to(ROOT)),
                cause='interrupted_without_completion_record' if meta is None else 'transport_failure',
                prior_return_code=None if meta is None else meta['return_code'],
                files=[dict(path=str(p.relative_to(folder)),sha256=sha256(p)) for p in sorted(folder.rglob('*')) if p.is_file()]))
    configs=[]
    for root in [ROOT/'outputs/qwen_supplement',ROOT/'outputs/guard_experiment']:
        for p in sorted(root.glob('execution_config*.json')):
            configs.append(dict(active_path=str(p.relative_to(ROOT)),archive_path=str((destination/p.relative_to(ROOT)).relative_to(ROOT)),sha256=sha256(p)))
    receipt=dict(status='ARCHIVE_PLANNED',round_number=round_number,created_utc=datetime.now(timezone.utc).isoformat(),
        attempts=(old['attempts'] if old else [])+attempts,configurations=(old['configurations'] if old else [])+configs,
        previous_receipt_sha256=sha256(snapshot) if old else None,
        timeout_attribution='A deadline exceeded is not proof of transport outage; long model trajectories may also time out. Recovery is disclosed repeated-attempt evaluation.',
        selection='Nonzero return code, timeout, or missing completion record only. Normal-return malformed/wrong model outputs are never selected.',
        amendment='The original no-replacement run was interrupted. Failed/interrupted infrastructure attempts remain archived; the active fixed-case panel is an explicitly recovered analysis, not the original first-attempt panel.',
        accounting='Report archived attempts and their observed token usage separately. Unreported tokens from timed-out/interrupted requests are unknown, not zero.')
    save_json(MANIFEST,receipt)
    for item in attempts+configs:
        source=ROOT/item['active_path'];target=ROOT/item['archive_path']
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(source),str(target))
    receipt['status']='ARCHIVED';save_json(MANIFEST,receipt)
    print(f'Preserved {len(attempts)} failed/interrupted attempt folders; successful responses unchanged.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,default=1)
    archive(parser.parse_args().round)
