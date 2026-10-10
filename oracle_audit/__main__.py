import argparse
import csv
import json
from pathlib import Path
from .core import checked_file, metrics, read_predictions

ROOT = Path(__file__).resolve().parents[1]


def template_check():
    fixture = json.loads((ROOT/'manifests/synthetic-fixture.json').read_text())
    checked_file(ROOT/fixture['path'], fixture['sha256'])
    index = json.loads((ROOT/'manifests/result-index.json').read_text())
    required = ('result_id', 'reported_result_or_boundary', 'original_sources',
                'query_code', 'queried_data', 'process_code', 'processed_data',
                'analysis_code', 'result_outputs', 'commands', 'evidence_status',
                'interpretation_boundary', 'data_revision')
    for record in index:
        if any(field not in record for field in required):
            raise ValueError('Incomplete result-index schema')
    print('Template structure and synthetic checksum verified; paper replication unverified.')


def release_check():
    manifest = json.loads((ROOT/'manifests/research-inputs.json').read_text())
    blockers = []
    for field in ('dataset_revision', 'descriptor_arxiv_version', 'input_files',
                  'model_specification', 'replay_command', 'inference_command',
                  'reference_outputs', 'license_approval', 'scientific_review_receipt'):
        if not manifest.get(field):
            blockers.append(field)
    index = json.loads((ROOT/'manifests/result-index.json').read_text())
    for record in index:
        if record['evidence_status'] in ('PLANNED', 'BLOCKED', 'SYNTHETIC'):
            blockers.append(record['result_id'])
    # Research execution and public release are distinct checked conditions.
    from oracle_audit.io import verify_inputs
    from analysis.collect import collect
    try:
        verify_inputs()
        _,_,status=collect(ROOT/'outputs/status')
        blockers.extend('incomplete_inference:'+k for k,v in status.items() if v.get('required',True) and not v['complete'])
    except (ValueError,OSError,KeyError) as error:
        blockers.append('input/run verification: '+str(error))
    for name in ['clean_replay_receipt.json','hosted_colab_receipt.json']:
        p=ROOT/'manifests'/name
        if not p.exists() or json.loads(p.read_text()).get('status')!='PASS':blockers.append(name)
    if blockers:raise ValueError('Research release BLOCKED: ' + '; '.join(blockers))
    print('Checked release requirements passed.')


def experiment_check():
    from analysis.collect import collect
    from experiments.scope import required_complete
    _,_,status=collect(ROOT/'outputs/status')
    print(json.dumps(status,indent=2))
    if not required_complete(status,successful=True):raise ValueError('Required inference still incomplete or has transport failures.')
    print('Every planned case-condition record is present; inspect recovery history, failures and results before interpretation.')


def smoke():
    fixture = json.loads((ROOT/'manifests/synthetic-fixture.json').read_text())
    rows = read_predictions(checked_file(ROOT/fixture['path'], fixture['sha256']))
    output = ROOT/'outputs/synthetic/metrics.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['system', 'evidence_status', 'case_runs', 'unique_cases', 'brier',
              'coverage', 'selective_error', 'mean_action_loss']
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for system in sorted({r['system'] for r in rows}):
            result = metrics([r for r in rows if r['system'] == system])
            writer.writerow({'system': system, 'evidence_status': 'SYNTHETIC', **result})
    print(f'SYNTHETIC example only: {output.relative_to(ROOT)}')


def main():
    parser = argparse.ArgumentParser(description='Oracle research execution and evidence checks')
    parser.add_argument('command', choices=('smoke', 'template-check', 'release-check','experiment-check','replay'))
    args = parser.parse_args()
    try:
        {'smoke': smoke, 'template-check': template_check,
         'release-check': release_check,'experiment-check':experiment_check,
         'replay':lambda:__import__('analysis.replay',fromlist=['run']).run(ROOT/'outputs/final')}[args.command]()
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
