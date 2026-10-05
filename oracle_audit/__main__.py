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
    # The scaffold must never certify real replication merely by filling text fields.
    blockers.append('real input/reference validation and scientific release gate not implemented')
    raise ValueError('Research release BLOCKED: ' + '; '.join(blockers))


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
    parser = argparse.ArgumentParser(description='Oracle research scaffold, not paper replication')
    parser.add_argument('command', choices=('smoke', 'template-check', 'release-check'))
    args = parser.parse_args()
    try:
        {'smoke': smoke, 'template-check': template_check,
         'release-check': release_check}[args.command]()
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
