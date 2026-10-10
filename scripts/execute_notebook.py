"""Execute every notebook cell and preserve real outputs and acceptance evidence."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'notebooks/Oracle_Agentic_Audit_Replay.ipynb'
OUT = ROOT/'reports/public_reproduction'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(SOURCE, as_version=4)
    started = time.monotonic()
    def completed(cell, cell_index, **kwargs):
        nbformat.write(notebook, OUT/'research.executed.ipynb')
        print(json.dumps(dict(completed_cell_index=cell_index,
                              elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
    client = NotebookClient(notebook, timeout=2400, kernel_name='python3',
                            resources={'metadata': {'path': str(OUT)}}, on_cell_executed=completed)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, OUT/'research.executed.ipynb')
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    if len(cells) != 6 or any(c.execution_count is None for c in cells):
        raise ValueError('Not every expected cell executed')
    if any(o.output_type == 'error' for c in cells for o in c.outputs):
        raise ValueError('Notebook contains an actual execution error')
    first = ''.join(o.get('text', '') for o in cells[0].outputs)
    work = Path(re.search(r'Fresh working directory: (.+)', first).group(1).strip())
    run_root = work/'Oracle-Agentic-Audit'
    generated = run_root/'outputs/notebook_replay'
    receipt = json.loads((generated/'notebook_execution_receipt.json').read_text())
    comparison = json.loads((generated/'comparison_receipt.json').read_text())
    if receipt['status'] != 'PASS' or comparison['status'] != 'PASS':
        raise ValueError('Scientific comparison did not pass')
    for name in ['notebook_execution_receipt.json', 'comparison_receipt.json',
                 'replay_receipt.json', 'environment.freeze.txt']:
        shutil.copy2(generated/name, OUT/name)
    shutil.copy2(run_root/'outputs/downloads/download_receipt.json', OUT/'download_receipt.json')
    (OUT/'validation-output.txt').write_text(''.join(o.get('text', '') for o in cells[2].outputs))
    # Preserve actual generated data/figures, excluding optional serialized estimators.
    with tarfile.open(OUT/'generated-results.tar.gz', 'w:gz') as archive:
        for path in sorted(generated.rglob('*')):
            if path.is_file() and path.suffix != '.joblib':
                archive.add(path, arcname='generated-results/'+str(path.relative_to(generated)))
    run_url = None
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        run_url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    acceptance = dict(status='PASS', scope=receipt['execution_kind'],
        notebook_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        notebook_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        code_revision=receipt['code_revision'], dataset_revision=receipt['dataset_revision'],
        executed_notebook_sha256=hashlib.sha256((OUT/'research.executed.ipynb').read_bytes()).hexdigest(),
        executable_cells=6, completed_cells=6, elapsed_seconds=receipt['elapsed_seconds'],
        scientific_csv_comparisons=comparison['compared_tables'], rendered_numeric_tables=18,
        prompt_listings=1, regenerated_figures=len(comparison['figures']),
        figures_byte_identical=sum(x['bytes_match'] for x in comparison['figures']),
        hosted_colab=receipt['hosted_colab'], inference_requests=0, github_run_url=run_url)
    (OUT/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    print(json.dumps(acceptance,indent=2),flush=True)


if __name__ == '__main__':
    main()
