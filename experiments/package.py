"""Build a local, checksum-indexed replay bundle after required inference completes."""
import gzip
import io
import json
import tarfile
from pathlib import Path
from oracle_audit.io import ROOT,sha256,save_json


def build():
    receipt=json.loads((ROOT/'outputs/final/replay_receipt.json').read_text())
    if receipt['status']!='COMPLETE_OFFLINE_REPLAY' or receipt['bootstrap_resamples']!=2000:
        raise ValueError('A complete strict replay is required before packaging final research artifacts')
    roots=['analysis','experiments','models','policies','guards','oracle_audit','figures','docs','manifests','tests','notebooks',
           'data/fixtures','data/local','outputs/qwen_supplement','outputs/guard_experiment','outputs/infrastructure_attempts','outputs/scope_amendment','outputs/tabular','outputs/final',
           'Readme.md','requirements.lock','Makefile','.gitignore']
    paths=set()
    for name in roots:
        path=ROOT/name
        for p in path.rglob('*') if path.is_dir() else [path]:
            if not p.is_file():continue
            relative=p.relative_to(ROOT)
            if '__pycache__' in relative.parts or p.suffix in ['.pyc','.tgz'] or p.name.startswith('.env'):continue
            if any(part.startswith('.venv') for part in relative.parts):continue
            paths.add(p)
    records=[dict(path=str(p.relative_to(ROOT)),sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(paths)]
    manifest=dict(format='oracle-local-replay-v1',scientific_status=receipt['status'],
                  hosting_status='LOCAL_ONLY; publication licensing and hosted Colab are not certified',
                  excludes=['credentials','model weights','Python environments','private manuscripts'],files=records)
    directory=ROOT/'outputs/delivery';directory.mkdir(parents=True,exist_ok=True)
    target=directory/'oracle-camera-ready-20261009.tar.gz'
    with target.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode='w') as archive:
                for p in sorted(paths):
                    info=archive.gettarinfo(str(p),arcname='Oracle-Agentic-Audit/'+str(p.relative_to(ROOT)))
                    info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                    with p.open('rb') as stream:archive.addfile(info,stream)
                data=(json.dumps(manifest,indent=2)+'\n').encode()
                info=tarfile.TarInfo('Oracle-Agentic-Audit/bundle_manifest.json');info.size=len(data);info.mtime=0
                archive.addfile(info,io.BytesIO(data))
    save_json(directory/'bundle_receipt.json',dict(path=str(target.relative_to(ROOT)),sha256=sha256(target),bytes=target.stat().st_size,files=len(records),status='LOCAL_BUNDLE_CREATED'))
    print(target)
    return target


if __name__=='__main__':build()
