"""Hashed local research inputs, distinct from a verified public data release."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(obj):
    if isinstance(obj, dict): return {str(k): jsonable(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, np.generic): return jsonable(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data=json.dumps(jsonable(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary=stream.name;stream.write(data)
    os.replace(temporary,path)


def table(out, name, frame):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / (name+'.csv'), index=False, float_format='%.12g')
    frame.to_parquet(out / (name+'.parquet'), index=False)


def verify_inputs():
    manifest = json.loads((ROOT/'manifests/local-inputs.json').read_text())
    for row in manifest['files']:
        p = ROOT / row['path']
        if not p.is_file() or sha256(p) != row['sha256']:
            raise ValueError(f'Missing or modified input: {p}')
    return manifest
