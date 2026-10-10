"""Compare scientific CSV references with explicit numeric tolerance and missingness."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from oracle_audit.io import save_json,sha256


def compare(reference,candidate,partial=False):
    reference=Path(reference);candidate=Path(candidate);rows=[]
    if not partial:
        for root in [reference,candidate]:
            receipt=json.loads((root/'replay_receipt.json').read_text())
            if receipt['status']!='COMPLETE_OFFLINE_REPLAY' or receipt['bootstrap_resamples']!=2000:
                raise ValueError('Complete replay and full bootstrap required for a PASS receipt')
    # While jobs run, these collections legitimately grow; completed panel analyses do not.
    excluded={'supplement_runs.csv','guard_runs.csv','infrastructure_attempt_audit.csv',
              'infrastructure_attempt_summary.csv','serving_runtime_cohorts.csv'} if partial else set()
    for source in sorted(reference.rglob('*.csv')):
        relative=source.relative_to(reference)
        if relative.name in excluded:continue
        other=candidate/relative
        if not other.exists():raise ValueError('Missing replay table: '+str(relative))
        try:left=pd.read_csv(source);right=pd.read_csv(other)
        except pd.errors.EmptyDataError:
            if source.read_bytes()!=other.read_bytes():raise ValueError('Empty-table mismatch')
            rows.append(dict(path=str(relative),status='PASS_EMPTY',reference_sha256=sha256(source),candidate_sha256=sha256(other)));continue
        if list(left.columns)!=list(right.columns) or left.shape!=right.shape:raise ValueError('Shape mismatch: '+str(relative))
        max_error=0.
        for col in left.columns:
            a,b=left[col],right[col]
            if not a.isna().equals(b.isna()):raise ValueError(f'Missingness mismatch: {relative}:{col}')
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                if not np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True):raise ValueError(f'Numeric mismatch: {relative}:{col}')
                if not pd.api.types.is_bool_dtype(a):max_error=max(max_error,float((a-b).abs().max()) if a.notna().any() else 0.)
            elif not a.fillna('<MISSING>').equals(b.fillna('<MISSING>')):raise ValueError(f'Value mismatch: {relative}:{col}')
        rows.append(dict(path=str(relative),rows=len(left),columns=len(left.columns),max_absolute_error=max_error,
                         status='PASS',reference_sha256=sha256(source),candidate_sha256=sha256(other)))
    if not rows:raise ValueError('No reference tables compared')
    rendered=[]
    if not partial:
        # Verify actual rendered tables/prompts, not only the files' existence.
        for source in sorted((reference/'tables').glob('*.tex')):
            other=candidate/'tables'/source.name
            if not other.exists() or source.read_bytes()!=other.read_bytes():
                raise ValueError('Rendered table/prompt mismatch: '+source.name)
            rendered.append(source.name)
        figure_check=[]
        for source in sorted((reference/'figures').glob('*.png')):
            other=candidate/'figures'/source.name
            if not other.exists():raise ValueError('Missing rendered figure: '+source.name)
            # Raster bytes can vary with platform fonts; underlying numeric inputs
            # are checked above. Record exact image identity separately.
            figure_check.append(dict(path=source.name,bytes_match=sha256(source)==sha256(other),
                                     reference_sha256=sha256(source),candidate_sha256=sha256(other)))
    receipt=dict(status='PASS_PARTIAL_SCOPE' if partial else 'PASS',scope='Scientific CSVs and rendered table/prompt text; figure hashes reported separately from numeric agreement',
                 absolute_tolerance=1e-10,relative_tolerance=1e-10,compared_tables=len(rows),tables=rows,
                 rendered_tables_and_prompts=rendered,figures=[] if partial else figure_check,
                 environment_receipt=str(candidate/'replay_receipt.json'),hosted_colab='NOT_RUN')
    save_json(candidate/'comparison_receipt.json',receipt)
    print(f'{receipt["status"]}: {len(rows)} tables agree within specified tolerances.')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();compare(a.reference,a.candidate,a.partial)
