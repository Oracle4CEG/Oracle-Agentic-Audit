# Original figures and result tables

`render.py` generates eight original figures from saved scientific outputs. `tables.py` exports 18 numeric LaTeX tables and extracts the exact prompt strings from archived source code. The illustrated UMA cash flows are recomputed and checked against transfer records.

The complete pipeline calls both exporters. `manifests/manuscript-outputs.json` maps each table/figure to inputs and code. LaTeX tables go under outputs/<run>/tables/; PDF/PNG figures go under outputs/<run>/figures/. These revised results replace superseded numerical claims rather than reproducing unsupported old claims.
