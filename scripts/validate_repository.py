#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()

def main() -> int:
    required=[
      ROOT/'data/processed/x_incidence_per_100k.csv', ROOT/'data/processed/x_log1p.csv',
      ROOT/'data/scenarios/scenario_manifest.json', ROOT/'results/tables/final_benchmark_table.csv',
      ROOT/'results/tables/causal_nowcasting_table.csv', ROOT/'supplementary_material/Methodology_and_Experimental_Setup.tex'
    ]
    missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing: raise SystemExit(f"Missing required files: {missing}")
    manifest=json.loads((ROOT/'data/scenarios/scenario_manifest.json').read_text(encoding='utf-8'))
    files=list((ROOT/'data/scenarios').glob('**/messages_seed_*.csv.gz'))
    assert len(files)==280, len(files)
    assert manifest['message_file_count']==280
    assert manifest['seed_count']==20
    assert manifest['scenario_count']==14
    inc=pd.read_csv(ROOT/'data/processed/x_incidence_per_100k.csv',index_col=0)
    log=pd.read_csv(ROOT/'data/processed/x_log1p.csv',index_col=0)
    assert inc.shape==(365,6) and log.shape==(365,6)
    subprocess.run([sys.executable, str(ROOT/'code/rki_data_pipeline/validate_outputs.py'),
      '--config',str(ROOT/'configs/data_config.json'),'--processed-dir',str(ROOT/'data/processed'),
      '--scenario-dir',str(ROOT/'data/scenarios')],check=True,cwd=ROOT)
    checksum=hashlib.md5((ROOT/'data/raw/IfSG_COVID-19_Erkrankungsbeginn_Erwartungswert.csv').read_bytes()).hexdigest()
    assert checksum=='e59953809e7a8050bb40045c6172ee30', checksum
    print(json.dumps({'status':'ok','processed_shape':[365,6],'scenario_files':280,'seeds':20,'scenarios':14,'raw_md5':checksum},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
