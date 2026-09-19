"""Reproduce the complete-recording held-out evaluation and final figures."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for name in ['models','predictions','results','figures']:(root/name).mkdir(exist_ok=True)
steps=[['-m','unittest','discover','-s','scripts','-p','test_policy_core.py','-v'],['scripts/fit_frozen_models.py'],['scripts/evaluate_complete.py'],['scripts/summarize_results.py'],['scripts/supplementary_audits.py'],['scripts/make_final_figures.py']]
for args in steps:
 print('RUN',*args,flush=True)
 subprocess.run([sys.executable,*args],cwd=root,check=True)
print('All analysis steps completed. See results/ and TABLE_PROVENANCE.csv.')
