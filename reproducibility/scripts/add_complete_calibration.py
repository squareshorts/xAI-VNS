from pathlib import Path
import pandas as pd,re,json,difflib
R=Path(__file__).resolve().parents[1];P=R/'patches';f=R/'patched_manuscript/main.tex';s=f.read_text()
cal=pd.read_csv(R/'results/calibration_summary.csv').query("exposure=='combined'").set_index('method')
new=r'''\subsection{Score calibration on completed exposure}
\label{sec:complete_calibration}
The additional recordings change the evaluation distribution, not the fitted model. On complete exposure the nested raw LightGBM score had AUROC $0.601\pm0.093$ and AUPRC $0.082\pm0.059$ (Table~\ref{tab:complete_calibration}). The same frozen isotonic mapping used for policy evaluation yielded Brier score $0.0244\pm0.0161$ and ECE $0.0242\pm0.0252$. Sigmoid calibration had a lower complete-exposure Brier score, but it was not substituted for the calibrator selected on original calibration data. Raw discrimination and calibration are shared by all common-score policy comparators. Changes between original and completed exposure should not be interpreted without the changed label prevalence; added-recording-only AUROC and AUPRC are undefined because that subset has no positive labels under the record-local convention.

\begin{table}[htbp]
\centering
\caption{Frozen nested-core LightGBM score diagnostics over all 175 EDFs. Values are equal-weight case means $\pm$ SD. Isotonic was selected on original cross-fitted calibration data, not from this table.}
\label{tab:complete_calibration}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Mapping & AUROC & AUPRC & Brier & ECE \\
\midrule
'''
for meth,name in [('raw','Raw'),('sigmoid','Sigmoid'),('isotonic','Isotonic')]:
 q=cal.loc[meth];new+=name+' & '+' & '.join(f'${q[c+"_mean"]:.3f}\\pm{q[c+"_sd"]:.3f}$' for c in ['auroc','auprc','brier','ece'])+r' \\'+'\n'
new+=r'''\bottomrule
\end{tabular}
\end{table}
\FloatBarrier

'''
anchor=r'\subsection{Timing at decision availability}';assert anchor in s;s=s.replace(anchor,new+anchor)
f.write_text(s)
(P/'19_complete_calibration.tex').write_text(new)
p=P/'patch_index.json';d=json.loads(p.read_text());d.append(dict(name='19_complete_calibration',insert_before=anchor,new=new));p.write_text(json.dumps(d,indent=2))
p=P/'MANUSCRIPT_PATCHES.md';p.write_text(p.read_text()+'\n## 19_complete_calibration\nInsert immediately before `'+anchor+'` after applying patch 11.\n```latex\n'+new+'\n```\n')
a=(R/'source_manuscript/main.tex').read_text();(P/'main.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),s.splitlines(True),fromfile='original/main.tex',tofile='patched/main.tex')))
