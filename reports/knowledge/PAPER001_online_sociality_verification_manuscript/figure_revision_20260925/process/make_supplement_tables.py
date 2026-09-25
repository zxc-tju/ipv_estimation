"""Format supplied aggregate subjective comparisons without recomputation."""
from pathlib import Path
import pandas as pd
from figure_style import ROOT, RUN
P = ROOT/'data/manually_organized/RQ028_v1_主观评分_20260914/论文可用结果提取_20260925/tables'
ITEMS={'q1_rev':'Atypicality','q2_yield_bipolar':'Q2 yielding','q3':'Q3 predictability','q4':'Comfort','q5_cost_cp':'Reported cost','q5b':'Q5b naturalness'}
def pvalue(x):
    if x>=.001: return f'{x:.4f}'.rstrip('0').rstrip('.')
    mant,exp=f'{x:.2e}'.split('e'); return rf'{mant}\!\times\!10^{{{int(exp)}}}'
def interval(r):
    est = '+0.6825' if abs(r.estimate - .6825) < 1e-12 else f'{r.estimate:+.3f}'
    return rf'${est}\ [{r.ci_low:+.3f},{r.ci_high:+.3f}]$'
def table(name,caption,label,header,alignment,rows):
    return '\n'.join([r'\begingroup\scriptsize',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.1}',rf'\begin{{longtable}}{{{alignment}}}',rf'\caption{{\textbf{{{caption}}}}}\label{{{label}}}\\',r'\toprule',header+r' \\',r'\midrule',r'\endfirsthead',r'\toprule',header+r' \\',r'\midrule',r'\endhead',r'\bottomrule',r'\endfoot',*rows,r'\end{longtable}',r'\endgroup',''])
parts=[r'\clearpage',r'\subsection*{Supplementary Note 3. Complete aggregate subjective results}',r'These tables reproduce the supplied frozen aggregate outputs without rerunning tests or resampling. A, assertive-side; C, accommodating-side; W, within-range comparison. Ego and counterpart (CP) denote rating positions. Each main contrast uses 40 participants in 20 pairs. Q2 uses its supplied bipolar coding; the remaining items use the supplied rating scales. The main pointwise intervals resample participant pairs 10,000 times. The two-sided tests use 20 pair-averaged differences with 19 degrees of freedom; Holm correction covers all 30 main comparisons. No sensitivity analysis changes the primary inclusion rule or supplies a new corrected significance claim.']
d=pd.read_csv(P/'participant_paired_contrasts.csv')
rows=[]
for r in d.itertuples():
    rows.append(' & '.join(['Ego' if r.seat=='ego' else 'CP',ITEMS[r.metric],r.contrast.replace('-', '--'),interval(r),f'${pvalue(r.p_unadjusted)}$',f'${pvalue(r.p_holm_30)}$'])+r' \\')
parts.append(table('primary','All 30 paired subjective comparisons.','tab:subjective_full','Position & Item & Contrast & Difference [95\\% CI] & Raw $p$ & Holm $p$','lllrrr',rows))
for fname,caption,label,ncol in [
 ('source_label_sensitivity.csv','Source-label stratification of the subjective contrasts.','tab:subjective_source','n_pairs'),
 ('alternative_code_sensitivity.csv','Sensitivity to excluding the supplied alternative-code records.','tab:subjective_alt','n_subjects'),
 ('common_scenario_sensitivity.csv','Equal-common-scenario sensitivity for all subjective endpoints.','tab:subjective_scenarios','n_common_scenarios')]:
    d=pd.read_csv(P/fname); is_source='arm' in d
    if is_source:
        parts.append(r'The source labels below describe the supplied stimulus groups; they do not establish source identification or a vehicle ranking. Intervals use participant-pair resampling within each group.')
    elif 'alternative' in fname:
        parts.append(r'This sensitivity excludes records labelled \texttt{alt27=1} in the supplied coding and preserves the primary analysis separately. The table reports available participants and the existing pair-bootstrap intervals; no primary records are removed by this display.')
    else:
        parts.append(r'Each contrast gives equal weight to segments within scenario and then to shared scenarios. Intervals resample shared scenarios 10,000 times with the observed raters fixed. This estimand differs from the participant-paired analysis and is not an independent replication.')
    rows=[]
    for r in d.itertuples():
        x=([str(r.arm).upper()] if is_source else [])+['Ego' if r.seat=='ego' else 'CP',ITEMS[r.metric],r.contrast.replace('-','--'),interval(r),str(getattr(r,ncol))]
        rows.append(' & '.join(x)+r' \\')
    parts.append(table(fname,caption,label,('Source & ' if is_source else '')+'Position & Item & Contrast & Difference [95\\% CI] & '+('Pairs' if is_source else 'Participants' if 'alternative' in fname else 'Scenarios'),('l' if is_source else '')+'lllrr',rows))
d=pd.read_csv(RUN/'source_data/fig4_counterpart_all_ratios.csv')
name_map={'counterpart_speed_drop_ratio':'Speed reduction','counterpart_speed_range_ratio':'Speed range','counterpart_heading_change_ratio':'Net heading change','counterpart_yaw_rate_ratio':'Peak yaw rate'}
rows=[]
for r in d.itertuples():
    rows.append(' & '.join([name_map[r.outcome],r.side+'--W',interval(r),str(r.n_side),str(r.n_within)])+r' \\')
parts += [r'\clearpage',r'\subsection*{Supplementary Note 4. Complete counterpart-response battery}',r'The four existing counterpart-response endpoints below retain both directions, including heading and yaw endpoints not shown on the main speed-response axis. Ratios compare pooled moment-weighted medians. Pointwise 95\% intervals use 2,000 whole-run bootstrap draws over 175 scenario runs. A--W is the frozen assertive battery; C--W is the previously reported post-hoc accommodating extension. The reference value is one; intervals containing one do not establish equivalence. No absolute-difference interval is inferred from these ratios.']
parts.append(table('objective','Complete counterpart-response median ratios.','tab:counterpart_full','Endpoint & Contrast & Ratio [95\\% CI] & Side moments & W moments','llrrr',rows))
(RUN/'assets/subjective_supplement_tables.tex').write_text('\n\n'.join(parts))
print('Formatted 30 main + 60 source-label + 30 alternative-code + 30 common-scenario rows.')
