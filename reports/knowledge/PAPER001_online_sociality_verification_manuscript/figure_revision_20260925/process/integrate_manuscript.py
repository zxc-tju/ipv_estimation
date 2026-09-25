"""Install verified figure assets and synchronize the v4.5 manuscript references."""
from pathlib import Path
import json, re, shutil
from figure_style import ROOT, RUN, ASSETS, DATA
PAPER=(ROOT/'../../2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle').resolve()
CAPS=json.loads((RUN/'process/captions_draft.json').read_text())
SUPPS=[
 ('figS2_subjective_sensitivity','fig:suppsubjectivesensitivity',r'\textbf{Subjective contrasts under participant-paired and common-scenario weighting.} Counterpart-position contrasts for \textbf{a} perceived atypicality, \textbf{b} comfort and \textbf{c} reported interaction cost. Filled circles reproduce equal-participant within-participant differences with participant-pair bootstrap intervals; open squares give equal-common-scenario estimates with scenario bootstrap intervals. These are distinct estimands with distinct resampling units. A--W, C--W and A--C use 8, 9 and 12 shared scenarios, respectively. All intervals are the existing pointwise $95\%$ intervals from 10,000 draws. Scenario resampling fixes the observed raters; participant-pair resampling conditions on the selected stimuli. This is a composition sensitivity analysis, not an independent replication. Source Data retain all 30 contrasts.'),
 ('figS3_human_levels','fig:supphumanlevels',r'\textbf{Matched human and automated flag rates across the three frozen reference levels.} Outside-range rates at nominal $80\%$, $90\%$ and $95\%$ coverage for the natural-driving test reference and the two matched benchmark arms. Denominators are judgeable moments in the corresponding population. The natural reference is an alarm-inflation check and is not a matched comparison of behavioural levels. The $90\%$ operating point remains the prespecified main display; the other two levels are retained here without selection by significance. Points are frozen estimates; no missing interval is reconstructed. Source Data provide exact counts, rates and available intervals.'),
 ('figS4_estimator_details','fig:suppcandidate',r'\textbf{Candidate preferences and the operational readability check.} \textbf{a} Seven candidate angles weight the agent\textquotesingle s own-progress and interaction costs through cosine and sine. They span $[-3\pi/8,3\pi/8]$ at $\pi/8$ spacing. \textbf{b},\textbf{c} Frozen candidate-weight examples from two frames of one interaction illustrate a concentrated and a near-uniform likelihood distribution, respectively. The readable value is the candidate-weighted mean; the unreadable frame has no reported reading. These panels describe the implementation of the previously published estimator, not a new numerical-recovery guarantee. Source Data contain the original candidate grid and weight vectors.')]
EDS=[
 ('figED3_source_transfer','fig:edtransfer',r'\textbf{Source transfer of the behavioural predictor and of the reference monitor are distinct checks.} \textbf{a} Full-state-space prediction of event-mean IPV in each source omitted during fitting, retaining negative out-of-source $R^2$ values. Stems connect point estimates to zero and are not uncertainty intervals. \textbf{b} Empirical coverage of the $90\%$ reference among accepted readable moments, shown alongside human-support abstention among all readable moments in each omitted source. Coverage and abstention have different denominators; both are needed to interpret transfer. The four sources remain separate and no universal transfer guarantee is inferred. These are existing source-held-out results, not new fits. Source Data contain exact evaluated, accepted and abstaining counts.'),
 ('figED5_reference_distribution','fig:eddistribution',r'\textbf{The human reading distribution and deployment reference summaries describe different populations.} \textbf{a} Frozen distribution of $486{,}660$ readable natural-driving test moments. \textbf{b} Three illustrative deployment situations, summarised by the separate medians of the frozen $90\%$ lower and upper bounds within each situation class. These summaries use all corresponding automated-benchmark candidate moments, before readability or human-support filtering: 2,336 crossing-with-priority, 10,291 merging-with-priority and 291 merging-with-equal-priority moments. The endpoints are marginal medians and need not form any individual moment\textquotesingle s interval. Across all 67,861 benchmark candidates, predicted widths span $1.05$--$2.37$\,rad. These deployment summaries are not intervals calculated from the human histogram in \textbf{a}. Source Data preserve the two populations separately.')]
def block(asset,label,caption):
 return '\n'.join([r'\begin{figure}[!htbp]\centering',r'\includegraphics[width=\linewidth]{figures/'+asset+'.pdf}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{figure}'])
required=list(CAPS.values())+[{'asset':x[0]} for x in SUPPS+EDS]
for x in required:
 for ext in ['pdf','svg','png']:
  assert (ASSETS/f"{x['asset']}.{ext}").exists(),f"Not yet available: {x['asset']}.{ext}"
# This version is the synchronized source text, before this task's manuscript edits.
t=(RUN/'process/main_before.tex').read_text()
assert 'fig:measured' in t and 'subjective_results' in t
for label,c in CAPS.items():
 pat=r'\\begin\{figure\}.*?\\end\{figure\}'
 matched=[]
 def replace(m):
  if r'\label{'+label+'}' in m.group(0):
   matched.append(1);return block(c['asset'],label,c['caption'])
  return m.group(0)
 t=re.sub(pat,replace,t,flags=re.S)
 assert len(matched)==1,label
# Keep the measured-role prediction target precise and align all panel callouts.
t=t.replace('including the candidate-based reading, the context-conditioned reference, explicit\nabstention and a complete interaction example.','including the trajectory-derived reading, the context-conditioned reference, explicit\nabstention and one unchanged interaction example.')
t=t.replace('which partner will ultimately give way at AUC','which partner has the higher mean IPV in the final quarter at AUC')
t=t.replace('correlation is negative in every source.','correlation is negative in every source (Fig.~\\ref{fig:context}a,b).')
t=t.replace('kinematics over the same windows remains near $0.56$--$0.57$.','kinematics over the same windows remains near $0.56$--$0.57$\n(Fig.~\\ref{fig:context}c). The target is the late-interaction IPV role, not observed passage order.')
t=t.replace("with one source's interval admitting\nzero.","with one source's interval admitting\nzero (Fig.~\\ref{fig:context}d).")
t=t.replace(r'Fig.~\ref{fig:context}a).',r'Fig.~\ref{fig:context}e).').replace(r'Fig.~\ref{fig:context}b),',r'Fig.~\ref{fig:context}f),').replace(r'Fig.~\ref{fig:context}c; Methods 4.4',r'Extended Data Fig.~\ref{fig:edtransfer}a; Methods 4.4')
old='''The cost of a single global reference is visible in the distribution itself. Across the $486{,}660$
readable human moments of the held-out fold, the range selected by the situation is narrow in some
real situations and wide in others: among three illustrative classes, crossing with priority is the
narrowest and merging with equal priority the widest, while the deployed range spans
$1.05$--$2.37$\\,rad across the benchmark (Fig.~\\ref{fig:measured}c). The classes make the variation
visible; the deployed range conditions on the continuous situation and varies within each class.'''
new='''The human reading distribution and the situation-dependent reference provide complementary views
of this heterogeneity (Extended Data Fig.~\\ref{fig:eddistribution}). The distribution contains
$486{,}660$ readable natural-driving test moments. Separately, across the automated benchmark's
$67{,}861$ candidate moments, the predicted $90\\%$ reference width spans $1.05$--$2.37$\\,rad.
Among three illustrative deployment classes, the interval formed by the median lower and upper bounds
is narrowest for crossing with priority and widest for merging with equal priority. These are
summaries over deployment candidates, before gate filtering; the actual range varies within each
class and the two median endpoints need not belong to the same moment.'''
assert old in t;t=t.replace(old,new)
# Use the exact supplied half-thousandth rather than inconsistent three-place rounding.
t=t.replace('$+0.682$', '$+0.6825$').replace('$+0.682\\ [', '$+0.6825\\ [')
# Clear superseded editing-state comments only.
t=re.sub(r'% Interim caption.*?\n(?=\\begin\{figure\})','',t,flags=re.S)
t=re.sub(r'% Existing asset retained.*?\n(?=\\begin\{figure\})','',t,flags=re.S)
t=re.sub(r'% Figure rendering is intentionally deferred.*?\n(?=\\begin\{figure\})','',t,flags=re.S)

# Layout-only repairs preserve the estimator and affiliation content.
t=t.replace(r'\doublespacing',r'\doublespacing'+'\n'+r'\setlength{\emergencystretch}{3em}',1)
t=t.replace(r'\textit{$^{1}$College of Transportation Engineering, Tongji University, Shanghai, China}',r'{\small\textit{$^{1}$College of Transportation Engineering, Tongji University, Shanghai, China}}')
t=t.replace(r'\textit{$^{2}$Institute for Transport Studies, University of Leeds, Leeds, United Kingdom}',r'{\small\textit{$^{2}$Institute for Transport Studies, University of Leeds, Leeds, United Kingdom}}')
old_eq=r"""\begin{align}
  \ell_{a,k}(t) &= p\!\left(X_a(W_t)\,\middle|\,\widetilde X_a(W_t;\theta_k),\,X_j(W_t),\,\mathcal{M}\right),
  & \pi_{a,k}(t) &= \frac{\ell_{a,k}(t)}{\sum_m \ell_{a,m}(t)},
  & \hat\theta_a(t) &= \sum_k \pi_{a,k}(t)\,\theta_k,
  \label{eq:estimator}
\end{align}"""
new_eq=r"""\begin{equation}
\begin{aligned}
  \ell_{a,k}(t) &= p\!\left(X_a(W_t)\,\middle|\,\widetilde X_a(W_t;\theta_k),\,X_j(W_t),\,\mathcal{M}\right),\\
  \pi_{a,k}(t) &= \frac{\ell_{a,k}(t)}{\sum_m \ell_{a,m}(t)},
  \qquad \hat\theta_a(t) = \sum_k \pi_{a,k}(t)\,\theta_k.
\end{aligned}
\label{eq:estimator}
\end{equation}"""
assert old_eq in t
t=t.replace(old_eq,new_eq)

# Supplementary figures follow existing S1 and Table S2; original source-package names are not numbering.
marker='% ======================================================================\n% Extended Data figures'
assert marker in t
t=t.replace(marker,'\n\n'.join(block(*x) for x in SUPPS)+'\n\n'+r'\input{figures/subjective_supplement_tables.tex}'+'\n\n'+marker)
marker=r'\clearpage'+'\n'+r'\bibliographystyle{unsrt}'
assert marker in t
t=t.replace(marker,'\n\n'.join(block(*x) for x in EDS)+'\n\n'+marker)
t=t.replace(r'\usepackage{booktabs}',r'\usepackage{booktabs}'+'\n'+r'\usepackage{longtable}')

t=t.replace(r'\pagestyle{fancy}',r'\setlength{\headheight}{15pt}'+'\n'+r'\pagestyle{fancy}')
t=t.replace(r'$\{-3,-2,-1,0,1,2,3\}\times\pi/8$',r'$k\pi/8$, $k\in\{-3,-2,-1,0,1,2,3\}$')
t=t.replace('The ego-position assertive-minus-within atypicality difference is', 'For ego-position atypicality, the assertive-minus-within difference is')

for f in ASSETS.iterdir():
 if f.suffix in ['.pdf','.png','.svg','.tex']:shutil.copy2(f,PAPER/'figures'/f.name)
out=PAPER/'figures/source_data';out.mkdir(exist_ok=True)
for f in DATA.glob('*.csv'):shutil.copy2(f,out/f.name)
(PAPER/'main.tex').write_text(t)
print('Installed figure assets, aggregate Source Data, captions, callouts and complete subjective supplement.')
