"""Extract candidate manuscript evidence using only the supplied final release.

Run with Python from any directory. Source reads are allowlisted, symlink-safe,
and hashed. Outputs never overwrite source data. No external data or models.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from scipy import stats
import scipy

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
TABLES = OUT / 'tables'
SOURCE_FILES = ['README.md', '文件清单.csv', 'metadata/field_dictionary.csv'] + [
    'data/' + x for x in ['ratings.csv', 'segments.csv', 'subjects.csv',
                          'sessions.csv', 'trials.csv', 'exclusions.csv',
                          'moments.parquet', 'dial.parquet']]
AUDIT = []
SEED = 20260925
B = 10000


def source_path(rel):
    assert rel in SOURCE_FILES, rel
    p = ROOT / rel
    assert p.resolve().is_relative_to(ROOT.resolve()), 'Out-of-scope source'
    assert not p.is_symlink(), 'Symlink source forbidden'
    return p


def save(df, name):
    df.to_csv(TABLES / name, index=False, encoding='utf-8-sig', float_format='%.10g')


def interval(x, seed=SEED):
    x = np.asarray(x, dtype=float)
    assert np.isfinite(x).all() and len(x) > 1
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return np.quantile(means, [.025, .975])


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order]))
    return q


for rel in SOURCE_FILES:
    p = source_path(rel)
    AUDIT.append({'relative_path': rel, 'size_bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
save(pd.DataFrame(AUDIT), 'source_hashes.csv')
manifest = pd.read_csv(source_path('文件清单.csv'))
current = {x['relative_path']: x for x in AUDIT}
manifest_check = []
for row in manifest.to_dict('records'):
    rel = row['relative_path']
    actual = current.get(rel)
    manifest_check.append({'relative_path': rel, 'status':
                          'not_present' if actual is None else
                          'match' if actual['sha256'] == row['sha256'] else 'differs',
                          'listed_sha256': row['sha256'],
                          'current_sha256': actual['sha256'] if actual else None})
save(pd.DataFrame(manifest_check), 'supplied_manifest_check.csv')

dfs = {name: pd.read_csv(source_path(f'data/{name}.csv'), keep_default_na=False)
       for name in ['ratings', 'segments', 'subjects', 'sessions', 'trials', 'exclusions']}
r, seg, sub, sess, trials, exclusions = [dfs[n] for n in
                                         ['ratings', 'segments', 'subjects', 'sessions', 'trials', 'exclusions']]
metrics = ['q1_rev', 'q2_yield_bipolar', 'q3', 'q4', 'q5_cost_cp', 'q5b']
for c in metrics:
    r[c] = pd.to_numeric(r[c].replace('', np.nan))
for c in ['exceed_peak_rad', 'inside_margin_min_rad']:
    seg[c] = pd.to_numeric(seg[c].replace('', np.nan))
assert r.shape[0] == 1192 and trials.shape[0] == 600
assert not r.duplicated(['trial_id', 'subject_id', 'seat']).any()
assert not seg.segment_id.duplicated().any()
assert not sub.subject_id.duplicated().any()
assert not trials.trial_id.duplicated().any()
completed = trials.query('included_in_reference_analysis == 1')
assert completed.status.eq('completed').all() and len(completed) == 596
assert set(r.trial_id) == set(completed.trial_id)
assert r.groupby('trial_id').size().eq(2).all()
assert r.groupby('trial_id').seat.apply(lambda s: set(s) == {'ego', 'cp'}).all()
assert set(exclusions.trial_id) == set(trials.query('status == "aborted"').trial_id)
for c in ['q1_rev', 'q3', 'q4', 'q5_cost_cp', 'q5b']:
    assert r[c].dropna().between(1, 7).all()
assert r.q2_yield_bipolar.between(-3, 3).all()
assert r.loc[r.seat.eq('ego'), ['q5_cost_cp','q5b']].isna().all().all()
assert r.loc[r.seat.eq('cp'), metrics].notna().all().all()
joined = r.merge(trials, on=['trial_id','segment_id'], validate='many_to_one')
for seat in ['ego','cp']:
    a = joined[joined.seat.eq(seat)]
    assert a.subject_id.eq(a[f'subject_{seat}_id']).all()
joined = joined.merge(seg, on='segment_id', validate='many_to_one').merge(
    sub[['subject_id','pair_id']], on='subject_id', validate='many_to_one').merge(
    sess[['session_id','session_order','block_id']], on='session_id', validate='many_to_one')
assert len(joined) == len(r) and joined.pair_id.nunique() == 20
save(joined, 'ratings_analysis_rows.csv')
save(seg.groupby(['scenario_id','class','arm']).size().rename('n_segments').reset_index(),
     'scenario_class_coverage.csv')
counts = joined.groupby(['class','arm','seat']).agg(
    n_ratings=('trial_id','size'), n_trials=('trial_id','nunique'),
    n_subjects=('subject_id','nunique'), n_pairs=('pair_id','nunique'),
    n_segments=('segment_id','nunique'), n_scenarios=('scenario_id','nunique')).reset_index()
save(counts, 'design_counts.csv')

descriptive = []
for keys, g in joined.groupby(['class','seat']):
    for m in metrics:
        a = g.dropna(subset=[m])
        if a.empty: continue
        v = a[m]
        subject = a.groupby(['pair_id','subject_id'])[m].mean()
        pair = subject.groupby('pair_id').mean()
        lo, hi = interval(pair)
        descriptive.append(dict(zip(['class','seat'],keys), metric=m,
            n_ratings=len(a), n_trials=a.trial_id.nunique(), n_subjects=a.subject_id.nunique(),
            n_pairs=len(pair), n_segments=a.segment_id.nunique(), n_scenarios=a.scenario_id.nunique(),
            raw_mean=v.mean(), raw_sd=v.std(), median=v.median(), q25=v.quantile(.25), q75=v.quantile(.75),
            subject_equal_mean=subject.mean(), ci_low=lo, ci_high=hi,
            ci_method='10000 pair-cluster percentile bootstrap; fixed stimulus set'))
desc = pd.DataFrame(descriptive)
save(desc, 'rating_descriptives.csv')

# Analysis is exploratory: all 30 metric-seat contrasts are reported, with Holm
# correction across the complete family. CIs are per-contrast, not simultaneous.
contrasts, person_rows, pair_rows = [], [], []
for seat in ['ego','cp']:
    for m in metrics:
        d = joined[joined.seat.eq(seat)].dropna(subset=[m])
        if d.empty: continue
        person = d.pivot_table(index=['pair_id','subject_id'], columns='class', values=m, aggfunc='mean')
        for first, ref in [('A','W'), ('C','W'), ('A','C')]:
            delta = person[first] - person[ref]
            assert delta.notna().all() and len(delta) == 40
            pair = delta.groupby('pair_id').mean()
            lo, hi = interval(pair)
            test = stats.ttest_1samp(pair, 0)
            contrasts.append({'seat':seat,'metric':m,'contrast':f'{first}-{ref}',
                'estimate':delta.mean(),'ci_low':lo,'ci_high':hi,'n_subjects':len(delta),
                'n_pairs':len(pair),'pair_t':test.statistic,'df':len(pair)-1,'p_unadjusted':test.pvalue,
                'estimand':'equal-subject mean within-subject class difference; fixed stimulus set',
                'ci_method':'10000 pair-cluster percentile bootstrap'})
            person_rows.extend({'pair_id':idx[0],'subject_id':idx[1],'seat':seat,'metric':m,
                'contrast':f'{first}-{ref}','difference':val} for idx,val in delta.items())
            pair_rows.extend({'pair_id':idx,'seat':seat,'metric':m,
                'contrast':f'{first}-{ref}','difference':val} for idx,val in pair.items())
con = pd.DataFrame(contrasts)
assert len(con) == 30
con['p_holm_30'] = holm(con.p_unadjusted)
save(con, 'participant_paired_contrasts.csv')
save(pd.DataFrame(person_rows), 'participant_contrast_source.csv')
save(pd.DataFrame(pair_rows), 'pair_contrast_source.csv')

# Same estimand separately within source labels; these are not AV-vs-human
# performance tests and do not authenticate the source labels.
arm_rows = []
for arm, g in joined.groupby('arm'):
    for seat in ['ego','cp']:
        for m in metrics:
            d = g[g.seat.eq(seat)].dropna(subset=[m])
            if d.empty: continue
            p = d.pivot_table(index=['pair_id','subject_id'],columns='class',values=m,aggfunc='mean')
            for first, ref in [('A','W'),('C','W'),('A','C')]:
                delta = (p[first]-p[ref]).dropna()
                pairs = delta.groupby('pair_id').mean()
                lo, hi = interval(pairs)
                arm_rows.append({'arm':arm,'seat':seat,'metric':m,'contrast':f'{first}-{ref}',
                    'estimate':delta.mean(),'ci_low':lo,'ci_high':hi,'n_subjects':len(delta),'n_pairs':len(pairs)})
save(pd.DataFrame(arm_rows),'source_label_sensitivity.csv')

# Scene matching removes the imbalance of scene composition. First average
# raters within each stimulus, then stimuli within each scenario and class.
segment_means = joined.groupby(['segment_id','scenario_id','class','arm','seat'])[metrics].mean().reset_index()
save(segment_means,'stimulus_mean_ratings.csv')
scene_rows, scene_summary = [], []
for seat in ['ego','cp']:
    for m in metrics:
        d = segment_means[segment_means.seat.eq(seat)].dropna(subset=[m])
        if d.empty: continue
        p = d.pivot_table(index='scenario_id',columns='class',values=m,aggfunc='mean')
        for first, ref in [('A','W'),('C','W'),('A','C')]:
            delta = (p[first]-p[ref]).dropna()
            lo,hi=interval(delta)
            scene_rows.extend({'scenario_id':idx,'seat':seat,'metric':m,'contrast':f'{first}-{ref}',
                               'difference':val} for idx,val in delta.items())
            scene_summary.append({'seat':seat,'metric':m,'contrast':f'{first}-{ref}',
                'estimate':delta.mean(),'ci_low':lo,'ci_high':hi,'n_common_scenarios':len(delta),
                'n_positive':int((delta>0).sum()),'n_negative':int((delta<0).sum()),
                'n_zero':int((delta==0).sum()),
                'ci_method':'10000 scenario percentile bootstrap; fixed observed raters',
                'scope':'descriptive sensitivity; selected scenario population only'})
save(pd.DataFrame(scene_rows),'common_scenario_source.csv')
save(pd.DataFrame(scene_summary),'common_scenario_sensitivity.csv')

# Robustness to the retained alternative-code flag, without changing primary cohort.
alt_rows=[]
for seat in ['ego','cp']:
    for m in metrics:
        d=joined[joined.seat.eq(seat)&joined.alt27.eq(0)].dropna(subset=[m])
        if d.empty: continue
        p=d.pivot_table(index=['pair_id','subject_id'],columns='class',values=m,aggfunc='mean')
        for first,ref in [('A','W'),('C','W'),('A','C')]:
            delta=(p[first]-p[ref]).dropna()
            lo,hi=interval(delta.groupby('pair_id').mean())
            alt_rows.append({'seat':seat,'metric':m,'contrast':f'{first}-{ref}',
                            'estimate':delta.mean(),'ci_low':lo,'ci_high':hi,
                            'n_subjects':len(delta),'n_ratings_in_metric':len(d),
                            'scope':'sensitivity excluding supplied alt27=1; no primary exclusion'})
save(pd.DataFrame(alt_rows),'alternative_code_sensitivity.csv')

m = pd.read_parquet(source_path('data/moments.parquet'))
dial = pd.read_parquet(source_path('data/dial.parquet'))
assert not m.duplicated(['trial_id','subject_id','seat','t_rel']).any()
assert not dial.duplicated(['trial_id','subject_id','seat','t_rel']).any()
assert set(m.trial_id) == set(completed.trial_id)
assert set(dial.trial_id) == set(trials.trial_id)
assert m.groupby(['trial_id','subject_id','seat']).size().eq(201).all()
mm=m.merge(joined[['trial_id','subject_id','seat','segment_id','class','scenario_id','pair_id']],
           on=['trial_id','subject_id','seat'],validate='many_to_one')
verdict=mm.groupby(['class','verdict_replayed']).size().rename('n_rating_moments').reset_index()
save(verdict,'verdict_class_mapping.csv')
for c,allowed in [('A',{'below','within','abstain'}),('C',{'above','within','abstain'}),('W',{'within','abstain'})]:
    assert set(mm.loc[mm['class'].eq(c),'verdict_replayed']) == allowed
assert mm.judgeable_replayed.eq(mm.verdict_replayed.ne('abstain').astype(int)).all()

# Retained onset comparison is descriptive only; no semantic meaning assigned
# to dial_z because the supplied normalization and endpoints are incomplete.
keys=['trial_id','subject_id','seat','segment_id','class','scenario_id','pair_id']
mm['t_since_onset'] = mm.t_since_onset.round(6)
pre=mm[mm.t_since_onset.ge(-2)&mm.t_since_onset.lt(0)].groupby(keys).dial_z.agg(['mean','count'])
post=mm[mm.t_since_onset.ge(0)&mm.t_since_onset.lt(2)].groupby(keys).dial_z.agg(['mean','count'])
win=pre.join(post,lsuffix='_pre',rsuffix='_post').reset_index()
assert win[['count_pre','count_post']].eq(20).all().all()
win['delta_dial_z']=win.mean_post-win.mean_pre
save(win,'dial_onset_window_rows.csv')
time_rows=[]
for (c,seat),g in win.groupby(['class','seat']):
    p=g.groupby(['pair_id','subject_id']).delta_dial_z.mean()
    pairs=p.groupby('pair_id').mean()
    lo,hi=interval(pairs)
    time_rows.append({'class':c,'seat':seat,'n_series':len(g),'n_subjects':len(p),
                     'n_pairs':len(pairs),'estimate':p.mean(),'ci_low':lo,'ci_high':hi,
                     'baseline':'[-2,0) seconds relative to retained onset',
                     'post':'[0,2) seconds relative to retained onset',
                     'status':'descriptive; dial_z normalization and meaning incomplete'})
save(pd.DataFrame(time_rows),'dial_onset_descriptive.csv')
temporal_person=win.groupby(['pair_id','subject_id','class']).delta_dial_z.mean().unstack('class')
temporal_con=[]
for first,ref in [('A','W'),('C','W'),('A','C')]:
    v=(temporal_person[first]-temporal_person[ref]).groupby('pair_id').mean()
    lo,hi=interval(v)
    temporal_con.append({'contrast':f'{first}-{ref}','estimate':v.mean(),'ci_low':lo,'ci_high':hi,
                         'n_pairs':len(v),'unit':'pair; average seats within participant first',
                         'status':'descriptive only; fixed stimulus set'})
save(pd.DataFrame(temporal_con),'dial_onset_contrasts.csv')
temporal_segment=win.groupby(['segment_id','scenario_id','class']).delta_dial_z.mean().reset_index()
temporal_scene=temporal_segment.pivot_table(index='scenario_id',columns='class',values='delta_dial_z')
common=temporal_scene.dropna(subset=['A','C','W'])
scene_temporal=[]
for first,ref in [('A','W'),('C','W'),('A','C')]:
    v=common[first]-common[ref]
    lo,hi=interval(v)
    scene_temporal.append({'contrast':f'{first}-{ref}','estimate':v.mean(),'ci_low':lo,'ci_high':hi,
                           'n_common_scenarios':len(v),'n_positive':int((v>0).sum()),
                           'loo_min':min(v.drop(i).mean() for i in v.index),
                           'loo_max':max(v.drop(i).mean() for i in v.index),
                           'status':'descriptive; all three classes present in each included scenario'})
save(common.reset_index(),'dial_common_scenario_source.csv')
save(pd.DataFrame(scene_temporal),'dial_common_scenario_sensitivity.csv')

health={'source_scope':str(ROOT),'outside_research_data_reads':[],
        'authority':'User-confirmed final release; no external provenance re-certification',
        'original_inputs_modified':False,'n_subjects':len(sub),'n_pairs':sub.pair_id.nunique(),
        'n_sessions':len(sess),'n_trials_attempted':len(trials),'n_trials_completed':len(completed),
        'n_trials_aborted':len(exclusions),'n_rating_rows':len(r),'n_segments':len(seg),
        'n_scenarios':seg.scenario_id.nunique(),'moment_rows':len(m),'dial_rows':len(dial),
        'duplicate_rating_keys':0,'duplicate_moment_keys':0,'join_validation':'all passed',
        'ego_q5_and_q5b_missing':'596 each, structurally not applicable',
        'segment_conditional_blanks':'exceed_peak_rad absent in 30 W; inside_margin_min_rad absent in 60 A/C',
        'manifest_issues':[x for x in manifest_check if x['status']!='match'],
        'analysis_type':'exploratory secondary extraction, no preregistration claimed',
        'pair_bootstrap_repetitions':B,'seed':SEED,
        'software':{'python':platform.python_version(),'numpy':np.__version__,
                    'pandas':pd.__version__,'scipy':scipy.__version__}}
(OUT/'process'/'integrity_and_methods.json').write_text(json.dumps(health,ensure_ascii=False,indent=2))
print(json.dumps(health,ensure_ascii=False,indent=2))
print(con.to_string(index=False))
