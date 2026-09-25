"""Recompose the matched-arm figure from accepted aggregate measurements."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_style import ROOT, RUN, DATA, HUMAN, NEUTRAL, TEXT, apply_style, panel, export, source_record

BASE = ROOT / '.codex-fleet/rq022-matched-scenario/work/T1_target_figure'
H = json.loads((BASE / 'human_arm_data.json').read_text())
V = json.loads((BASE / 'av_reference_values.json').read_text())
E = json.loads((BASE / 'ego_three_second_window.json').read_text())
REPORT = ROOT / 'reports/knowledge/PAPER001_online_sociality_verification_manuscript/imported_from_paper_repo_20260620/agent_handoff.md'
INK = HUMAN


def write_csv(name, rows):
    DATA.mkdir(exist_ok=True, parents=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (DATA / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def audit_sources():
    assert H['data_status'] == 'REAL_VERIFIED'
    for arm in (H, V):
        for counts in arm['flag_counts'].values():
            assert sum(counts[k] for k in ('n_below', 'n_above', 'n_inside')) == counts['n_total']
        per = arm['per_scenario_alpha90']
        assert len(per) == 15
        assert sum(x['n_both'] for x in per.values()) == arm['flag_counts']['90']['n_total']
        assert sum(x['n_flagged'] for x in per.values()) == sum(arm['flag_counts']['90'][k] for k in ('n_below', 'n_above'))
    assert set(H['per_scenario_alpha90']) == set(V['per_scenario_alpha90'])
    assert '1.95 [1.62, 2.39]' in REPORT.read_text()


def marker(ax, x, y, arm, ci=None, size=5):
    fill = INK if arm == 'AV' else 'white'
    if ci is not None:
        ax.plot(ci, [y, y], color=INK, lw=1.1, zorder=2)
    ax.plot(x, y, 'o' if arm == 'AV' else 's', ms=size, mfc=fill, mec=INK, mew=1, zorder=4)


def signature_rows():
    out = []
    titles = ['Ego TTC: median', 'Ego TTC: upper quartile', 'Counterpart: speed reduction',
              'Counterpart: speed range', 'Ego TTC < 2 s', 'Counterpart braking < −3 m/s²']
    for arm, src in [('AV', V), ('Human', H)]:
        s = src['signature']
        vals = [s['ego_ttc_median']['lower'] / s['ego_ttc_median']['inside'],
                s['ego_ttc_q75']['lower'] / s['ego_ttc_q75']['inside'],
                s['counterpart_speed_drop_ratio']['ratio'],
                s['counterpart_speed_range_ratio']['ratio'],
                s['ego_tail_ttc_lt2']['lower']['share'] / s['ego_tail_ttc_lt2']['inside']['share'],
                s['counterpart_tail_brake_lt_m3']['lower_share'] / s['counterpart_tail_brake_lt_m3']['inside_share']]
        cis = [E['ratios_against_inside']['lower']['q50']['ci95'] if arm == 'AV' else None,
               E['ratios_against_inside']['lower']['q75']['ci95'] if arm == 'AV' else None,
               s['counterpart_speed_drop_ratio']['ci95'], s['counterpart_speed_range_ratio']['ci95'], None, None]
        for i, (name, value, ci) in enumerate(zip(titles, vals, cis)):
            out.append(dict(index=i, endpoint=name, arm=arm, contrast='A/W', ratio=value,
                            ci_low=ci[0] if ci else '', ci_high=ci[1] if ci else '',
                            interval_status='available' if ci else 'unavailable',
                            interval_note='95% run bootstrap' if ci else 'ratio interval unavailable; no interval reconstructed'))
    return out


def main():
    audit_sources(); apply_style()
    rate_rows = []
    for arm, src in [('Human', H), ('AV', V)]:
        for level, c in src['flag_counts'].items():
            for side, key in [('A', 'n_below'), ('C', 'n_above'), ('outside', None)]:
                n = c[key] if key else c['n_below'] + c['n_above']
                has_ci=arm=='Human' and level=='90' and side=='outside'
                rate_rows.append(dict(arm=arm, coverage=int(level), side=side, numerator=n,
                    denominator=c['n_total'], percent=100*n/c['n_total'],
                    ci_low=100*H['flag_rate_ci95_alpha90'][0] if has_ci else '',
                    ci_high=100*H['flag_rate_ci95_alpha90'][1] if has_ci else '',
                    ci_method='driver-by-scenario bootstrap' if has_ci else '',
                    ci_n_boot=1000 if has_ci else '',
                    n_drivers=H['provenance']['n_drivers'] if arm=='Human' else '',
                    n_runs=H['provenance']['n_runs'] if arm=='Human' else '',
                    n_scenarios=len(src['per_scenario_alpha90'])))
    write_csv('fig5_rates.csv', rate_rows)
    write_csv('fig5_natural_reference.csv', [dict(coverage=int(k), outside_percent=v, status='frozen rounded percentage') for k,v in V['natural_human_outside_pct'].items()])
    paired = [dict(scenario=k, human_n=H['per_scenario_alpha90'][k]['n_both'], human_flagged=H['per_scenario_alpha90'][k]['n_flagged'],
                   human_rate=H['per_scenario_alpha90'][k]['rate'], av_n=V['per_scenario_alpha90'][k]['n_both'],
                   av_flagged=V['per_scenario_alpha90'][k]['n_flagged'], av_rate=V['per_scenario_alpha90'][k]['rate'])
              for k in sorted(H['per_scenario_alpha90'])]
    write_csv('fig5_scenarios.csv', paired)
    sig = signature_rows(); write_csv('fig5_consequence_signature.csv', sig)
    write_csv('fig5_rate_ratio.csv', [dict(contrast='AV/Human', estimate=1.95, ci_low=1.62, ci_high=2.39,
        resampling_unit='scenario', repetitions=20000, seed=20260820, source='accepted handoff 2026-08-20, lines 1101-1103', precision='as reported')])
    write_csv('fig5_gates.csv', [dict(arm=arm, **src['gates']) for arm,src in [('Human',H),('AV',V)]])

    fig = plt.figure(figsize=(180/25.4, 163/25.4))
    gs = fig.add_gridspec(2, 2, left=.105, right=.98, top=.88, bottom=.105, wspace=.43, hspace=.51)
    axa, axb, axc, axd = [fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
    fig.text(.105,.985,'20 drivers × 15 scenarios  ·  same frozen 90% human reference',va='top',fontsize=8)
    fig.text(.105,.956,'Judgeable moments: 15,598 human / 14,099 AV',va='top',fontsize=7.5,color=TEXT)

    values = [V['natural_human_outside_pct']['90'],100*786/15598,100*1388/14099]
    for i, val in enumerate(values):
        if i == 0: axa.bar(i,val,width=.52,color=NEUTRAL,alpha=.65)
        else: axa.bar(i,val,width=.52,color='white' if i==1 else INK,edgecolor=INK,lw=1.2)
        axa.text(i,val+.6,f'{val:.2f}%',ha='center',fontsize=8)
    human_ci=100*np.asarray(H['flag_rate_ci95_alpha90'])
    axa.errorbar(1,values[1],yerr=np.array([[values[1]-human_ci[0]],[human_ci[1]-values[1]]]),fmt='none',ecolor=INK,capsize=3,lw=1)
    axa.axhline(10,color=NEUTRAL,ls='--',lw=.8)
    axa.set(ylim=(0,14.7),xticks=[0,1,2],xticklabels=['Natural\nreference','Matched\nhuman','Matched\nAV'],ylabel='Outside-range moments (%)')
    axa.text(.02,.97,'AV / human: 1.95 [1.62, 2.39]',transform=axa.transAxes,va='top',fontsize=8)
    axa.text(.01,-.29,'786/15,598 human; 1,388/14,099 AV',transform=axa.transAxes,fontsize=7)
    panel(axa,'a','Overall departures')

    for i, side in enumerate(['A','C']):
        for arm,shift in [('AV',-.14),('Human',.14)]:
            row=next(r for r in rate_rows if r['arm']==arm and r['coverage']==90 and r['side']==side)
            axb.plot(i+shift,row['percent'],'o' if arm=='AV' else 's',mfc=INK if arm=='AV' else 'white',mec=INK,ms=6)
            axb.text(i+shift,row['percent']+.45,f"{row['percent']:.2f}",ha='center',fontsize=7.5)
    axb.set(xlim=(-.5,1.5),ylim=(0,8.1),xticks=[0,1],xticklabels=['Assertive side','Accommodating side'],ylabel='Judgeable moments (%)')
    axb.text(.02,.96,'Point estimates',transform=axb.transAxes,va='top',fontsize=7,color=TEXT)
    panel(axb,'b','Departure direction')

    axc.plot([0,19],[0,19],ls='--',color=NEUTRAL,lw=.8)
    axc.scatter([100*r['human_rate'] for r in paired],[100*r['av_rate'] for r in paired],s=28,color=INK)
    axc.set(xlim=(0,19),ylim=(0,19),xticks=[0,5,10,15],yticks=[0,5,10,15],xlabel='Human outside-range moments (%)',ylabel='AV outside-range moments (%)')
    axc.set_aspect('equal',adjustable='box')
    axc.text(.98,.03,'15/15 above equality\n1 point = 1 scenario',transform=axc.transAxes,ha='right',va='bottom',fontsize=7.5)
    panel(axc,'c','Matched scenarios')

    for i in range(6):
        rows=[r for r in sig if r['index']==i]; y=(5-i)*1.7
        axd.text(.255,y+.54,rows[0]['endpoint'],ha='left',va='bottom',fontsize=7.4)
        for r in rows:
            yy=y+(.13 if r['arm']=='AV' else -.13)
            ci=[r['ci_low'],r['ci_high']] if r['ci_low']!='' else None
            marker(axd,r['ratio'],yy,r['arm'],ci,size=4.2)
            if ci is None: axd.annotate('†',(r['ratio'],yy),xytext=(4,0),textcoords='offset points',fontsize=7,va='center')
    axd.axvline(1,color=NEUTRAL,ls='--',lw=.8,zorder=0)
    axd.set(xscale='log',xlim=(.25,5),ylim=(-.5,9.7),yticks=[],xticks=[.5,1,2,4],xticklabels=['0.5','1','2','4'],xlabel='Assertive / within-range ratio')
    axd.spines['left'].set_visible(False)
    panel(axd,'d','Shared consequence pattern')
    handles=[Line2D([],[],marker='o',ls='',color=INK,mfc=INK,label='AV'),Line2D([],[],marker='s',ls='',color=INK,mfc='white',label='Human')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.09,.01),ncol=2,handletextpad=.5,columnspacing=1.5)
    fig.text(.46,.038,'Whiskers: 95% CI. † Ratio interval unavailable.',fontsize=7.3)
    fig.text(.46,.017,'Natural reference is a calibration check.',fontsize=7.3)
    export(fig,'fig5_human_arm')

    fig,ax=plt.subplots(figsize=(180/25.4,70/25.4)); fig.subplots_adjust(left=.12,right=.98,bottom=.23,top=.85)
    x=np.arange(3)
    for arm,shift in [('Human',-.12),('AV',.12)]:
        ys=[next(r['percent'] for r in rate_rows if r['arm']==arm and r['coverage']==lev and r['side']=='outside') for lev in [80,90,95]]
        ax.plot(x+shift,ys,'s' if arm=='Human' else 'o',mfc='white' if arm=='Human' else INK,mec=INK,ms=6,label=arm)
    ax.plot(x,[V['natural_human_outside_pct'][str(k)] for k in [80,90,95]],'D',color=NEUTRAL,ms=5,label='Natural reference')
    ax.set(xticks=x,xticklabels=['80%','90%','95%'],xlabel='Nominal reference coverage',ylabel='Outside-range moments (%)',ylim=(0,24))
    ax.legend(ncol=3,loc='upper right'); ax.set_title('Matched-arm departures at all frozen coverage levels')
    export(fig,'figS3_human_levels')

    sources=[source_record('5','a/b/c/d',BASE/'human_arm_data.json','flag_counts; per_scenario_alpha90; signature; gates','aggregate counts and frozen statistics','REAL_VERIFIED; 90% main point','Human row-level archive not available locally; no recomputation'),
             source_record('5','a/b/c/d',BASE/'av_reference_values.json','flag_counts; natural_human_outside_pct; per_scenario_alpha90; signature','aggregate counts and frozen statistics','90% main point'),
             source_record('5','d',BASE/'ego_three_second_window.json','ratios_against_inside.lower.q50/q75.ci95','ratio','fixed 3-second window; 2000 whole-run bootstrap'),
             source_record('5','a',REPORT,'lines 1101-1103: 1.95 [1.62,2.39]; 20000 draws; seed 20260820','reported ratio and CI','accepted 2026-08-20 handoff','Reported CI annotation only, not reconstructed from aggregate arm intervals')]
    sources.append(source_record('5','a',ROOT/'reports/knowledge/RQ022_matched_scenario_human_arm/decision.md','Accepted endpoint 1: rate CI 3.4–5.5%, driver-by-scenario bootstrap B1000','human outside-rate CI','20 drivers × 15 scenarios; 300 runs','CI method differs from AV/human scenario bootstrap'))
    path=RUN/'process/fig45_sources.csv'
    existing=[]
    if path.exists():
        with path.open(newline='') as f: existing=[r for r in csv.DictReader(f) if r['figure']!='5']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sources[0]));w.writeheader();w.writerows(existing+sources)
    print('Figure 5 and Supplementary Figure S3 exported; sources checked.')


if __name__=='__main__': main()
