"""Draw frozen three-second consequences, retaining both departure directions."""
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_style import ROOT, RUN, DATA, A, C, W, NEUTRAL, COLORS, apply_style, panel, export, source_record

BASE = ROOT / '.codex-fleet/rq022-matched-scenario/work/T1_target_figure'
RQ21 = ROOT / 'reports/studies/RQ021_contemporaneous_envelope/RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff'
E = json.loads((BASE/'ego_three_second_window.json').read_text())
S = json.loads((BASE/'accommodating_side_battery.json').read_text())
B = json.loads((RQ21/'rq019_rerun/distribution_results.json').read_text())
MAP = {'A':'lower','C':'upper','W':'inside'}


def write_csv(name, rows):
    DATA.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (DATA/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def forest(ax, rows, labels, limits, xlabel, ratio=False):
    ax.axvline(1 if ratio else 0,color=NEUTRAL,ls='--',lw=.8,zorder=0)
    for r in rows:
        y=len(labels)-1-r['index']+(.13 if r['side']=='A' else -.13)
        ax.plot([r['ci_low'],r['ci_high']],[y,y],color=COLORS[r['side']],lw=1.25)
        ax.plot(r['estimate'],y,'o' if r['side']=='A' else '^',color=COLORS[r['side']],ms=5,mec='white',mew=.35,zorder=3)
    ax.set(yticks=np.arange(len(labels))[::-1],yticklabels=labels,xlim=limits,ylim=(-.6,len(labels)-.4),xlabel=xlabel)
    ax.tick_params(axis='y',length=0,pad=6);ax.spines['left'].set_visible(False)
    if ratio:
        ax.set_xscale('log');ax.set_xticks([.25,.5,1,2,4]);ax.set_xticklabels(['0.25','0.5','1','2','4'])


def main():
    apply_style()
    assert E['analysis_counts']=={'inside':12344,'lower':499,'upper':817}
    ttc=[]
    for side,band in MAP.items():
        vals=E['ego_ttc_values_sorted'][band]
        assert len(vals)==E['analysis_counts'][band] and vals==sorted(vals)
        assert np.isfinite(vals).all()
        ttc += [dict(side=side,sorted_position=i+1,ttc_s=v,cumulative_fraction=(i+1)/len(vals),n_moments=len(vals)) for i,v in enumerate(vals)]
    write_csv('fig4_ttc_distribution.csv',ttc)
    quant=[]
    for side,band in MAP.items():
        for q,v in E['ego_ttc_quantiles'][band]['q'].items():
            quant.append(dict(side=side,quantile_percent=q,ttc_s=v))
    write_csv('fig4_ttc_quantiles.csv',quant)
    write_csv('fig4_ttc_ratio_intervals.csv',[dict(side=side,quantile=q,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))},ci_low=r['ci95'][0],ci_high=r['ci95'][1])
        for side,band in [('A','lower'),('C','upper')] for q,r in E['ratios_against_inside'][band].items()])

    tails=[]
    keys=['ttc_lt_1.0','ttc_lt_1.5','ttc_lt_2.0','ttc_lt_3.0']
    for i,key in enumerate(keys):
        for side,band in [('A','lower'),('C','upper')]:
            r=E['ego_danger_bootstrap'][key][band];sh=E['ego_danger_shares'][key]
            tails.append(dict(index=i,side=side,threshold_s=float(key.split('_')[-1]),
                estimate=100*r['observed_diff_comparator_minus_inside'],ci_low=100*r['case_bootstrap_ci95'][0],ci_high=100*r['case_bootstrap_ci95'][1],
                side_numerator=sh[band]['numerator'],side_denominator=sh[band]['denominator'],side_share=sh[band]['share'],
                within_numerator=sh['inside']['numerator'],within_denominator=sh['inside']['denominator'],within_share=sh['inside']['share'],
                unit='percentage points',n_cases=r['n_cases'],n_boot=r['n_boot']))
    write_csv('fig4_ego_tail_differences.csv',tails)

    speeds=[]; all_cp=[]
    full_keys=['counterpart_speed_drop_ratio','counterpart_speed_range_ratio','counterpart_heading_change_ratio','counterpart_yaw_rate_ratio']
    for i,key in enumerate(full_keys):
        for side,band in [('A','lower'),('C','upper')]:
            r=S['battery'][key][band]
            row=dict(index=i,side=side,outcome=key,estimate=r['ratio'],ci_low=r['ci95'][0],ci_high=r['ci95'][1],
                median_side=r['median_'+band],median_within=r['median_inside'],n_side=r['n_'+band],n_within=r['n_inside'],
                n_cases=r['n_cases'],n_boot=r['n_boot'],estimand='pooled moment-weighted median ratio',
                inference='whole-run bootstrap',extension_status='post-hoc extension to accommodating side' if side=='C' else 'frozen assertive battery')
            all_cp.append(row)
            if i<2:speeds.append(row)
    write_csv('fig4_counterpart_all_ratios.csv',all_cp)
    write_csv('fig4_counterpart_speed_ratios.csv',speeds)

    brakes=[]
    raw=[r for r in B['alpha90_case_bootstrap_threshold_contrasts'] if r['window']=='fixed3' and r['stratum']=='non_scripted']
    assert len(raw)==6
    for r in raw:
        side='A' if r['comparison']=='lower_minus_inside' else 'C'
        brakes.append(dict(index=abs(r['threshold_mps2'])-2,side=side,estimate=100*r['share_difference'],
            ci_low=100*r['case_bootstrap_ci_95'][0],ci_high=100*r['case_bootstrap_ci_95'][1],unit='percentage points',**r))
    assert {r['comparison_denominator'] for r in brakes if r['side']=='A'}=={13800}
    assert {r['comparison_denominator'] for r in brakes if r['side']=='C'}=={21025}
    assert {r['inside_denominator'] for r in brakes}=={310246}
    write_csv('fig4_counterpart_braking.csv',brakes)

    fig=plt.figure(figsize=(180/25.4,153/25.4))
    gs=fig.add_gridspec(2,2,left=.105,right=.975,bottom=.115,top=.88,wspace=.54,hspace=.57)
    axa,axb,axc,axd=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
    fig.text(.105,.983,'Frozen 3 s post-verdict windows  ·  90% situation-conditioned reference',va='top',fontsize=8)
    legend=[Line2D([],[],color=A,marker='o',ms=4,label='A  Assertive side'),Line2D([],[],color=C,marker='^',ms=4,label='C  Accommodating side'),Line2D([],[],color=W,ls='--',label='W  Within range')]
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.09,.96),ncol=3,handlelength=1.5,columnspacing=1.6)

    for side in ['A','C','W']:
        vals=np.asarray(E['ego_ttc_values_sorted'][MAP[side]])
        axa.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=COLORS[side],ls={'A':'-','C':':','W':'--'}[side],lw=1.15)
    axa.set(xscale='log',xlim=(.2,1e6),ylim=(0,1.04),xlabel='Minimum TTC over 3 s (s, log scale)',ylabel='Cumulative fraction')
    axa.set_xticks([1,10,100,1e4,1e6]);axa.set_xticklabels(['1','10','100',r'$10^{4}$',r'$10^{6}$'])
    for q in [.25,.5,.75]:axa.axhline(q,color=NEUTRAL,lw=.4,ls=':',alpha=.6,zorder=0)
    axa.text(.43,.42,'A versus W\nLower quartile: 3.46 / 2.99 s\nMedian ratio: 0.90 [0.76, 1.11]\nUpper quartile: −39.9%\n[−56.4, −16.9]%',transform=axa.transAxes,fontsize=7.2,va='top',linespacing=1.3)
    axa.text(.99,.035,'A 499  /  C 817  /  W 12,344',transform=axa.transAxes,ha='right',va='bottom',fontsize=7)
    panel(axa,'a','Ego interaction margin')

    forest(axb,tails,['< 1 s','< 1.5 s','< 2 s','< 3 s'],(-13,3),'Difference from W (percentage points)')
    axb.set_xticks([-10,-5,0]);panel(axb,'b','Ego emergency tails')
    axb.text(.02,.98,'A − W  /  C − W',transform=axb.transAxes,va='top',fontsize=7.5)

    forest(axc,speeds,['Speed\nreduction','Speed\nrange'],(.2,4.5),'Median response / W (ratio)',ratio=True)
    panel(axc,'c','Counterpart speed response')
    axc.text(.02,.98,'A 469  /  C 719  /  W 10,483 moments',transform=axc.transAxes,va='top',fontsize=7)
    axc.text(.02,.025,'95% CI over 175 runs; 2,000 draws',transform=axc.transAxes,va='bottom',fontsize=7)

    forest(axd,brakes,['< −2 m/s²','< −3 m/s²','< −4 m/s²'],(-7.5,3),'Difference from W (percentage points)')
    axd.set_xticks([-6,-3,0,3]);panel(axd,'d','Counterpart emergency braking')
    axd.text(.02,.98,'Acceleration records: A 13,800\nC 21,025  /  W 310,246',transform=axd.transAxes,va='top',fontsize=7)
    fig.text(.105,.035,'Whiskers: frozen 95% CIs over whole runs. Thresholds are nested.',fontsize=7.5)
    fig.text(.105,.014,'b: 227 runs, 1,000 draws. d: 1,000 draws; record-weighted rates.',fontsize=7.5)
    export(fig,'fig4_consequence')

    sources=[source_record('4','a/b',BASE/'ego_three_second_window.json','ego_ttc_values_sorted; ego_ttc_quantiles; ego_danger_shares; ego_danger_bootstrap; ratios_against_inside','moments and frozen statistics','fixed 3-second window; alpha90; A499/C817/W12344','Full sorted data exported; no new statistics or tests'),
             source_record('4','c',BASE/'accommodating_side_battery.json','battery.counterpart_speed_drop_ratio; battery.counterpart_speed_range_ratio','moment-weighted median ratios','fixed3; A469/C719/W10483; whole-run bootstrap B2000','Only counterpart fields used; old ego fields deliberately excluded'),
             source_record('4','d',RQ21/'rq019_rerun/distribution_results.json','alpha90_case_bootstrap_threshold_contrasts','acceleration records','window=fixed3; stratum=non_scripted; both comparisons; thresholds -2,-3,-4','A13800/C21025/W310246; run bootstrap B1000')]
    path=RUN/'process/fig45_sources.csv'
    with path.open(newline='') as f: existing=list(csv.DictReader(f))
    existing=[r for r in existing if r['figure']!='4']+sources
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sources[0]));w.writeheader();w.writerows(existing)
    print('Figure 4 exported; both directions and all frozen thresholds retained.')


if __name__=='__main__': main()
