"""Render frozen monitor metrics and distinct source-transfer diagnostics."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from figure_style import ROOT,RUN,DATA,W,TEXT,HUMAN,ABSTAIN,apply_style,panel,export,source_record

RQ=ROOT/'reports/studies/RQ021_contemporaneous_envelope'
KEY=RQ/'RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff/key_numbers.json'
TRANSFER=RQ/'RQ021_2_lodo_transfer_20260807T114305Z_0c4d280/key_numbers_e2.json'
FUNNEL=ROOT/'.codex-fleet/paper-figure-upgrade/work/S1_scoring/fig1_three_layer_data.json'
PREDICT=ROOT/'reports/studies/RQ004_ipv_state_space/RQ004_1_state_space_law_nature_20260618/02_process/agent_I_figures/F7_lodo_summary_source_data.csv'
NAVY='#263850'


def load():
    k=json.loads(KEY.read_text())['human_only_envelope'];f=json.loads(FUNNEL.read_text())['funnel']
    rows=[]
    for level in ['80','90','95']:
        for kind,m in [('Global',k['circularity_diagnostics']['marginal_envelopes']['ipv_log']['metrics'][level]),('Conditioned',k['metrics'][level])]:
            assert m['n']==461937 and m['total_n']==486660
            rows.append(dict(level=int(level),kind=kind,mean_width_rad=m['mean_width'],width_candidate_span_pct=100*m['mean_width']/(3*np.pi/4),coverage=m['coverage'],coverage_minus_nominal_pp=100*(m['coverage']-int(level)/100),covered_n=m['covered_n'],accepted_n=m['n'],readable_n=m['total_n']))
    d=pd.DataFrame(rows);d.to_csv(DATA/'fig3ab_width_coverage.csv',index=False)
    r=k['circularity_diagnostics']['D2_contemporaneous_test_r2']
    assert r['rows']==486660 and np.isclose(r['r2'],1-r['sse']/r['sst'])
    assert f['candidate']==663282 and f['readable']==486660 and f['judgeable']==461937
    assert f['inside_90']==k['metrics']['90']['covered_n'] and f['inside_90']+f['outside_90']==f['judgeable']
    assert f['readable']-f['judgeable']==24723==f['unsupported']
    pd.DataFrame([dict(stage=a,count=f[a],universe='pure-human naturalistic test fold') for a in ['candidate','readable','judgeable','inside_90','outside_90','near_uniform','no_ipv_effect','solver_failure','unsupported']]).to_csv(DATA/'fig3c_gates.csv',index=False)
    pd.DataFrame([r]).to_csv(DATA/'fig3d_readable_r2.csv',index=False)
    return d,f,r


def width(ax,d):
    x=np.arange(3);g=d[d.kind.eq('Global')].reset_index(drop=True);c=d[d.kind.eq('Conditioned')].reset_index(drop=True)
    ax.bar(x-.18,g.width_candidate_span_pct,.34,color='white',edgecolor=HUMAN,hatch='///',lw=.8,label='Global')
    ax.bar(x+.18,c.width_candidate_span_pct,.34,color=W,label='Conditioned')
    ax.axhline(100,color=NAVY,lw=.7,ls=':')
    for j in x:
        reduction=100*(1-c.mean_width_rad.iloc[j]/g.mean_width_rad.iloc[j])
        ax.text(j,max(g.width_candidate_span_pct.iloc[j],c.width_candidate_span_pct.iloc[j])+5,f'−{reduction:.0f}%',ha='center',fontsize=8,color=W,fontweight='bold')
    ax.set_ylim(0,120);ax.set_yticks([0,25,50,75,100]);ax.set_xticks(x,['80%','90%','95%'])
    ax.set_ylabel('Mean width (% of candidate span)');ax.set_xlabel('Nominal coverage')

    panel(ax,'a','Narrower conditional ranges')


def coverage(ax,d):
    x=np.arange(3)
    ax.axhline(0,color=NAVY,lw=.7,ls=':')
    for kind,off,col,marker,fill in [('Global',-.075,HUMAN,'o','white'),('Conditioned',.075,W,'s',W)]:
        y=d[d.kind.eq(kind)].coverage_minus_nominal_pp.to_numpy()
        ax.scatter(x+off,y,facecolor=fill,edgecolor=col,marker=marker,s=25,zorder=3,label=kind)
        for j,v in enumerate(y):
            dx,dy=(-7,-14) if kind=='Global' and j==1 else ((7,7) if kind=='Conditioned' and j==1 else (0,7 if kind=='Global' else -14))
            ax.annotate(f'{v:+.2f}',(j+off,v),xytext=(dx,dy),textcoords='offset points',ha='center',fontsize=7,color=col)
    ax.set_ylim(-.6,3.1);ax.set_yticks([0,1,2,3]);ax.set_xticks(x,['80%','90%','95%']);ax.set_xlim(-.4,2.4)
    ax.set_xlabel('Nominal coverage');ax.set_ylabel('Empirical − nominal coverage (pp)')
    ax.legend(loc='upper left',fontsize=7,handlelength=1)
    panel(ax,'b','Empirical marginal coverage')


def gates(ax,f):
    ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    panel(ax,'c','Gates delimit when a verdict is available')
    for y,key,label,color in [(.83,'candidate','Candidate moments',HUMAN),(.61,'readable','Readable moments',HUMAN),(.39,'judgeable','Also supported',W)]:
        width=.50*f[key]/f['candidate'];ax.add_patch(Rectangle((0,y),width,.13,facecolor=color,alpha=.7,edgecolor=color))
        ax.text(.54,y+.065,f'{f[key]:,}\n{label}',va='center',fontsize=7.4,linespacing=1.15)
    w=.50*f['judgeable']/f['candidate'];i=w*f['inside_90']/f['judgeable']
    ax.add_patch(Rectangle((0,.16),i,.13,facecolor=W))
    ax.add_patch(Rectangle((i,.16),w-i,.13,facecolor='white',edgecolor=NAVY,hatch='////',lw=.7))
    ax.text(.54,.22,f'{f["inside_90"]:,} inside\n{f["outside_90"]:,} outside',va='center',fontsize=7.4,linespacing=1.2)
    ax.text(0,-.03,f'Support abstention: {f["unsupported"]:,} / {f["readable"]:,} = {100*f["unsupported"]/f["readable"]:.2f}%',fontsize=7)


def variance(ax,r):
    p=100*r['r2']
    ax.barh([.6],[p],height=.28,color=HUMAN)
    ax.barh([.6],[100-p],left=[p],height=.28,color='white',edgecolor=NAVY,hatch='///',lw=.7)
    ax.text(p/2,.6,f'{p:.1f}%',ha='center',va='center',fontsize=9,color='white',fontweight='bold')
    ax.text(p+(100-p)/2,.6,f'{100-p:.1f}%',ha='center',va='center',fontsize=9,color=NAVY,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=1))
    ax.text(p/2,.34,'Explained',ha='center',fontsize=7);ax.text(p+(100-p)/2,.34,'Residual',ha='center',fontsize=7)
    ax.set_xlim(0,100);ax.set_ylim(0,1);ax.set_yticks([]);ax.set_xticks([0,25,50,75,100]);ax.spines['left'].set_visible(False)
    ax.set_xlabel('Variance of the IPV reading (%)')
    ax.text(0,.94,'All readable moments: n = 486,660',fontsize=7)
    panel(ax,'d','Situation-linked variation')


def transfer():
    pred=pd.read_csv(PREDICT);pred=pred[pred.outcome_spec.eq('case_mean_ipv')].set_index('holdout_dataset').loc[['Waymo','nuPlan','Lyft','AV2']].reset_index()
    pred.to_csv(DATA/'figED3a_source_prediction.csv',index=False)
    raw=json.loads(TRANSFER.read_text());rows=[]
    names=[('waymo_train','Waymo'),('nuplan_train','nuPlan'),('lyft_train_full','Lyft'),('av2_motion_forecasting','AV2')]
    for key,label in names:
        r=raw['lodo'][key]['metrics']['90'];assert np.isclose(r['coverage'],r['covered_n']/r['n']) and np.isclose(r['abstention'],r['abstained_n']/r['total_n'])
        rows.append(dict(source=label,source_key=key,coverage_pct=100*r['coverage'],abstention_pct=100*r['abstention'],covered_n=r['covered_n'],accepted_n=r['n'],abstained_n=r['abstained_n'],readable_n=r['total_n'],nominal_pct=90))
    d=pd.DataFrame(rows);d.to_csv(DATA/'figED3b_source_reference.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3));fig.subplots_adjust(left=.10,right=.97,bottom=.25,top=.81,wspace=.47)
    a,b=axes;y=np.arange(4)[::-1]
    a.axvline(0,color=NAVY,lw=.7,ls=':')
    for yy,(_,r) in zip(y,pred.iterrows()):
        a.plot([0,r.full_state_space_r2],[yy,yy],color=HUMAN,lw=1)
        a.scatter([r.full_state_space_r2],[yy],color=HUMAN,s=26)
        a.annotate(f'{r.full_state_space_r2:+.3f}',(r.full_state_space_r2,yy),xytext=(-5 if r.full_state_space_r2<0 else 5,0),textcoords='offset points',ha='right' if r.full_state_space_r2<0 else 'left',va='center',fontsize=7)
    a.set_xlim(-.40,.14);a.set_ylim(-.55,3.6);a.set_yticks(y,pred.holdout_dataset);a.set_xticks([-.4,-.2,0]);a.set_xlabel('Held-out-source prediction R²');a.spines['left'].set_visible(False)
    panel(a,'a','Episode-summary prediction')
    a.text(0,-.29,'Case-mean IPV target · point estimates',transform=a.transAxes,fontsize=6.7)
    b.axvline(90,color=W,lw=.7,ls=':')
    for yy,(_,r) in zip(y,d.iterrows()):
        b.scatter(r.coverage_pct,yy+.10,s=26,color=W,marker='o')
        b.scatter(r.abstention_pct,yy-.10,s=26,color='#9AA0A6',marker='s')
        for val,off in [(r.coverage_pct,.10),(r.abstention_pct,-.10)]:
            b.annotate(f'{val:.1f}',(val,yy+off),xytext=(4,0),textcoords='offset points',ha='left',va='center',fontsize=6.6)
    b.set_xlim(0,112);b.set_ylim(-.55,4.1);b.set_yticks(y,d.source);b.set_xticks([0,25,50,75,100]);b.set_xlabel('Held-out-source moments (%)');b.spines['left'].set_visible(False)
    panel(b,'b','90% reference under source transfer')
    b.scatter([],[],s=22,c=W,label='Coverage',marker='o');b.scatter([],[],s=22,c='#9AA0A6',label='Support abstention',marker='s')
    b.legend(loc='upper left',bbox_to_anchor=(-.02,1.01),ncol=2,fontsize=6.7,handletextpad=.3,columnspacing=.9)
    b.text(0,-.29,'Coverage / accepted; abstention / readable',transform=b.transAxes,fontsize=6.7)
    export(fig,'figED3_source_transfer')
    return rows


def main():
    apply_style();DATA.mkdir(exist_ok=True)
    d,f,r=load();fig=plt.figure(figsize=(7.2,5.35))
    width(fig.add_axes([.105,.59,.345,.29]),d);coverage(fig.add_axes([.62,.59,.34,.29]),d)
    gates(fig.add_axes([.085,.105,.43,.285]),f);variance(fig.add_axes([.655,.115,.30,.275]),r)
    fig.text(.105,.49,'a–b: n = 461,937 accepted moments; candidate span = 3π/4 rad; 90% conditional mean = 1.87 rad.',fontsize=6.7)
    fig.text(.105,.463,'All panels show frozen point estimates; no uncertainty intervals are shown.',fontsize=6.7)
    export(fig,'fig3_monitor');tr=transfer()
    rec=[source_record('3','a,b,d',KEY,'human_only_envelope.metrics; circularity_diagnostics.marginal_envelopes.ipv_log.metrics; D2_contemporaneous_test_r2','accepted or readable test moments','pure-human test-fold; a,b supported and readable; d all readable','Accepted RQ021 outputs; distinct denominators preserved'),source_record('3','c',FUNNEL,'funnel','candidate test moments','pure-human test-fold universe','Support abstention denominator is readable moments'),source_record('ED3','a',PREDICT,'outcome_spec; full_state_space_r2; test_n','independent cases','outcome_spec=case_mean_ipv','Episode-summary prediction, separate from reference coverage'),source_record('ED3','b',TRANSFER,'lodo.<source>.metrics.90','moments','all four held-out sources; 90% level','Coverage denominator accepted; support abstention denominator readable')]
    fp=RUN/'process/fig13_sources.csv';old=pd.read_csv(fp);old=old[~old.figure.astype(str).isin(['3','ED3'])];pd.concat([old,pd.DataFrame(rec)]).to_csv(fp,index=False)
    qp=RUN/'process/fig13_qa.json';q=json.loads(qp.read_text());q.update(fig3_numeric_checks=True,fig3_width_reductions_pct=[100*(1-d[(d.kind=='Conditioned')&(d.level==l)].mean_width_rad.iloc[0]/d[(d.kind=='Global')&(d.level==l)].mean_width_rad.iloc[0]) for l in [80,90,95]],fig3_r2=r['r2'],fig3_abstention_fraction=f['unsupported']/f['readable'],source_transfer_all_four_sources=True);qp.write_text(json.dumps(q,indent=2));print(json.dumps(q))

if __name__=='__main__':main()
