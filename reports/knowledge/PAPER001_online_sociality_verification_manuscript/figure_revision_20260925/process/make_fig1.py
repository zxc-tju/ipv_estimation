"""Recompose the existing monitored case and migrate its measurement context."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Arc
from figure_style import ROOT, RUN, DATA, ASSETS, A, C, W, ABSTAIN, TEXT, HUMAN, apply_style, panel, export, source_record

S1 = ROOT / '.codex-fleet/paper-figure-upgrade/work/S1_scoring'
CASE = S1 / 'all_candidates_scored.parquet'
PAYLOAD = S1 / 'fig1_three_layer_data.json'
DEPLOYED = ROOT / '.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet'
CASE_KEY = 'ipv_004992'
TIMES = (13.2, 19.0, 22.0)
NAVY = '#263850'


def setup():
    apply_style()
    DATA.mkdir(exist_ok=True)
    d = pd.read_parquet(CASE, filters=[('case_key', '==', CASE_KEY), ('perspective', '==', 'key_agent_1')]).sort_values('elapsed_time_s').reset_index(drop=True)
    assert len(d) == 221 and d.anchor_frame_index.diff().dropna().eq(1).all()
    assert d.geometry_path_category.eq('CP').all()
    assert d.both_gates_ok.eq(d.status.eq('OK') & d.mechanism2_gate_ok).all()
    d['plot_reading_rad'] = d.ipv_log.where(d.both_gates_ok)
    d['plot_lower_rad'] = d.lo_90.where(d.both_gates_ok)
    d['plot_upper_rad'] = d.hi_90.where(d.both_gates_ok)
    d['plot_state'] = d.exceed_side_90.map({'lower':'A','upper':'C','inside':'W'}).fillna('abstain')
    assert d.loc[~d.both_gates_ok, ['plot_reading_rad','plot_lower_rad','plot_upper_rad']].isna().all().all()
    assert d.loc[d.plot_state.eq('A'), 'ipv_log'].lt(d.loc[d.plot_state.eq('A'),'lo_90']).all()
    assert d.loc[d.plot_state.eq('C'), 'ipv_log'].gt(d.loc[d.plot_state.eq('C'),'hi_90']).all()
    return d, json.loads(PAYLOAD.read_text())


def box(ax, x, y, w, h, label, edge=NAVY, fill='white', fs=7.1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.012',lw=.75,ec=edge,fc=fill))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,linespacing=1.25)


def arrow(ax,start,end,color=NAVY):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=.85,color=color,shrinkA=2,shrinkB=2))


def mechanism(ax):
    ax.axis('off'); ax.set_xlim(0,1);ax.set_ylim(0,1)
    panel(ax,'a','Two inputs, two evidence gates, one comparison')
    ax.texts[-1].set_x(-.05)
    for y, inp, value, gate in [(.68,'Causal trajectory\nwindow','IPV reading','Readable?'),(.32,'Observable\nsituation','Human reference\nrange','Supported?')]:
        box(ax,.005,y,.16,.20,inp)
        box(ax,.21,y,.18,.20,value)
        box(ax,.435,y,.14,.20,gate)
        arrow(ax,(.165,y+.1),(.21,y+.1));arrow(ax,(.39,y+.1),(.435,y+.1))
        arrow(ax,(.575,y+.1),(.62,.60))
    box(ax,.62,.48,.14,.24,'Compare\nreading with\n90% range')
    ax.text(.595,.88,'pass',ha='center',fontsize=6.6)
    ax.text(.595,.30,'pass',ha='center',fontsize=6.6)
    for y, label, col in [(.80,'A · assertive side',A),(.57,'C · accommodating side',C),(.34,'W · inside range',W)]:
        box(ax,.81,y,.185,.16,label,edge=col,fill=col+'20',fs=6.7)
        arrow(ax,(.76,.60),(.81,y+.08))
    box(ax,.81,.045,.185,.16,'Abstain · no verdict',edge='#9AA0A6',fill=ABSTAIN,fs=6.7)
    ax.plot([.505,.505,.79],[.32,.125,.125],color='#9AA0A6',lw=.8)
    ax.plot([.46,.41,.41],[.68,.61,.44],color='#9AA0A6',lw=.8)
    ax.plot([.41,.41,.505],[.40,.125,.125],color='#9AA0A6',lw=.8)
    arrow(ax,(.79,.125),(.81,.125),color='#9AA0A6')
    ax.text(.63,.08,'either gate fails',ha='center',va='center',fontsize=6.6)
    ax.text(.005,.04,'Later evaluation: objective interaction outcomes and subjective ratings',fontsize=6.5,color=NAVY)


def trajectories(ax,d):
    t=d.elapsed_time_s.to_numpy(float)
    e=d[['ego_px_anchor','ego_py_anchor']].to_numpy(float)
    o=d[['counterpart_px_anchor','counterpart_py_anchor']].to_numpy(float)
    dist=np.linalg.norm(e[:,None,:]-o[None,:,:],axis=2)
    i,j=np.unravel_index(dist.argmin(),dist.shape);origin=(e[i]+o[j])/2
    e=e-origin;o=o-origin;shown=t>=13.2
    ax.plot(e[shown,0],e[shown,1],color=NAVY,label='Monitored vehicle')
    ax.plot(o[shown,0],o[shown,1],color=HUMAN,ls='--',label='Counterpart')
    ax.add_patch(Circle((0,0),5,facecolor='#E9DDBD',edgecolor='#887540',lw=.75,alpha=.75))
    ax.annotate('Conflict zone',xy=(3,2),xytext=(17,12),fontsize=7,arrowprops=dict(arrowstyle='-',lw=.7),ha='center')
    offsets=[(-4,-13),(-10,-10),(0,10)]
    for tm,off in zip(TIMES,offsets):
        k=np.abs(t-tm).argmin()
        ax.scatter(*e[k],s=23,marker='o',facecolor='white',edgecolor=NAVY,zorder=5)
        ax.scatter(*o[k],s=23,marker='s',facecolor='white',edgecolor=HUMAN,zorder=5)
        ax.annotate(f'{tm:.1f} s',xy=e[k],xytext=off,textcoords='offset points',fontsize=6.6,ha='right' if off[0]<0 else 'center')
    for p,c in [(e[shown],NAVY),(o[shown],HUMAN)]:
        ax.annotate('',xy=p[-1],xytext=p[-4],arrowprops=dict(arrowstyle='-|>',lw=.9,color=c))
    points=np.vstack([e[shown],o[shown]]);lo=points.min(0);hi=points.max(0);pad=(hi-lo)*.16
    ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
    ax.set_aspect('equal');ax.set_xlabel('Local x position (m)');ax.set_ylabel('Local y position (m)')
    ax.legend(loc='upper right',fontsize=6.5,handlelength=1.8,borderpad=0)
    ax.figure.text(.069,.585,'b',fontweight='bold',fontsize=11)
    ax.figure.text(.113,.585,'One observed interaction',fontweight='bold',fontsize=9)
    out=d[['case_key','perspective','elapsed_time_s','ego_px_anchor','ego_py_anchor','counterpart_px_anchor','counterpart_py_anchor']].copy()
    out[['ego_local_x_m','ego_local_y_m']]=e;out[['counterpart_local_x_m','counterpart_local_y_m']]=o
    out['shown']=shown;out['origin_x_m']=origin[0];out['origin_y_m']=origin[1]
    out.to_csv(DATA/'fig1b_trajectory.csv',index=False)


def timeline(ax,d):
    t=d.elapsed_time_s.to_numpy(float);ok=d.both_gates_ok.to_numpy(bool)
    ax.fill_between(t,d.plot_lower_rad.to_numpy(float),d.plot_upper_rad.to_numpy(float),color=W,alpha=.15,lw=0)
    ax.plot(t,d.plot_lower_rad,color=W,lw=.7);ax.plot(t,d.plot_upper_rad,color=W,lw=.7)
    ax.plot(t,d.plot_reading_rad,color=NAVY,lw=.9)
    for key,marker,col,size in [('A','v',A,20),('C','^',C,26),('W','o',W,7)]:
        m=d.plot_state.eq(key);ax.scatter(t[m],d.loc[m,'plot_reading_rad'],s=size,marker=marker,c=col,edgecolors='none',zorder=4)
    ax.set_xlim(11,22.8);ax.set_ylim(-1.4,1.45);ax.set_xticks([12,14,16,18,20,22]);ax.set_yticks([-1,-.5,0,.5,1])
    ax.set_xlabel('Time in the interaction (s)');ax.set_ylabel('IPV reading (rad)')
    ax.figure.text(.44,.585,'c',fontweight='bold',fontsize=11)
    ax.figure.text(.485,.585,'Reading and dynamic 90% range',fontweight='bold',fontsize=9)
    for tm in TIMES:
        ax.axvline(tm,color=NAVY,lw=.5,ls=':',alpha=.7)
        ax.text(tm,1.31,f'{tm:.1f} s',ha='center',fontsize=6.5)
    ax.text(13.7,-.93,'90% human range',color=W,fontsize=7)
    ax.annotate('C · accommodating side',xy=(19,1.1639),xytext=(20.2,.60),ha='center',fontsize=6.8,color=C,arrowprops=dict(arrowstyle='-',lw=.6,color=C))
    # Each frame has its own 0.1-second bin. Missing values remain true gaps.
    dt=np.median(np.diff(t));strip=ax.inset_axes([0,1.045,1,.042])
    for tm,state in zip(t,d.plot_state):
        strip.add_patch(Rectangle((tm-dt/2,0),dt,1,facecolor={'A':A,'C':C,'W':W,'abstain':ABSTAIN}[state],edgecolor='none'))
    strip.set_xlim(ax.get_xlim());strip.set_ylim(0,1);strip.axis('off')
    ax.text(.01,-.235,'State:  W inside     C accommodating     grey abstention',transform=ax.transAxes,fontsize=6.6)
    cols=['case_key','perspective','elapsed_time_s','anchor_frame_index','status','reason_code','mechanism2_gate_ok','both_gates_ok','ipv_log','lo_90','hi_90','plot_state','plot_reading_rad','plot_lower_rad','plot_upper_rad']
    d[cols].to_csv(DATA/'fig1c_timeline.csv',index=False)


def estimator(p):
    grid=np.array(p['grid']);assert np.allclose(grid,np.arange(-3,4)*np.pi/8)
    assert np.isclose(np.dot(grid,p['exemplar_readable']['w']),p['exemplar_readable']['reading'])
    assert p['exemplar_abstain']['reading'] is None
    fig=plt.figure(figsize=(7.2,3.05));a=fig.add_axes([.065,.20,.30,.68]);b=fig.add_axes([.47,.58,.48,.30]);c=fig.add_axes([.47,.18,.48,.30])
    panel(a,'a','Candidate preference angle')
    a.axhline(0,color=NAVY,lw=.7);a.axvline(0,color=NAVY,lw=.7)
    theta=np.linspace(-3*np.pi/8,3*np.pi/8,250);a.plot(np.cos(theta),np.sin(theta),color=NAVY,lw=.8)
    for q in grid:a.plot([0,np.cos(q)],[0,np.sin(q)],color=NAVY,lw=.5,alpha=.5)
    a.add_patch(Arc((0,0),.65,.65,theta1=0,theta2=45,ec=NAVY,lw=.8))
    a.text(.37,.13,'θ',fontsize=10);a.text(.7,-1.06,'Assertive direction',color=A,ha='center',fontsize=7)
    a.text(.7,1.06,'Accommodating direction',color=C,ha='center',fontsize=7)
    a.set_xlim(-.15,1.2);a.set_ylim(-1.25,1.25);a.set_aspect('equal');a.set_xticks([]);a.set_yticks([])
    a.set_xlabel('Own-progress weight: cos(θ)');a.set_ylabel('Interaction weight: sin(θ)');a.spines[['bottom','left']].set_visible(False)
    rows=[]
    for ax,key,letter,title,col in [(b,'exemplar_readable','b','Concentrated weights → reading',HUMAN),(c,'exemplar_abstain','c','Flat weights → abstention','#9AA0A6')]:
        ex=p[key];ax.bar(grid,ex['w'],width=.23,color=col,edgecolor=NAVY,lw=.45);ax.axhline(1/7,color=NAVY,lw=.6,ls=':')
        ax.set_ylim(0,.57);ax.set_yticks([0,.25,.5]);ax.set_ylabel('Weight');ax.set_xticks(grid)
        ax.set_xticklabels(['−3π/8','−π/4','−π/8','0','π/8','π/4','3π/8'],fontsize=7)
        panel(ax,letter,title);ax.text(.98,.86,f'Frame {ex["frame"]}',transform=ax.transAxes,ha='right',fontsize=7)
        if ex['reading'] is not None:
            ax.axvline(ex['reading'],color=A,lw=1.1);ax.text(.98,.63,f'Reading = {ex["reading"]:.3f} rad',transform=ax.transAxes,ha='right',fontsize=7)
        for q,w in zip(grid,ex['w']):rows.append(dict(example=key,frame=ex['frame'],candidate_rad=q,weight=w,reading_rad=ex['reading'],status=ex['status']))
    b.set_xticklabels([]);c.set_xlabel('Candidate preference (rad)')
    pd.DataFrame(rows).to_csv(DATA/'fig1_supp_estimator.csv',index=False)
    export(fig,'figS4_estimator_details')


def reference_distribution(p):
    edges=np.array(p['human_bins']);counts=np.array(p['human_hist']);assert counts.sum()==p['human_n']==486660
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9));fig.subplots_adjust(left=.09,right=.97,bottom=.24,top=.84,wspace=.45)
    a,b=axes;centres=(edges[:-1]+edges[1:])/2
    a.bar(centres,100*counts/counts.sum(),width=np.diff(edges),color=HUMAN,edgecolor='none')
    a.set_xlabel('IPV reading (rad)');a.set_ylabel('Readable human moments (%)');a.set_xlim(edges[0],edges[-1]);a.set_ylim(0,11)
    panel(a,'a','Human test-fold readings');a.text(.97,.94,'n = 486,660',transform=a.transAxes,ha='right',fontsize=7)
    pd.DataFrame(dict(bin_left_rad=edges[:-1],bin_right_rad=edges[1:],count=counts,denominator=counts.sum(),percent=100*counts/counts.sum())).to_csv(DATA/'figED5a_human_distribution.csv',index=False)
    dep=pd.read_parquet(DEPLOYED,columns=['context_cell','lo_90','hi_90','width_90'])
    rows=[]
    for y,key,label in [(2,'CP|priority','Crossing · priority'),(1,'MP|priority','Merging · priority'),(0,'MP|equal','Merging · equal')]:
        r=p['deployed_ranges'][key];part=dep[dep.context_cell.eq(key)]
        assert len(part)==r['n'] and np.isclose(part.lo_90.median(),r['lo']) and np.isclose(part.hi_90.median(),r['hi'])
        b.plot([r['lo'],r['hi']],[y,y],color=W,lw=2.2);b.scatter([r['lo'],r['hi']],[y,y],color=W,s=18,marker='|')
        b.text((r['lo']+r['hi'])/2,y+.19,label,ha='center',fontsize=7)
        b.text(r['hi']+.04,y,f'n = {r["n"]:,}',ha='left',va='center',fontsize=6.5)
        rows.append(dict(context_cell=key,n_candidate_moments=r['n'],median_lower_rad=r['lo'],median_upper_rad=r['hi'],difference_of_median_bounds_rad=r['w'],median_width_rad=part.width_90.median(),population='all OnSite candidate moments'))
    b.set_ylim(-.4,2.8);b.set_xlim(-1.25,1.7);b.set_yticks([]);b.spines['left'].set_visible(False);b.set_xlabel('IPV reading (rad)')
    panel(b,'b','Ranges in three deployment contexts')
    b.text(0,-.30,'Separate medians of the lower and upper bounds',transform=b.transAxes,fontsize=6.4)
    pd.DataFrame(rows).to_csv(DATA/'figED5b_deployed_range_summaries.csv',index=False)
    export(fig,'figED5_reference_distribution')


def main():
    d,p=setup()
    fig=plt.figure(figsize=(7.2,5.55))
    mechanism(fig.add_axes([.067,.65,.90,.27]))
    trajectories(fig.add_axes([.08,.135,.31,.385]),d)
    timeline(fig.add_axes([.485,.135,.485,.385]),d)
    export(fig,'fig1_monitoring');estimator(p);reference_distribution(p)
    sources=[source_record('1','a',RUN/'process/input_plan.md','Section 3: monitor logic','conceptual','schematic; no measured quantities'),source_record('1','b,c',CASE,'coordinates; elapsed_time_s; ipv_log; lo_90; hi_90; both_gates_ok; exceed_side_90','frames','case_key=ipv_004992; perspective=key_agent_1','221 source rows; timeline 11.0–22.7 s; trajectories from 13.2 s'),source_record('S4','a–c',PAYLOAD,'grid; exemplar_readable; exemplar_abstain','candidate weights','frozen two-frame examples','No resampling or fitted uncertainty'),source_record('ED5','a',PAYLOAD,'human_bins; human_hist; human_n','readable human moments','human test fold','486660 moments'),source_record('ED5','b',PAYLOAD,'deployed_ranges','context summaries','CP|priority; MP|priority; MP|equal','Lower and upper medians computed separately; not one observed interval'),source_record('ED5','b',DEPLOYED,'context_cell; lo_90; hi_90; width_90','all deployment candidate moments','same three context cells','Confirms all frozen counts and bound medians')]
    sp=RUN/'process/fig13_sources.csv'
    old=pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    if not old.empty: old=old[~old.figure.astype(str).isin(['1','S4','ED5'])]
    pd.concat([old,pd.DataFrame(sources)]).to_csv(sp,index=False)
    q=dict(fig1_source_rows=len(d),fig1_state_counts=d.plot_state.value_counts().to_dict(),fig1_rejected_plot_values_all_nan=True,case=CASE_KEY,source_masks_checked=True,reference_range_median_check=True)
    qp=RUN/'process/fig13_qa.json'
    prior=json.loads(qp.read_text()) if qp.exists() else {}
    prior.update(q);qp.write_text(json.dumps(prior,indent=2))
    print(json.dumps(q))

if __name__=='__main__':main()
