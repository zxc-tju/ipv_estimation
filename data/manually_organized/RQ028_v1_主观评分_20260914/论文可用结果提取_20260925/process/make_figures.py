from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
T=OUT/'tables'
F=OUT/'figures'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'legend.frameon':False,'savefig.facecolor':'white'})
desc=pd.read_csv(T/'rating_descriptives.csv')
ratings=pd.read_csv(T/'ratings_analysis_rows.csv')
contrasts=pd.read_csv(T/'participant_paired_contrasts.csv')
scene=pd.read_csv(T/'common_scenario_sensitivity.csv')
colors={'ego':'#457797','cp':'#B56542'}
labels={'ego':'Ego position','cp':'Counterpart position'}
classes=['A','C','W']
class_labels=['A','C','W']
metrics=['q1_rev','q4','q5_cost_cp']
titles=['Perceived atypicality','Comfort','Reported interaction cost']

def export(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(F/f'{name}.{ext}',dpi=300)
    plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(183/25.4,90/25.4))
for j,(ax,m,title) in enumerate(zip(axes,metrics,titles)):
    seats=['cp'] if m=='q5_cost_cp' else ['ego','cp']
    for seat in seats:
        offset=0 if len(seats)==1 else (-.13 if seat=='ego' else .13)
        a=desc[(desc.metric==m)&(desc.seat==seat)].set_index('class').loc[classes]
        for i,c in enumerate(classes):
            g=ratings[(ratings['class']==c)&(ratings.seat==seat)]
            p=g.groupby(['pair_id','subject_id'])[m].mean().groupby('pair_id').mean()
            jitter=np.linspace(-.055,.055,len(p))
            ax.scatter(i+offset+jitter,p,color=colors[seat],s=6,alpha=.2,linewidths=0,zorder=1)
        y=a.subject_equal_mean.to_numpy()
        ax.errorbar(np.arange(3)+offset,y,yerr=np.vstack([y-a.ci_low,a.ci_high-y]),
            fmt='o' if seat=='cp' else 's',markersize=4,capsize=3,lw=1,color=colors[seat],
            label=labels[seat],zorder=4)
    ax.set_xticks(range(3),class_labels)
    ax.set_ylim(.8,7.2);ax.set_yticks([1,2,3,4,5,6,7])
    ax.set_ylabel('Rating (1–7)')
    ax.set_title(title,loc='left',pad=12)
    ax.text(-.16,1.09,'abc'[j],fontweight='bold',transform=ax.transAxes,fontsize=11)
    ax.grid(axis='y',color='#dddddd',lw=.4,zorder=0)
axes[0].legend(loc='upper left',fontsize=7,handletextpad=.4)
axes[2].text(.03,.96,'Counterpart only',transform=axes[2].transAxes,va='top',fontsize=7,color=colors['cp'])
fig.subplots_adjust(left=.065,right=.985,bottom=.27,top=.84,wspace=.43)
fig.text(.5,.035,'A: assertive side; C: accommodating side; W: within range.\n40 participants in 20 pairs; 158 A, 239 C and 199 W ratings per position.\nPoints: equal-participant means; bars: 95% pair-bootstrap intervals; faint dots: pair means.',ha='center',fontsize=7)
export(fig,'FigS1_subjective_ratings')

fig,axes=plt.subplots(1,3,figsize=(183/25.4,90/25.4))
for j,(ax,m,title) in enumerate(zip(axes,metrics,titles)):
    for df,off,color,mark,label in [(contrasts,-.1,'#457797','o','Participant paired'),
                                   (scene,.1,'#7A705F','s','Common scenarios')]:
        a=df[(df.metric==m)&(df.seat=='cp')].set_index('contrast').loc[['A-W','C-W','A-C']]
        y=a.estimate.to_numpy()
        ax.errorbar(y,np.arange(3)+off,xerr=np.vstack([y-a.ci_low,a.ci_high-y]),
                    fmt=mark,markersize=4,capsize=3,lw=1,color=color,label=label)
    ax.axvline(0,lw=.7,color='#999999',ls='--')
    ax.set_yticks(range(3),['A − W','C − W','A − C'])
    ax.invert_yaxis();ax.set_ylim(2.45,-.6)
    ax.set_xlabel('Difference in rating points')
    ax.set_title(title,loc='left',pad=12)
    ax.text(-.16,1.09,'abc'[j],fontweight='bold',transform=ax.transAxes,fontsize=11)
    ax.grid(axis='x',color='#dddddd',lw=.4)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',ncol=2,fontsize=7,bbox_to_anchor=(.5,1.0))
fig.subplots_adjust(left=.07,right=.985,bottom=.26,top=.84,wspace=.52)
fig.text(.5,.035,'Counterpart position; common-scenario contrasts use 8 (A − W), 9 (C − W) and 12 (A − C) scenarios.\nIntervals resample participant pairs or common scenarios separately; these are sensitivity analyses.',ha='center',fontsize=7)
export(fig,'FigS2_common_scenario_check')
print('Wrote two candidate figures in PNG, PDF and SVG.')
