from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[1]);HERE=_P(__file__).resolve().parent
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif']})
BLUE,GREEN,RED='#000000','#000000','#000000'
boxes=[('Regional epidemic reports','Daily aggregate reports with arrival dates',BLUE),
('Data-quality screening','Completeness, timestamps, missing or late reports, abnormal values',BLUE),
('Historical communication profiling','Delay and drop priors from historical generation–arrival records',BLUE),
('Delay-aware data representation','Received-message stream with possible delays, gaps, and outliers',BLUE),
('Robust graph-based inference','Latent-state reconstruction; exploratory association graph',GREEN),
None,
('Level calibration and empirical variability','Region-specific level check against consolidated history;\nseed-based variability (not calibrated posterior intervals)',GREEN),
('Analyst review','Trends, flagged state-days, variability, exploratory associations',RED),
('Public-health decision support','Situational awareness, data-quality review, burden revision,\nand stress testing under human oversight',RED)]
fig,ax=plt.subplots(figsize=(10,9.4));ax.set_xlim(-0.3,11.8);ax.set_ylim(-0.4,12.4);ax.axis('off')
x0,w=2.1,7.2;gap=0.36;y=12.25;C=[]
for b in boxes:
    if b is None:
        hh=1.55;ax.add_patch(Rectangle((x0,y-hh),w,hh,fill=False,lw=1.4,ec=GREEN))
        ax.text(x0+w/2,y-0.25,'Two analysis modes',ha='center',va='center',fontsize=13,color=GREEN)
        for k,(t,s,ls,c) in enumerate([('Retrospective reconstruction','History recovered from reports\navailable at audit time (evaluated)','-',GREEN),('Causal rolling nowcasting','Experimental; not validated for\noperational decision use','--','#000000')]):
            bx=x0+0.12+k*(w/2);ax.add_patch(Rectangle((bx,y-hh+0.1),w/2-0.24,0.98,fill=False,lw=1.1,ls=ls,ec=c))
            ax.text(bx+(w/2-0.24)/2,y-hh+0.83,t,ha='center',va='center',fontsize=10.2,color=c)
            ax.text(bx+(w/2-0.24)/2,y-hh+0.42,s,ha='center',va='center',fontsize=8.8,color='black')
        C.append((y,y-hh));y-=hh+gap;continue
    two='\n' in b[1];hh=0.95+(0.25 if two else 0)
    ax.add_patch(Rectangle((x0,y-hh),w,hh,fill=False,lw=1.4,ec=b[2]))
    ax.text(x0+w/2,y-0.3,b[0],ha='center',va='center',fontsize=13,color=b[2])
    ax.text(x0+w/2,y-hh+0.32+(0.12 if two else 0),b[1],ha='center',va='center',fontsize=9)
    C.append((y,y-hh));y-=hh+gap
for (t1,b1),(t2,b2) in zip(C[:-1],C[1:]): ax.add_patch(FancyArrowPatch((x0+w/2,b1),(x0+w/2,t2),arrowstyle='-|>',mutation_scale=12,lw=1.1,color='k'))
ya=sum(C[7])/2;yb=sum(C[2])/2
ax.plot([x0+w,x0+w+0.45,x0+w+0.45],[ya,ya,yb],color=RED,lw=1.1);ax.add_patch(FancyArrowPatch((x0+w+0.45,yb),(x0+w,yb),arrowstyle='-|>',mutation_scale=12,lw=1.1,color=RED))
ax.text(x0+w+0.58,(ya+yb)/2,'Feedback loop\nUpdate priors,\ncalibration factors,\nand monitoring\nrules as reporting\nbehavior changes',ha='left',va='center',fontsize=10,color=RED)
for (i,j,lab,c) in [(0,3,'Inputs and\npreprocessing',BLUE),(4,6,'Model-based\nanalytics',GREEN),(7,8,'Operational\nuse',RED)]:
    top,bot=C[i][0],C[j][1];ax.plot([x0-0.15,x0-0.3,x0-0.3,x0-0.15],[top,top,bot,bot],color='0.45',lw=2.2)
    ax.text(x0-0.42,(top+bot)/2,lab,ha='right',va='center',fontsize=13,color=c)
fig.savefig(str(HERE/'figures'/'fig_operational_workflow.png'),dpi=400,bbox_inches='tight',facecolor='white')
