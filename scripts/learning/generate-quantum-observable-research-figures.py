"""Bilingual calibration-gap and repeatable-instrument comparisons."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'public/assets/learning';PREVIEW=ROOT/'.runtime/learning/quantum-observable-research';PREVIEW.mkdir(parents=True,exist_ok=True)
font=next(p for p in font_manager.findSystemFonts()if'NotoSansCJK-Regular'in p);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','svg.hashsalt':'suanlab-quantum-observable-research','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'axes.labelcolor':'#1e293b','text.color':'#1e293b','figure.facecolor':'white'})
BLUE,ORANGE,GREEN='#2563eb','#c2410c','#047857'
def save(fig,name,lang):
    p=OUT/f'quantum-observable-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.25,metadata={'Date':None});p.write_text('\n'.join(s.rstrip()for s in p.read_text().splitlines())+'\n');fig.savefig(PREVIEW/f'{name}.{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.25);plt.close(fig)
for lang in ['ko','en']:
    ko=lang=='ko';fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True);eps=np.linspace(0,1,400)
    a.plot(eps,eps,color=BLUE,lw=2,label='표준편차'if ko else'Standard deviation');a.plot(eps,eps*eps,color=GREEN,ls='--',label='분산'if ko else'Variance');a.set(xlabel='epsilon',ylabel='수치 척도 (A는 무차원)'if ko else'Numerical scale (dimensionless A)',title='영 관측량으로 접근하는 판독값'if ko else'Readouts approach the zero observable');a.legend(fontsize=10)
    b.plot(eps[1:],np.ones(399)*.5,color=ORANGE,lw=2);b.scatter([0],[.5],facecolors='white',edgecolors=ORANGE,s=70,zorder=3);b.scatter([0],[1],color=BLUE,s=70,zorder=3);b.annotate('epsilon = 0: one branch',(0,1),xytext=(18,-2),textcoords='offset points',fontsize=10)
    b.set(xlabel='epsilon',ylabel='후속 B 확률'if ko else'Later B probability',title='지정한 이상적 도구의 불연속'if ko else'Discontinuity of the specified ideal instrument',ylim=(.4,1.08))
    for ax in [a,b]:ax.grid(alpha=.15)
    save(fig,'closing-spectral-gap',lang)
    fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True);theta=np.linspace(0,np.pi,400)
    for ax,ys in [(a,[np.ones(400),np.ones(400)*.5,(1+np.cos(theta))/2]),(b,[np.ones(400),np.ones(400)*.5,np.ones(400)])]:
        for y,color,style,label in zip(ys,[BLUE,GREEN,ORANGE],['-','--',':'],['coarse Lüders','rank-one refinement','repeatable rotation']):ax.plot(theta,y,color=color,ls=style,lw=2,label=label)
        ax.set_xticks([0,np.pi/2,np.pi],['0','pi/2','pi']);ax.set_ylim(-.06,1.12);ax.grid(alpha=.15);ax.legend(loc='lower left',fontsize=10)
    a.set(xlabel='theta [rad]',ylabel='후속 B 확률'if ko else'Later B probability',title='모든 도구가 A = 2를 확실히 보고한다'if ko else'Every instrument reports A = 2 certainly')
    b.set(xlabel='theta [rad]',ylabel='순수성'if ko else'Output purity',title='같은 순수성이 같은 상태를 뜻하지 않는다'if ko else'Equal purity does not identify the state')
    save(fig,'repeatable-block-rotation',lang)
