"""Bilingual finite-count tomography and dephasing-divisibility figures."""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'public/assets/learning';PREVIEW=ROOT/'.runtime/learning/quantum-density-research';PREVIEW.mkdir(parents=True,exist_ok=True)
font=next(p for p in font_manager.findSystemFonts()if'NotoSansCJK-Regular'in p);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','svg.hashsalt':'suanlab-quantum-density-research','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'axes.labelcolor':'#1e293b','text.color':'#1e293b','figure.facecolor':'white'})
BLUE,ORANGE,GREEN='#2563eb','#c2410c','#047857'
def save(fig,name,lang):
    p=OUT/f'quantum-density-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.25,metadata={'Date':None});p.write_text('\n'.join(s.rstrip()for s in p.read_text().splitlines())+'\n');fig.savefig(PREVIEW/f'{name}.{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.25);plt.close(fig)
ratios=[1,2,5,10];points=[];gains=[]
for ratio in ratios:
    def loss(angle):
        x,y=np.cos(angle),np.sin(angle)
        return sum(-n*(.9*np.log((1+r)/2)+.1*np.log((1-r)/2))for n,r in [(10*ratio,x),(10,y)])
    result=minimize_scalar(loss,bounds=(np.arccos(.8),np.arcsin(.8)),method='bounded',options={'xatol':1e-14});points.append((np.cos(result.x),np.sin(result.x)));gains.append(loss(np.pi/4)-result.fun)
for lang in ['ko','en']:
    ko=lang=='ko';fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True);arc=np.linspace(0,np.pi/2,400);a.plot(np.cos(arc),np.sin(arc),color='#94a3b8');a.scatter([.8],[.8],marker='x',s=70,color=ORANGE,label='선형 추정 (비물리적)'if ko else'Linear estimate (unphysical)');rad=1/np.sqrt(2);a.scatter([rad],[rad],color=GREEN,s=60,label='방사형 사영'if ko else'Radial projection')
    for ratio,(x,y)in zip(ratios,points):a.scatter([x],[y],color=BLUE,s=45);a.annotate(str(ratio),(x,y),xytext=(5,5),textcoords='offset points',fontsize=10)
    a.set(xlabel='r_x',ylabel='r_y',title='ML 적합: 점의 숫자는 N_x/N_y'if ko else'ML fits: point labels are N_x/N_y',xlim=(.5,.88),ylim=(.5,.88));a.set_aspect('equal');a.legend(loc='lower left',fontsize=9)
    b.plot(ratios,gains,'o-',color=BLUE);b.set(xlabel='Sample allocation N_x/N_y',ylabel='Log-likelihood gain',title='같은 빈도, 다른 통계적 무게'if ko else'Same frequencies, different statistical weights');b.set_xticks(ratios)
    for ax in [a,b]:ax.grid(alpha=.15)
    save(fig,'constrained-tomography',lang)
    fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True);time=np.linspace(0,np.pi,601);eta=np.cos(time)
    a.plot(time,eta,color=BLUE,label='Static frequency: eta');a.plot(time,np.abs(eta),color=ORANGE,ls='--',label='Pair trace distance');a.plot(time,np.exp(-time),color=GREEN,ls=':',label='Semigroup eta = exp(-Omega*t)');a.set(ylabel='eta / distance',title='각 map은 유효하지만 erasure 뒤 회복한다'if ko else'Each map is valid, yet distinguishability revives',ylim=(-1.08,1.08));a.legend(fontsize=9)
    p=(1+np.abs(eta))/2;entropy=np.array([-sum(v*math.log2(v)for v in [q,1-q]if 0<v<1)for q in p]);b.plot(time,entropy,color=BLUE,label='Entropy of +x probe [bits]');b.plot(time,(1+eta)/2,color=ORANGE,ls='--',label='Later +x probability');b.set(ylabel='Entropy / probability',title='단일 탐침 entropy와 두 상태 거리를 구별한다'if ko else'One-probe entropy differs from two-state distance',ylim=(-.05,1.08));b.legend(fontsize=9)
    for ax in [a,b]:ax.axvline(np.pi/2,color='#94a3b8',ls=':');ax.set(xlabel='Dimensionless physical time Omega*t');ax.set_xticks([0,np.pi/2,np.pi],['0','pi/2','pi']);ax.grid(alpha=.15)
    save(fig,'dephasing-divisibility',lang)
