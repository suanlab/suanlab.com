"""Reproduce bilingual ray-speed, projection and fixed-exposure splitting figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'public/assets/learning';PREVIEW=ROOT/'.runtime/learning/quantum-unitary-research';PREVIEW.mkdir(parents=True,exist_ok=True)
font=next(p for p in font_manager.findSystemFonts()if'NotoSansCJK-Regular'in p);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','svg.hashsalt':'suanlab-quantum-unitary-research','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'axes.labelcolor':'#1e293b','text.color':'#1e293b','figure.facecolor':'white'})
BLUE,ORANGE,GREEN='#2563eb','#c2410c','#047857'
def save(fig,name,lang):
    p=OUT/f'quantum-unitary-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.25,metadata={'Date':None});p.write_text('\n'.join(s.rstrip()for s in p.read_text().splitlines())+'\n');fig.savefig(PREVIEW/f'{name}.{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.25);plt.close(fig)
X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.]);target=expm(-1j*(X+Z));counts=[1,2,4,8,16,32];lie=[];symmetric=[]
for n in counts:
    a=expm(-1j*X/n);b=expm(-1j*Z/n);half=expm(-1j*X/(2*n));lie.append(np.linalg.norm(np.linalg.matrix_power(b@a,n)-target,2));symmetric.append(np.linalg.norm(np.linalg.matrix_power(half@b@half,n)-target,2))
for lang in ['ko','en']:
    ko=lang=='ko';fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True);time=np.linspace(0,3.2,600);w=.25;sigma=2*np.sqrt(w*(1-w));allowed=time<=np.pi/(2*sigma)
    a.plot(time,1-4*w*(1-w)*np.sin(time)**2,color=BLUE,label='실제 fidelity, w = 1/4'if ko else'Actual fidelity, w = 1/4');a.plot(time[allowed],np.cos(sigma*time[allowed])**2,color=ORANGE,ls='--',label='허용 구간의 하한'if ko else'Lower bound on allowed interval');a.axhline(.25,color=GREEN,ls=':',label='최소 fidelity: 1/4'if ko else'Minimum fidelity: 1/4');a.set(xlabel='t (b = hbar = 1)',ylabel='Fidelity',title='유한한 속도 경계는 직교 도달을 보장하지 않는다'if ko else'A finite speed bound need not yield orthogonality',ylim=(0,1.06));a.legend(fontsize=10)
    n=np.arange(1,33);b.plot(n,np.cos(1/n)**(2*n),color=BLUE,label='실제 사영의 all-yes'if ko else'All yes with actual projections');b.axhline(np.cos(1)**2,color=ORANGE,ls='--',label='측정 없는 최종 생존'if ko else'Final survival without measurement');b.set(xlabel='Projection count N',ylabel='Probability',title='같은 총 시간 T = 1, 서로 다른 측정 절차'if ko else'Same total T = 1, different measurement protocol',ylim=(.2,1.06));b.legend(fontsize=10)
    for ax in [a,b]:ax.grid(alpha=.15)
    save(fig,'ray-speed-zeno',lang)
    fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True);ns=np.array(counts)
    for ax,errors,bounds,title in [(a,lie,1/ns,'Lie: first order'),(b,symmetric,np.minimum(2,8*np.exp(2/ns)/(3*ns*ns)),'Symmetric: second order')]:
        ax.loglog(ns,errors,'o-',color=BLUE,label='계산 오차'if ko else'Computed error');ax.loglog(ns,bounds,'s--',color=ORANGE,label='유한 노름 상한'if ko else'Finite norm upper bound');ax.set_xticks(counts,list(map(str,counts)));ax.set(xlabel='Subdivisions n (total physical time = 2)',ylabel='Operator norm error',title=title);ax.grid(alpha=.15);ax.legend(fontsize=10)
    save(fig,'fixed-exposure-splitting',lang)
