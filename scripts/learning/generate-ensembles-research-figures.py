"""Generate bilingual native SVGs for the three ensemble research extensions."""
from pathlib import Path
from math import exp, pi
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue,orange,green='#2563eb','#c2410c','#047857'
out=Path('public/assets/learning');preview=Path('.runtime/learning/ensembles-research');preview.mkdir(parents=True,exist_ok=True)
def save(fig,name,lang):
    fig.tight_layout(pad=2)
    p=out/f'ensembles-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.2)
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    fig.savefig(preview/f'{name}.{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.2);plt.close(fig)
for lang in ['en','ko']:
    ko=lang=='ko'
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));labels=['00','01','10','11'];indices=np.arange(4)
    for offset,N,label,color in [(-.25,4,'N=4',orange),(0,20,'N=20',blue),(.25,None,'독립 비교' if ko else 'Independent comparison',green)]:
        d=0 if N is None else 1/(N-1);P=np.array([1-d,1+d,1+d,1-d])/4
        ax[0].bar(indices+offset,P,width=.23,label=label,color=color)
    ax[0].set(xticks=indices,xticklabels=labels,xlabel='같은 준비에서 관측한 두 자리' if ko else 'Two sites in the same preparation',ylabel='표지 패턴 확률' if ko else 'Labeled pattern probability',title='한 자리 평균은 모두 1/2' if ko else 'Every one-site mean remains 1/2');ax[0].set_ylim(0,.42);ax[0].legend(ncol=3,fontsize=10,loc='upper center')
    sizes=np.arange(4,102,2);ax[1].plot(sizes,.5/(sizes-1),color=blue,label='쌍 법칙 TV' if ko else 'Pair-law TV');ax[1].plot(sizes,.25/(sizes-1),color=orange,label='공분산 크기' if ko else 'Covariance magnitude')
    ax[1].set(xlabel='총 자리 수 N, K=N/2' if ko else 'Total sites N, K=N/2',ylabel='무차원 의존성 신호' if ko else 'Dimensionless dependence signal',title='큰 예산에서 약해지는 식별 신호' if ko else 'A weaker identification signal at larger budgets');ax[1].legend();save(fig,'reservoir-identification',lang)
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));x=np.logspace(-2,2,250)
    ax[0].semilogx(x,1/np.sqrt(1+x*x),color=blue,label='진폭 / 정적 열용량' if ko else 'Amplitude / static capacity');ax[0].semilogx(x,1/(1+x*x),color=green,ls=':',label='동상 성분' if ko else 'In-phase component');ax[0].semilogx(x,x/(1+x*x),color=orange,ls='--',label='지연 성분' if ko else 'Lag component');ax[0].axvline(1,color='#64748b',ls=':')
    ax[0].set(xlabel='주파수 / 완화율, omega/gamma' if ko else 'Frequency / relaxation rate, omega/gamma',ylabel='C0로 정규화한 응답' if ko else 'Response normalized by C0',title='같은 정적 요동, 다른 속도 의존성' if ko else 'Same static fluctuations, speed-dependent response');ax[0].legend()
    phase=np.linspace(0,2*pi,200);ax[1].plot(phase,np.cos(phase),color=orange,label='온도 / 구동 진폭' if ko else 'Temperature / drive amplitude');ax[1].plot(phase,np.cos(phase-pi/4)/np.sqrt(2),color=blue,label='에너지 / (C0 x 구동 진폭)' if ko else 'Energy / (C0 x drive amplitude)')
    ax[1].set(xlabel='위상 omega x 시간 (rad)' if ko else 'Phase omega x time (rad)',ylabel='정규화한 일차 변화' if ko else 'Normalized first-order change',title='omega=gamma: 진폭 감소와 45도 지연' if ko else 'omega=gamma: amplitude loss and 45-degree lag');ax[1].legend();save(fig,'driven-canonical-response',lang)
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));e2=np.linspace(1,3,150);r1=1/(1+exp(1));r2=1/(1+np.exp(e2));det=r1*(1-r1)*r2*(1-r2)*(e2-1)**2
    ax[0].plot(e2,det,color=blue);ax[0].scatter([1],[0],color=orange,zorder=3)
    ax[0].set(xlabel='두 번째 에너지 / E0' if ko else 'Second energy / E0',ylabel='det Cov(E,N) / E0 squared',title='교정한 에너지: beta x E0=1, mu=0' if ko else 'Calibrated energies: beta x E0=1, mu=0')
    ax[1].plot([0,2],[0,2],ls=':',color='#64748b',label='E=E0 x N')
    ax[1].scatter([0,1,2],[0,1,2],s=90,color=orange,label='동일 에너지 1,1' if ko else 'Equal energies 1,1');ax[1].scatter([0,1,1,2],[0,1,2,3],s=40,color=blue,marker='x',label='서로 다른 에너지 1,2' if ko else 'Unequal energies 1,2')
    for n,e,label in [(0,0,'00'),(1,1,'10'),(1,2,'01'),(2,3,'11')]:ax[1].annotate(label,(n,e),xytext=(8,6),textcoords='offset points')
    ax[1].set(xlabel='입자 수 N' if ko else 'Particle number N',ylabel='에너지 / E0' if ko else 'Energy / E0',xlim=(-.2,2.35),ylim=(-.2,3.35),xticks=[0,1,2],title='가능한 상태: 궤적이나 확률 막대가 아님' if ko else 'Possible states: not trajectories or probability bars');ax[1].legend(loc='upper left');save(fig,'grand-response-rank',lang)
