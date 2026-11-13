"""Generate bilingual native figures for calibrated laws and sampling fidelity."""
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue,orange,green='#2563eb','#c2410c','#047857'
out=Path('public/assets/learning');preview=Path('.runtime/learning/boltzmann-research');preview.mkdir(parents=True,exist_ok=True)
def save(fig,name,lang):
    fig.tight_layout(pad=2)
    p=out/f'boltzmann-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.2);p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    fig.savefig(preview/f'{name}.{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.2);plt.close(fig)
for lang in ['en','ko']:
    ko=lang=='ko';fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));b=np.linspace(.1,4,150);eta=np.log(2)
    ax[0].plot(b,eta-b,color=blue,label='간격 1, 기준 비 2' if ko else 'Gap 1, reference ratio 2');ax[0].plot(b,eta+.4-1.8*b,color=orange,ls='--',label='다른 간격과 기준 비' if ko else 'Different gap and reference ratio');ax[0].axvline(.5,color='#64748b',ls=':');ax[0].scatter([.5],[eta-.5],color=green,zorder=3)
    ax[0].set(xlabel='교정된 역온도 b=beta x E0' if ko else 'Calibrated inverse temperature b=beta x E0',ylabel='모집단 로그 오즈' if ko else 'Population log odds',title='한 설정에서는 겹치고 다른 설정에서는 분리' if ko else 'Coincident at one setting, separated at another');ax[0].legend(loc='upper right')
    b2=np.linspace(.1,10,200);r1=1/(1+np.exp(.5-eta));r2=1/(1+np.exp(b2-eta));det=r1*(1-r1)*r2*(1-r2)*(b2-.5)**2
    ax[1].plot(b2,det,color=blue);ax[1].scatter([.5],[0],color=orange,zorder=3);ax[1].set(xlabel='둘째 역온도 b2, 첫 설정 b1=0.5' if ko else 'Second inverse temperature b2, first b1=0.5',ylabel='관측 쌍당 Fisher 행렬식' if ko else 'Fisher determinant per observation pair',title='서로 다른 설정과 포화 사이의 균형' if ko else 'Distinct settings and saturation both matter');save(fig,'two-temperature-identification',lang)
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));x=np.arange(4);a=.9
    ax[0].bar(x-.15,[(1-a)/2]*2+[a/2]*2,width=.28,color=orange,label='비보정 정상 법칙' if ko else 'Uncorrected stationary law');ax[0].bar(x+.15,[.25]*4,width=.28,color=blue,label='목표 = 보정 정상 법칙' if ko else 'Target = corrected stationary law');ax[0].set(xticks=x,xticklabels=['00','01','10','11'],xlabel='표지된 두 비트 상태' if ko else 'Labeled two-bit state',ylabel='확률' if ko else 'Probability',ylim=(0,.6),title='올바른 조건부, 편향된 자리 선택 (a=0.9)' if ko else 'Correct conditionals, biased selection (a=0.9)');ax[0].legend(loc='upper center',fontsize=10)
    k=np.arange(51);ax[1].plot(k,a*(1-.5**k),color=orange,label='비보정' if ko else 'Uncorrected');ax[1].plot(k,.5*(1-(1-min(a,1-a))**k),color=blue,label='선택 비 보정' if ko else 'Selection-ratio corrected');ax[1].axhline(.5,color=green,ls=':',label='목표 평균 1/2' if ko else 'Target mean 1/2');ax[1].set(xlabel='자리 갱신 시도 k (초가 아님)' if ko else 'Coordinate update attempts k (not seconds)',ylabel='첫 비트의 평균' if ko else 'Mean first bit',title='초기 첫 비트 0: 빠른 수렴도 목표 점검이 필요' if ko else 'Initial first bit 0: fast convergence needs a target check');ax[1].legend(loc='lower right');save(fig,'state-dependent-scan',lang)
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));step=np.linspace(.001,.495,250)
    for lam,color in [(1,blue),(4,orange)]:
        ax[0].semilogy(step,1/(lam*(1-step*lam/2)),color=color,label=f'Euler: lambda={lam}');ax[0].axhline(1/lam,color=color,ls=':',label=f'target: 1/{lam}')
    ax[0].axvline(.5,color='#64748b',ls='--');ax[0].set(xlabel='알고리즘 단계 eta' if ko else 'Algorithm step eta',ylabel='고유 모드 정상 분산' if ko else 'Stationary eigenmode variance',xlim=(0,.51),title='목표는 유효해도 단계에 따라 부푸는 분산' if ko else 'A valid target, step-dependent variance inflation');ax[0].legend(loc='upper left',fontsize=10)
    theta=np.linspace(0,2*np.pi,250);circle=np.array([np.cos(theta),np.sin(theta)]);Q=np.array([[1,-1],[1,1]])/np.sqrt(2)
    for variances,color,label in [([1,.25],blue,'목표 = 정확 선형 전이' if ko else 'Target = exact linear transition'),([1/(1-.45/2),1/(4*(1-.45*4/2))],orange,'Euler eta=0.45')]:
        ellipse=Q@np.diag(np.sqrt(variances))@circle;ax[1].plot(ellipse[0],ellipse[1],color=color,label=label)
    ax[1].set(xlabel='중심화 좌표 y1' if ko else 'Centered coordinate y1',ylabel='중심화 좌표 y2' if ko else 'Centered coordinate y2',title='표준편차 공분산 타원: 표집 경로가 아님' if ko else 'One-standard-deviation covariance ellipses, not paths');ax[1].set_xlim(-1.7,1.7);ax[1].set_ylim(-1.7,2.2);ax[1].set_aspect('equal');ax[1].legend(loc='upper center',fontsize=10);save(fig,'gaussian-sampler-bias',lang)
