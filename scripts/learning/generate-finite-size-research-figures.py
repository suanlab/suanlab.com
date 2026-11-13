"""Native bilingual plots for boundary metrics, moment escape and shifted zeros."""
from pathlib import Path
from fractions import Fraction as F
from math import log,cosh,tanh,pi,sqrt
import subprocess,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from importlib.util import spec_from_file_location,module_from_spec
sys.dont_write_bytecode=True
spec=spec_from_file_location('verify',Path(__file__).with_name('verify-finite-size-research.py'));v=module_from_spec(spec);spec.loader.exec_module(v)
font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue,orange,green='#2563eb','#c2410c','#047857';out=Path('public/assets/learning');preview=Path('.runtime/learning/finite-size-research');preview.mkdir(parents=True,exist_ok=True)
def save(fig,name,lang):
    fig.tight_layout(pad=2);p=out/f'finite-size-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.2);p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');fig.savefig(preview/f'{name}.{lang}.png',dpi=110,bbox_inches='tight',pad_inches=.2);plt.close(fig)
for lang in ('en','ko'):
    ko=lang=='ko';fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));ns=np.array([8,16,32,64,128,256,512]);k=1.;t=tanh(k)
    ax[0].loglog(ns,log(cosh(k))/ns,'o-',color=blue,label='정확한 abs(beta DeltaF)/N' if ko else 'Exact abs(beta DeltaF)/N');ax[0].loglog(ns,k/ns,color=orange,ls='--',label='경계 상계 K/N' if ko else 'Cut bound K/N');ax[0].set(xlabel='열적 열린 사슬의 사이트 수 N' if ko else 'Thermal open-chain site count N',ylabel='무차원 사이트당 변화' if ko else 'Dimensionless change per site',title='K=1: 한 결합 자유에너지의 밀도 영향' if ko else 'K=1: one-bond free-energy density effect');ax[0].legend(loc='upper right',fontsize=10)
    ax[1].semilogy(ns,t**(ns/2-1),'o-',color=blue,label='두 내부 스핀' if ko else 'Two interior spins');ax[1].axhline(t,color=orange,ls='--',label='계면 양쪽 스핀' if ko else 'Spins adjacent to the cut');ax[1].set(xlabel='N (내부 관측 간 결합 거리 N/2-1)' if ko else 'N (interior pair distance N/2-1 bonds)',ylabel='연결 공분산' if ko else 'Connected covariance',title='깊은 내부의 감쇠와 계면에 남는 의존' if ko else 'Deep-interior decay and persistent interface dependence',ylim=(1e-32,10));ax[1].legend(loc='lower left',fontsize=10);save(fig,'boundary-free-energy-information',lang)

    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));ns=np.geomspace(16,4096,200)
    for alpha,col in [(.5,green),(1,orange),(2,blue)]:ax[0].loglog(ns,.25*(1+(ns-1)*.5/ns**alpha),color=col,label=f'alpha={alpha}')
    ax[0].axhline(.25,color='#64748b',ls=':',label='약한 Gaussian의 분산' if ko else 'Weak Gaussian variance');ax[0].set(xlabel='사이트 수 N' if ko else 'Site count N',ylabel='실제 E[W_N²] = N Var(rho)' if ko else 'Actual E[W_N²] = N Var(rho)',title='p=c=1/2: 같은 약한 극한, 다른 모멘트' if ko else 'p=c=1/2: same weak limit, different moments',ylim=(.2,20));ax[0].legend(loc='upper left',fontsize=10)
    n=256;p=F(1,2);common=F(1,2*n);law=v.mixture(n,p,common);variance=float(F(1,4)*(1+(n-1)*common)/n);std=sqrt(variance);xs=(np.arange(n+1)/n-.5)/std;spacing=1/(n*std);ax[1].plot(xs,np.array(list(map(float,law)))/spacing,color=blue,label='유한 중앙 격자 질량 / 간격' if ko else 'Finite central mass / spacing');grid=np.linspace(-3.5,3.5,400)
    for var,col,style,label in [(2/3,orange,'--','약한 극한: 분산 2/3' if ko else 'Weak limit: variance 2/3'),(1,green,':','단위 Gaussian 기준' if ko else 'Unit Gaussian reference')]:ax[1].plot(grid,np.exp(-grid**2/(2*var))/sqrt(2*pi*var),color=col,ls=style,label=label)
    ax[1].set(xlabel='실제 표준편차로 나눈 T_N' if ko else 'T_N using the actual standard deviation',ylabel='중앙 밀도 비교' if ko else 'Central density comparison',title='alpha=1, N=256: 희귀 집단 원자는 그림 밖' if ko else 'alpha=1, N=256: rare collective atoms are outside view',xlim=(-3.5,3.5),ylim=(0,.7));ax[1].legend(loc='upper right',fontsize=9);save(fig,'rare-mode-limit-moments',lang)

    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));g=np.linspace(-.6,.2,500)
    for n,col in [(16,blue),(64,orange),(256,green)]:
        odds=n*g+2*log(n);prob=1/(1+np.exp(-odds));ax[0].plot(g,prob,color=col,label=f'kappa=2, N={n}');ax[0].scatter([-2*log(n)/n],[.5],color=col,s=18)
    ax[0].plot(g,1/(1+np.exp(-64*g)),color='#64748b',ls='--',label='등가중 N=64' if ko else 'Equal weights, N=64');ax[0].set(xlabel='실수 로그 장 g' if ko else 'Real log field g',ylabel='점유 가지 확률' if ko else 'Occupied-branch probability',title='앞 계수의 중심 이동과 1/N 둥글림 폭' if ko else 'Prefactor-induced center shift and 1/N rounding width',ylim=(-.04,1.42));ax[0].legend(loc='upper left',fontsize=9)
    sizes=np.array([4,8,16,32,64,128,256]);real=-2*np.log(sizes)/sizes;imag=np.pi/sizes
    ax[1].plot(real,imag,'o--',color=blue,ms=4);ax[1].plot(real,-imag,'o--',color=orange,ms=4);ax[1].axvline(0,color='#64748b',lw=.8);ax[1].axhline(0,color='#64748b',lw=.8);ax[1].scatter([0],[0],color=green,s=20)
    for n,dx,dy in [(4,10,0),(16,10,8),(64,-50,25),(256,-90,-48)]:
        x,y=-2*log(n)/n,pi/n;ax[1].annotate(f'N={n}',(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=10,arrowprops={'arrowstyle':'-','color':'#64748b','lw':.7})
    ax[1].set(xlabel='Re(g): 중심 이동과 같은 값' if ko else 'Re(g): same shift as the finite center',ylabel='Im(g)',title='가장 가까운 켤레 영점: kappa=2' if ko else 'Nearest conjugate zeros: kappa=2',xlim=(-.8,.04),ylim=(-.9,.9));save(fig,'degeneracy-shifted-zeros',lang)
