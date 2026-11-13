"""Bilingual native figures for path likelihoods and estimator-specific efficiency."""
from pathlib import Path
import subprocess,sys
from fractions import Fraction as F
from math import log
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from importlib.util import spec_from_file_location,module_from_spec
sys.dont_write_bytecode=True
spec=spec_from_file_location('verify',Path(__file__).with_name('verify-detailed-balance-research.py'));v=module_from_spec(spec);spec.loader.exec_module(v)
font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue,orange,green='#2563eb','#c2410c','#047857';out=Path('public/assets/learning');preview=Path('.runtime/learning/detailed-balance-research');preview.mkdir(parents=True,exist_ok=True)
def save(fig,name,lang):
    fig.tight_layout(pad=2);p=out/f'detailed-balance-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.2);p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');fig.savefig(preview/f'{name}.{lang}.png',dpi=110,bbox_inches='tight',pad_inches=.2);plt.close(fig)
for lang in ('en','ko'):
    ko=lang=='ko';fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));law=v.cycle_law(12,F(1,2),F(1,4));ds=sorted(law);xs=np.array(ds)*log(2);ps=[float(law[d]) for d in ds]
    ax[0].bar(xs,ps,width=.55,color=[orange if x<0 else green if x==0 else blue for x in xs]);ax[0].set(xlabel='경로 로그 비 S (nat)' if ko else 'Path log ratio S (nats)',ylabel='확률 질량' if ko else 'Probability mass',title='정상 삼각형, n=12: 음의 S도 가능한 유한 경로' if ko else 'Stationary triangle, n=12: finite paths can have negative S')
    ys=[log(float(law[d]/law[-d])) for d in ds];ax[1].plot(xs,xs,color=orange,ls='--',label='y = S');ax[1].scatter(xs,ys,color=blue,s=20,label='정확한 확률의 로그 비' if ko else 'Log ratio of exact probabilities');ax[1].set(xlabel='S (nat)' if ko else 'S (nats)',ylabel='log [P(S) / P(-S)]',title='역순 경로 짝의 유한 시간 대칭' if ko else 'Finite-time symmetry from reversed path pairs');ax[1].legend(loc='upper left',fontsize=10);save(fig,'path-irreversibility',lang)

    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));x=np.linspace(-4,4,200);ratio=np.exp(x);ax[0].plot(x,np.minimum(1,ratio),color=blue,label='Metropolis');ax[0].plot(x,ratio/(1+ratio),color=orange,label='Barker');ax[0].set(xlabel='로그 제안·목표 비 log R' if ko else 'Log target/proposal ratio log R',ylabel='조건부 수락' if ko else 'Conditional acceptance',title='같은 양의 비 R, 서로 다른 올바른 수락 규칙' if ko else 'Same positive ratio R, different valid acceptance rules',ylim=(-.03,1.3));ax[0].legend(loc='upper left')
    stays=list(range(21))
    for barker,col,label in [(False,blue,'Metropolis'),(True,orange,'Barker')]:
        variances=[]
        for s in stays:
            pi,p=v.kernel((1,2,4),2,1,s,barker);phi=[F(i==2)-pi[2] for i in range(3)];variances.append(float(v.poisson_variance(pi,p,phi)))
        ax[1].plot(stays,variances,color=col,label=label)
    ax[1].set(xlabel='자기 제안 가중치 s (정방향2, 역방향1)' if ko else 'Self-proposal weight s (forward 2, backward 1)',ylabel='점근 분산 계수 sigma²' if ko else 'Asymptotic variance coefficient sigma²',title='상태2 점유율: 같은 목표, 시도 횟수로 비교' if ko else 'State-2 occupancy: same target, proposal-attempt clock');ax[1].legend(loc='upper left');save(fig,'acceptance-variance-order',lang)

    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));ns=np.arange(1,61)
    for a,col,label in [(F(9,10),blue,'원래 a=b=0.9' if ko else 'Original a=b=0.9'),(F(1,2),orange,'재설정 a=b=0.5' if ko else 'Reset a=b=0.5')]:
        mse=[float(v.binary_error(int(n),a,a,F(0))[2]) for n in ns];ax[0].semilogy(ns,mse,color=col,label=label)
    ax[0].set(xlabel='기록한 상태 수 n (초기 상태 포함)' if ko else 'Recorded state count n (including initial state)',ylabel='정확한 유한 MSE' if ko else 'Exact finite MSE',title='u=0: 빠른 분포 혼합이 평균 오차의 전부는 아님' if ko else 'u=0: fast distribution mixing is not the whole mean error');ax[0].legend(loc='upper right',fontsize=10)
    delta=np.linspace(0,.7,220);lam=delta+(1-delta)*(-.8);tau=(1+lam)/(1-lam)
    ax[1].plot(delta,np.abs(lam),color=blue,ls='--',label='분포 수축 |lambda_delta|' if ko else 'Distribution contraction |lambda_delta|');ax[1].plot(delta,tau,color=orange,label='분산 팽창 tau_delta' if ko else 'Variance inflation tau_delta');ax[1].axvline(4/9,color=green,ls=':',label='재설정 delta=4/9' if ko else 'Reset delta=4/9');ax[1].set(xlabel='추가 머무름 확률 delta' if ko else 'Added holding probability delta',ylabel='무차원 지표 (범례별 의미 다름)' if ko else 'Dimensionless quantities (see each legend)',title='원래 lambda=-0.8: 두 목적의 다른 최적점' if ko else 'Original lambda=-0.8: different objectives, different optima',ylim=(-.08,3.5));ax[1].legend(loc='upper left',fontsize=9);save(fig,'initialization-laziness-error',lang)
