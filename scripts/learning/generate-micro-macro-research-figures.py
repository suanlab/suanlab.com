"""Reproduce the six bilingual figures for the finite micro/macro extensions."""
from pathlib import Path
from math import comb, log
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK KR'], text=True)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(), 'font.size': 12, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
blue, orange, green = '#2563eb', '#c2410c', '#047857'
out = Path('public/assets/learning'); preview = Path('.runtime/learning/micro-macro-research'); preview.mkdir(parents=True, exist_ok=True)
def save(fig, name, lang):
    fig.tight_layout(pad=2.0)
    path = out / f'micro-macro-{name}.{lang}.svg'
    fig.savefig(path, bbox_inches='tight', pad_inches=.2)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(preview / f'{name}.{lang}.png', dpi=120, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)
a = np.arange(5); weights = np.array([comb(4,int(x))**2 for x in a]); pi = weights/70
T = np.zeros((5,5))
for x in range(5):
    birth, death = .5*(4-x)**2/16, .5*x*x/16
    if x<4: T[x,x+1]=birth
    if x>0: T[x,x-1]=death
    T[x,x]=1-birth-death
P = np.ones(5)/5; histories=[]; P10=None
for t in range(16):
    hc = -sum(p*log(p) for p in P)
    histories.append([hc,hc+sum(P*np.log(weights))])
    if t==10: P10=P.copy()
    P=P@T
r=np.linspace(0,1,101); q=.5-r/6; history=2*(r/3)*(1-2*r/3)/(1-r/3); actual=2*(r/3)*(1-2*r/3); naive=(1-r/3)*q
n=np.arange(1,11); e=.1
def h(p): return 0 if p in (0,1) else -p*log(p)-(1-p)*log(1-p)
info=[]
for count in n:
    hidden=0
    for k in range(count+1):
        p0=.5*comb(int(count),k)*e**k*(1-e)**(count-k)
        p1=.5*comb(int(count),k)*(1-e)**k*e**(count-k)
        hidden+=(p0+p1)*h(p1/(p0+p1))
    info.append(log(2)-hidden)
for lang in ['en','ko']:
    ko=lang=='ko'
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5))
    for offset,data,label,color in [(-.25,np.ones(5)/5,'초기 t=0' if ko else 'Initial t=0',orange),(0,P10,'시도 t=10' if ko else 'Attempt t=10',blue),(.25,pi,'정상 분포' if ko else 'Stationary law',green)]: ax[0].bar(a+offset,data,width=.23,label=label,color=color)
    ax[0].set(xlabel='블록 A의 들뜸 수' if ko else 'Excitations in block A',ylabel='확률' if ko else 'Probability',xticks=a,title='같은 평균 2, 다른 분포' if ko else 'Same mean 2, different distributions'); ax[0].legend()
    ax[1].plot(range(16),np.array(histories)[:,1],color=blue,label='전체 미시 상태 H' if ko else 'Full microstate H')
    ax[1].plot(range(16),np.array(histories)[:,0],color=orange,label='블록 수 H' if ko else 'Block count H')
    ax[1].axhline(log(70),color=green,ls=':',label='ln 70')
    ax[1].set(xlabel='교환 시도 횟수 (물리 시간이 아님)' if ko else 'Exchange attempts (not physical seconds)',ylabel='정보 엔트로피 (nat)' if ko else 'Information entropy (nats)',title='고정 껍질, 조건부 균일 준비' if ko else 'Fixed shell, conditionally uniform preparation'); ax[1].legend()
    save(fig,'shell-exchange',lang)
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5))
    ax[0].plot(r,q,color=blue,label='A만 관측' if ko else 'Given A only'); ax[0].plot(r,history,color=orange,label='B, A를 관측' if ko else 'Given history B, A')
    ax[0].set(ylabel='다음 B의 조건부 확률' if ko else 'Conditional probability of next B',title='숨겨진 상태의 기억' if ko else 'Memory in the hidden state'); ax[0].legend()
    ax[1].plot(r,actual,color=blue,label='실제 A, B | B' if ko else 'Actual A, B | B'); ax[1].plot(r,naive,color=orange,ls='--',label='한 단계 표 Q의 곱' if ko else 'Product of one-step Q entries')
    ax[1].set(ylabel='사건 확률' if ko else 'Event probability',title='정상 한 단계 표의 예측 오차' if ko else 'Prediction error of the stationary one-step table'); ax[1].legend()
    for axis in ax: axis.set_xlabel('균일 재추출 확률 r' if ko else 'Uniform reset probability r'); axis.set_xlim(0,1)
    save(fig,'coarse-memory',lang)
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5))
    ax[0].plot(n,info,'o-',color=blue,label='독립 오류' if ko else 'Independent errors'); ax[0].axhline(log(2)-h(e),color=orange,label='공유 오류' if ko else 'Shared error'); ax[0].axhline(log(2),ls=':',color=green,label='입력 엔트로피 ln 2' if ko else 'Input entropy ln 2')
    ax[0].set(ylabel='입력에 대한 정보 (nat)' if ko else 'Information about input (nats)',title='같은 단일 센서, 다른 결합 법칙' if ko else 'Same single sensor, different joint laws'); ax[0].legend()
    ax[1].plot(n,1/(1+(e/(1-e))**n),'o-',color=blue,label='독립 모형의 사후확률' if ko else 'Independent-model posterior'); ax[1].axhline(1-e,color=orange,label='공유 모형의 사후확률' if ko else 'Shared-model posterior')
    ax[1].set(ylabel='모두 1일 때 P(X=1)' if ko else 'P(X=1) after all-one readings',ylim=(.88,1.01),title='공유 오류에 독립 가정을 적용하면 과신' if ko else 'Independence applied to shared errors overstates confidence'); ax[1].legend()
    for axis in ax: axis.set_xlabel('고정 입력의 반복 관측 수 n' if ko else 'Readings n of one fixed input'); axis.set_xticks(n)
    save(fig,'repeated-sensors',lang)
