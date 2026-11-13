"""Reproduce six native SVG figures for partition-function research lessons."""
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue,orange,green,purple='#2563eb','#c2410c','#047857','#7c3aed'
out=Path('public/assets/learning');preview=Path('.runtime/learning/partition-research');preview.mkdir(parents=True,exist_ok=True)
def save(fig,name,lang):
    fig.tight_layout(pad=2)
    p=out/f'partition-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.2)
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    fig.savefig(preview/f'{name}.{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.2);plt.close(fig)
q=np.array([4,2,1])/7
for lang in ['en','ko']:
    ko=lang=='ko';fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));i=np.arange(3);M=np.logspace(0,3,100)
    for offset,m,label,color in [(-.25,0,'균일 목표 m=0' if ko else 'Uniform target m=0',orange),(0,1,'출발 m=1' if ko else 'Source m=1',blue),(.25,3,'차가운 목표 m=3' if ko else 'Colder target m=3',green)]:
        w=2.**(-m*i);p=w/w.sum();ax[0].bar(i+offset,p,width=.23,label=label,color=color)
        if m!=1:
            v=sum((p-q)**2/q);ax[1].loglog(M,np.sqrt(v/M),color=color,label=f'm={m}')
    p2=2.**(-2*i);p2=p2/p2.sum();ax[1].loglog(M,np.sqrt(sum((p2-q)**2/q)/M),color=purple,label='m=2')
    ax[0].set(xticks=i,xlabel='에너지 / 간격' if ko else 'Energy / gap',ylabel='정준 확률' if ko else 'Canonical probability',ylim=(0,1.08),title='같은 스펙트럼, 다른 강조' if ko else 'Same spectrum, different emphasis');ax[0].legend(ncol=3,fontsize=9,loc='upper center')
    ax[1].axvline(100,color='#64748b',ls=':');ax[1].set(xlabel='독립 준비 수 M' if ko else 'Independent preparations M',ylabel='비 추정량의 이론적 상대 표준오차' if ko else 'Theoretical relative SE of ratio estimate',title='표본 수와 분포 불일치를 구별' if ko else 'Separate sample count from distribution mismatch');ax[1].legend();save(fig,'finite-reweighting',lang)
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));b=np.linspace(0,2,160);k=.5*np.log(np.cosh(2*b));corr=np.tanh(b)**2;same=np.cosh(2*b)/(1+np.cosh(2*b));between=4*np.tanh(2*b)**2*same*(1-same);hidden=4*same/np.cosh(2*b)**2;total=2/np.cosh(b)**2
    ax[0].plot(b,k,color=blue,label='유도 결합 k' if ko else 'Induced coupling k');ax[0].plot(b,corr,color=orange,label='끝점 상관' if ko else 'Endpoint correlation');ax[0].set(ylabel='무차원 값' if ko else 'Dimensionless value',title='가운데 스핀을 합쳐 나타나는 끝점 의존성' if ko else 'Endpoint dependence after summing the middle spin');ax[0].legend(loc='upper left')
    ax[1].plot(b,total,color=green,ls='--',label='전체' if ko else 'Full variance');ax[1].plot(b,hidden,color=orange,label='숨은 조건부 분산' if ko else 'Hidden conditional variance');ax[1].plot(b,between,color=blue,label='조건부 평균 사이 분산' if ko else 'Between conditional means');ax[1].set(ylabel='에너지 분산 / J squared' if ko else 'Energy variance / J squared',title='정확한 주변 확률과 남아 있는 숨은 요동' if ko else 'Exact marginal probabilities retain hidden fluctuations');ax[1].legend(loc='upper right')
    for axis in ax:axis.set_xlabel('무차원 역온도 b=beta x J' if ko else 'Dimensionless inverse temperature b=beta x J')
    save(fig,'hidden-spin-marginal',lang)
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));L=np.arange(81)
    for t,color,label in [(.75,blue,'유한' if ko else 'finite'),(.5,orange,'경계: 발산' if ko else 'threshold: divergent'),(.4,green,'발산' if ko else 'divergent'),(.25,purple,'발산' if ko else 'divergent')]:
        A=(-np.expm1(-t))**2/(-np.expm1(-1));r=np.exp(1-2*t);partial=A*np.cumsum(r**L)
        ax[0].semilogy(L,partial,color=color,label=f't={t}: {label}');ax[1].semilogy(L,np.exp(-t*(L+1)),color=color,label=f't={t}')
    ax[0].set(ylabel='원래 출발의 부분 이차 모멘트' if ko else 'Partial second moment under original source',title='유한 절단은 발산을 숨길 수 있다 (s=1)' if ko else 'Finite cutoffs can hide divergence (s=1)');ax[0].legend(loc='upper left')
    ax[1].set(ylabel='빠진 일차 모멘트' if ko else 'Omitted first moment',title='같은 목표에서도 일차 모멘트는 수렴' if ko else 'The first moment still converges for the same targets');ax[1].legend(loc='lower left')
    for axis in ax:axis.set_xlabel('절단 L (물리 시간이 아님)' if ko else 'Cutoff L (not physical time)')
    save(fig,'infinite-weight-variance',lang)
