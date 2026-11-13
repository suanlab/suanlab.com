"""Native bilingual figures for equilibrium criticality, branch weights and interfaces."""
from pathlib import Path
from math import gamma, sqrt
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from importlib.util import spec_from_file_location, module_from_spec

sys.dont_write_bytecode=True
spec=spec_from_file_location('verify',Path(__file__).with_name('verify-phase-transition-research.py'))
verify=module_from_spec(spec);spec.loader.exec_module(verify)
font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue,orange,green,purple='#2563eb','#c2410c','#047857','#7c3aed'
out=Path('public/assets/learning');preview=Path('.runtime/learning/phase-transition-research');preview.mkdir(parents=True,exist_ok=True)

def save(fig,name,lang):
    fig.tight_layout(pad=2)
    p=out/f'phase-transition-research-{name}.{lang}.svg'
    fig.savefig(p,bbox_inches='tight',pad_inches=.2)
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    fig.savefig(preview/f'{name}.{lang}.png',dpi=110,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)

for lang in ('en','ko'):
    ko=lang=='ko'
    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5))
    c2=sqrt(12)*gamma(.75)/gamma(.25);binder=1-1/c2**2
    sizes=[32,128,512,2048];us=[]
    for n,col in zip(sizes,[orange,blue,green,purple]):
        ms,ps,_=verify.law(n);xs=np.array(ms)*n**.25
        ax[0].plot(xs,np.array(ps)/(2/n**.75),color=col,label=f'N={n}',lw=1.5)
        e2=sum(p*x*x for x,p in zip(xs,ps));e4=sum(p*x**4 for x,p in zip(xs,ps));us.append(1-e4/(3*e2**2))
    x=np.linspace(-3.2,3.2,400)
    ax[0].plot(x,2/(12**.25*gamma(.25))*np.exp(-x**4/12),color='#334155',ls='--',label='사차 극한' if ko else 'Quartic limit')
    ax[0].set(xlim=(-3.2,3.2),ylim=(0,.44),xlabel='척도화한 자화 X = M / N^(3/4)' if ko else 'Scaled magnetization X = M / N^(3/4)',ylabel='격자 질량 / 간격' if ko else 'Lattice mass / spacing',title='K=1, b=0: 임계 요동의 밀도 비교' if ko else 'K=1, b=0: critical fluctuation density');ax[0].legend(loc='upper right',fontsize=9)
    ax[1].semilogx(sizes,us,'o-',color=blue,label='유한합 U_N' if ko else 'Finite-sum U_N');ax[1].axhline(binder,color=orange,ls='--',label=f'U* = {binder:.6f}');ax[1].set(xlabel='준비 크기 N' if ko else 'Preparation size N',ylabel='Binder U',ylim=(.25,.37),title='비정규 모양으로 접근: 유한계는 여전히 해석적' if ko else 'Approaching a non-Gaussian shape; finite laws stay analytic');ax[1].legend();save(fig,'critical-fluctuations',lang)

    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));root=verify.positive_root(2);vs=np.linspace(-2,2,61)
    for n,col in [(32,orange),(128,blue)]:
        means=[]
        for v in vs:
            ms,ps,_=verify.law(n,2,v/n);means.append(sum(p*m for m,p in zip(ms,ps)))
        ax[0].plot(vs,means,color=col,label=f'N={n}')
    ax[0].plot(vs,root*np.tanh(vs*root),color=green,ls='--',label='혼합 극한' if ko else 'Mixture limit');ax[0].set(xlabel='축소장 척도 v = N b' if ko else 'Shrinking-field scale v = N b',ylabel='평균 m' if ko else 'Mean m',title='K=2, b=v/N: 작은 장과 큰 혼합 변화' if ko else 'K=2, b=v/N: small fields, finite mixture changes');ax[0].legend()
    ns=[16,32,64,128,256,512];responses=[]
    for n in ns:
        ms,ps,_=verify.law(n,2,0);responses.append(n*sum(p*m*m for m,p in zip(ms,ps)))
    ax[1].loglog(ns,responses,'o-',color=blue,label='유한 평형 응답' if ko else 'Finite equilibrium response');ax[1].loglog(ns,np.array(ns)*root**2,color=green,ls='--',label='N m*²');ax[1].axhline(1/(1/(1-root**2)-2),color=orange,ls=':',label='선택 가지 미분' if ko else 'Selected-branch derivative');ax[1].set(xlabel='준비 크기 N' if ko else 'Preparation size N',ylabel='장 응답 (b=0)' if ko else 'Field response (b=0)',ylim=(.04,3000),title='골짜기 사이 분산과 내부 분산은 다른 척도' if ko else 'Between-valley and within-valley fluctuations differ');ax[1].legend(loc='upper left',fontsize=10);save(fig,'laplace-branch-weights',lang)

    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));x=np.linspace(-4,4,400);u=1/(1+np.exp(-x/.5))
    ax[0].plot(x,np.sqrt(u),color=blue,label='m / m0');ax[0].plot(x,u,color=orange,ls='--',label='u = (m/m0)²');ax[0].axvline(0,color='#64748b',ls=':');ax[0].set(xlabel='위치 x (m)' if ko else 'Position x (m)',ylabel='무차원 질서변수' if ko else 'Dimensionless order parameter',ylim=(-.04,1.3),title='공존 0 → m0 계면: ell=0.5 m' if ko else 'Coexistent 0 → m0 interface: ell=0.5 m');ax[0].legend(loc='upper left')
    r=np.linspace(0,1.65,300)
    ax[1].plot(r,3*r*r,color=orange,ls='--',label='표면 3r²' if ko else 'Surface 3r²');ax[1].plot(r,-2*r**3,color=green,ls=':',label='벌크 -2r³' if ko else 'Bulk -2r³');ax[1].plot(r,3*r*r-2*r**3,color=blue,label='합계' if ko else 'Total');ax[1].scatter([1],[1],color=purple,zorder=3);ax[1].axhline(0,color='#64748b',lw=.8);ax[1].set(xlabel='반지름 r = R / Rc' if ko else 'Radius r = R / Rc',ylabel='자유에너지 / 장벽' if ko else 'Free energy / barrier',ylim=(-9.5,9.5),title='얇은 계면 근사: 구형 방울의 정적 장벽' if ko else 'Thin-wall approximation: a spherical droplet barrier');ax[1].legend(loc='upper left',fontsize=10);save(fig,'interface-droplet',lang)
