"""Bilingual native figures; requires Matplotlib/NumPy and fontconfig/Noto CJK.

Run at repository root. Public SVGs use the book palette. Raster review previews
stay in .runtime and are not shipped. No frontend package is introduced.
"""
from pathlib import Path
from subprocess import check_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

run=Path('.runtime/learning/thermodynamic-laws-research');run.mkdir(parents=True,exist_ok=True)
font=check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
 'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue,orange,green='#2563eb','#c2410c','#047857'
def save(fig,name,lang):
    p=Path(f'public/assets/learning/laws-{name}.{lang}.svg');fig.savefig(p,bbox_inches='tight',pad_inches=.2)
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    fig.savefig(run/f'{name}-{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.2);plt.close(fig)
for lang in ['en','ko']:
    ko=lang=='ko'
    fig,axes=plt.subplots(2,1,figsize=(8.5,9.5),layout='constrained');fig.set_facecolor('white')
    u=np.linspace(0,1,400);Tf=np.sqrt(600*300);Th=600-(600-Tf)*u;Tc=300*600/Th
    axes[0].plot(u,Th,color=orange,lw=2.5,label='Th');axes[0].plot(u,Tc,color=blue,lw=2.5,label='Tc')
    axes[0].axhline(Tf,color='#64748b',ls='--',label='Tf = 424.264 K');axes[0].legend()
    axes[0].set(title='A. '+('유한 물체의 변하는 두 온도' if ko else 'Two changing finite-body temperatures'),ylabel='T (K)',xlabel='u ('+('경로 위치' if ko else 'path position')+')')
    Ah=600-Th;Ac=Tc-300
    for ys,color,label in [(Ah,orange,'Ah'),(Ac,blue,'Ac'),(Ah-Ac,green,'W = Ah - Ac')]:axes[1].plot(u,ys,color=color,lw=2.5,label=label)
    axes[1].legend();axes[1].set(title='B. '+('제거한 열·전달한 열·추출한 일' if ko else 'Removed heat, deposited heat and extracted work'),xlabel='u ('+('시간 아님' if ko else 'not time')+')',ylabel='A, W (J)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('유한 열원과 최대 일' if ko else 'Finite sources and maximum work',fontsize=16);save(fig,'finite-source-work',lang)

    fig,axes=plt.subplots(2,1,figsize=(8.5,9.5),layout='constrained');fig.set_facecolor('white')
    labels=['A','B','C','D'];work=[500,650,800,740];reset=[0,150,300,300]
    axes[0].bar(labels,work,color=[blue,orange,green,orange]);axes[0].set(title='A. '+('복원 전 운전의 일' if ko else 'Work before device restoration'),ylabel='W (J)',ylim=(0,900))
    axes[1].bar(labels,[w-r for w,r in zip(work,reset)],color=blue);axes[1].set(title='B. '+('가역 복원을 포함한 순일' if ko else 'Net work including a reversible reset'),ylabel='Net W (J)',ylim=(0,900),xlabel='A–D: '+('본문 표의 네 수지' if ko else 'the four table budgets'))
    for ax in axes:
        ax.axhline(500,color='#64748b',ls='--',label='500 J '+('순환 기준' if ko else 'cyclic reference'));ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('장치의 끝 상태와 복원 비용' if ko else 'Device endpoints and restoration cost',fontsize=16);save(fig,'unrestored-device-reset',lang)

    fig,axes=plt.subplots(2,1,figsize=(8.5,9.5),layout='constrained');fig.set_facecolor('white')
    N=np.logspace(1,6,400);x=np.log(N);q=1/(1+N);P0=np.exp(-N*np.log1p(1/N))
    axes[0].semilogx(N,q,color=blue,lw=2.5,label='q ('+('복사본 하나' if ko else 'one copy')+')')
    axes[0].semilogx(N,1-P0,color=orange,lw=2.5,label='P(M ≥ 1) ('+('전체' if ko else 'collection')+')')
    axes[0].set(title='A. '+('드문 사건과 늘어나는 기회' if ko else 'Rare events, increasing opportunities'),xlabel='N',ylabel='P',ylim=(0,.75));axes[0].legend()
    S=N*np.log1p(1/N)+x*N*q;C=N*x*x*q*(1-q)
    axes[1].semilogx(N,S/x,color=blue,lw=2.5,label='S / (kB log N)')
    axes[1].semilogx(N,C/x**2,color=green,lw=2.5,label='C / (kB (log N)^2)')
    axes[1].axhline(1,color='#64748b',ls='--');axes[1].set(title='B. '+('총량 증가를 나눈 비율: a = 0' if ko else 'Normalize growing totals: a = 0'),xlabel='N',ylabel=''+('무차원 비율' if ko else 'Dimensionless ratio'));axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('독립 복사본의 크기·저온 동시 극한' if ko else 'Growing independent copies at decreasing temperature',fontsize=16);save(fig,'growing-copy-low-temperature',lang)
