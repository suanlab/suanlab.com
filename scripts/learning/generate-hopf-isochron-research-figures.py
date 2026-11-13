"""Reproducible bilingual isochron and radial-pulse plots."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(FONT).exists():font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'svg.fonttype':'none','svg.hashsalt':'hopf-isochron-research','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
BLUE='#2563eb';ORANGE='#c2410c';GREEN='#047857'
for lang in ['ko','en']:
    ko=lang=='ko';fig,ax=plt.subplots(2,1,figsize=(7,8.2),gridspec_kw={'height_ratios':[1.25,1]});fig.subplots_adjust(hspace=.46)
    r=np.linspace(.23,1.8,400);angles=np.linspace(0,2*np.pi,300)
    for n in range(8):
        phi=n*np.pi/4;theta=phi+np.log(r);ax[0].plot(r*np.cos(theta),r*np.sin(theta),color=BLUE if n%2==0 else ORANGE,lw=1.2,alpha=.8)
    ax[0].plot(np.cos(angles),np.sin(angles),'--',color=GREEN,lw=1.5,label='끌개 원 r*=1'if ko else 'Attracting cycle r*=1')
    ax[0].plot([0],[0],marker='o',mfc='white',mec='#64748b',ms=5,label='원점: 위상 없음'if ko else 'Origin: phase undefined')
    # Equal phase φ=0, at radii 0.5 and 1.5.
    for radius in [.5,1.5]:ax[0].scatter(radius*np.cos(np.log(radius)),radius*np.sin(np.log(radius)),color=BLUE,s=27,zorder=5)
    ax[0].set_aspect('equal');ax[0].set(xlabel='x',ylabel='y',title='같은 나중 시계 눈금: φ = θ − ln r'if ko else 'Same eventual clock reading: φ = θ − ln r');ax[0].legend(loc='upper left',fontsize=8);ax[0].grid(alpha=.12)
    k=np.linspace(-.65,.65,400)
    ax[1].plot(k,-np.log1p(k),color=BLUE,lw=1.7,label='정확 −ln(1+κ)'if ko else 'Exact −ln(1+κ)');ax[1].plot(k,-k,color=ORANGE,ls='--',label='선형 −κ'if ko else 'Linear −κ')
    ax[1].scatter([.5],[-np.log(1.5)],color=GREEN,zorder=5,label='κ=0.5');ax[1].axhline(0,color='#94a3b8',lw=.6);ax[1].axvline(0,color='#94a3b8',lw=.6)
    ax[1].set(xlabel='방사 자극 κ = h_radial/r*'if ko else 'Radial impulse κ = h_radial/r*',ylabel='남는 위상 변화 [rad]'if ko else 'Lasting phase shift [rad]',title='각도를 즉시 바꾸지 않아도 위상은 바뀐다'if ko else 'A radial displacement leaves a lasting phase shift');ax[1].grid(alpha=.15);ax[1].legend(fontsize=8)
    name=f'hopf-isochron-research-phase-response.{lang}';target=ROOT/'public/assets/learning'/f'{name}.svg';fig.savefig(target,bbox_inches='tight',pad_inches=.25,metadata={'Date':None});target.write_text('\n'.join(line.rstrip()for line in target.read_text().splitlines())+'\n')
    runtime=ROOT/'.runtime/learning/hopf-isochron-research';runtime.mkdir(parents=True,exist_ok=True);fig.savefig(runtime/f'phase-response.{lang}.png',dpi=150,bbox_inches='tight',pad_inches=.25);plt.close(fig)
