"""Reproduce bilingual cyclic-overlap and probability/energy scientific figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'public/assets/learning'; PREVIEW=ROOT/'.runtime/learning/quantum-state-research'; PREVIEW.mkdir(parents=True,exist_ok=True)
font=next(p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','svg.hashsalt':'suanlab-quantum-state-research','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'axes.labelcolor':'#1e293b','text.color':'#1e293b','figure.facecolor':'white'})
BLUE,ORANGE,GREEN='#2563eb','#c2410c','#047857'
def save(fig,name,lang):
    p=OUT/f'quantum-state-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.25,metadata={'Date':None});p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n');fig.savefig(PREVIEW/f'{name}.{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.25);plt.close(fig)
for lang in ['ko','en']:
    ko=lang=='ko';fig,axes=plt.subplots(3,1,figsize=(8,11),constrained_layout=True);angle=np.linspace(-np.pi+.001,np.pi-.001,900);real=(1+np.cos(angle))/4;imag=np.sin(angle)/4
    axes[0].plot(real,imag,color=GREEN);axes[0].set_aspect('equal');axes[0].axhline(0,color='#cbd5e1',lw=1)
    for sign,color in [(-1,ORANGE),(1,BLUE)]:
        axes[0].scatter([.25],[sign*.25],color=color,s=60);axes[0].annotate('phi = '+('-'if sign<0 else'+')+'pi/2',(.25,sign*.25),xytext=(15,0),textcoords='offset points',color=color)
    axes[0].scatter([0],[0],facecolors='white',edgecolors='#64748b',s=45,zorder=4)
    axes[0].set(xlabel='Re B',ylabel='Im B',title='같은 실수부, 반대 방향'if ko else'Same real part, opposite orientation',xlim=(-.1,.7),ylim=(-.32,.32))
    axes[1].plot(angle,(1+np.cos(angle))/2,color=BLUE,lw=2);axes[1].set(xlabel='phi [rad]',ylabel='F_bc',title='fidelity는 phi의 짝함수'if ko else'Fidelity is even in phi',ylim=(0,1.05))
    axes[2].plot(angle,angle/2,color=ORANGE,lw=2);axes[2].set(xlabel='phi [rad]',ylabel='Arg B [rad]',title='위상은 phi의 홀함수: 끝점 제외'if ko else'Phase is odd in phi: endpoints excluded')
    for ax in axes:ax.grid(alpha=.15)
    save(fig,'cyclic-ray-phase',lang)
    fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True);n=np.arange(2,33)
    a.plot(n,1/n,color=BLUE,lw=2);a.scatter([2,4,8,16],[.5,.25,.125,.0625],color=BLUE);a.set(xlabel='목표 모드 번호 n'if ko else'Target mode number n',ylabel='TV_X upper bound',title='고정 사영 Pi_0: 확률 오차의 상한'if ko else'Fixed projection Pi_0: probability-error bound',ylim=(0,.55))
    b.plot(n,2*(n+1)*(2*n+1)/n,color=ORANGE,lw=2);b.set(xlabel='목표 모드 번호 n'if ko else'Target mode number n',ylabel='E_gradient / E_L',title='같은 목표 수열의 기울기 에너지'if ko else'Gradient energy of the same target sequence')
    for ax in [a,b]:ax.grid(alpha=.15)
    save(fig,'probability-energy-error',lang)
