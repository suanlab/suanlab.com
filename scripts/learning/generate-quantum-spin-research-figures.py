"""Bilingual spin readout information and coherent/cat alias comparisons."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'public/assets/learning';PREVIEW=ROOT/'.runtime/learning/quantum-spin-research';PREVIEW.mkdir(parents=True,exist_ok=True)
font=next(p for p in font_manager.findSystemFonts()if'NotoSansCJK-Regular'in p);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','svg.hashsalt':'suanlab-quantum-spin-research','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'axes.labelcolor':'#1e293b','text.color':'#1e293b','figure.facecolor':'white'})
BLUE,ORANGE,GREEN='#2563eb','#c2410c','#047857'
def save(fig,name,lang):
    p=OUT/f'quantum-spin-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.25,metadata={'Date':None});p.write_text('\n'.join(s.rstrip()for s in p.read_text().splitlines())+'\n');fig.savefig(PREVIEW/f'{name}.{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.25);plt.close(fig)
for lang in ['ko','en']:
    ko=lang=='ko';angle=np.linspace(0,np.pi,601);fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True)
    a.plot(angle,(1+.6*np.sin(angle))/2,color=BLUE,lw=2);a.set(ylabel='p_plus',title='영 확률이 없어도 일차 민감도는 사라진다'if ko else'Positive probabilities can have zero first-order sensitivity',ylim=(.15,.85));a.scatter([np.pi/2],[.8],color=ORANGE,s=50)
    b.plot(angle,9*np.cos(angle)**2/(25-9*np.sin(angle)**2),color=BLUE,lw=2,label='기울어진 검출기 FI'if ko else'Tilted detector FI');b.axhline(1,color=ORANGE,ls='--',label='Quantum FI = 1');b.scatter([np.pi/2],[0],color=BLUE,s=50);b.set(ylabel='Information per trial',title='상태의 정보와 판독의 정보'if ko else'Available state information versus readout information',ylim=(-.05,1.08));b.legend(fontsize=10)
    for ax in [a,b]:ax.axvline(np.pi/2,color=GREEN,ls=':',alpha=.6);ax.set_xticks([0,np.pi/2,np.pi],['0','pi/2','pi']);ax.set_xlabel('Rotation angle phi [rad]');ax.grid(alpha=.15)
    save(fig,'readout-information',lang)
    fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True)
    a.plot(angle,np.cos(angle/2)**8,color=BLUE,lw=2,label='Coherent, j = 2');a.plot(angle,np.cos(2*angle)**2,color=ORANGE,ls='--',lw=2,label='Cat, j = 2');a.axvline(np.pi/2,color=GREEN,ls=':',label='Cat ray return: pi/2');a.set(xlabel='Rotation angle phi [rad]',ylabel='Survival fidelity',title='더 촘촘한 fringe와 반복되는 ray'if ko else'Finer fringes and repeated rays',ylim=(-.05,1.08));a.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi],['0','pi/4','pi/2','3pi/4','pi']);a.legend(fontsize=10)
    j=np.array([.5,1,2,4]);b.plot(j,2*j,'o-',color=BLUE,label='Coherent FI = 2j');b.plot(j,4*j*j,'s--',color=ORANGE,label='Cat FI = 4j^2');b.set(xlabel='Spin label j',ylabel='Quantum information',title='국소 곡률은 전역 식별을 보장하지 않는다'if ko else'Local curvature does not guarantee global identifiability');b.set_xticks(j,['1/2','1','2','4']);b.legend(fontsize=10)
    for ax in [a,b]:ax.grid(alpha=.15)
    save(fig,'coherent-cat-aliases',lang)
