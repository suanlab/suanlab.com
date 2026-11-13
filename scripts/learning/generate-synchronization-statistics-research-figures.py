"""Bilingual deterministic occupation and finite balanced-spectrum figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'public/assets/learning';PREVIEW=ROOT/'.runtime/learning/synchronization-statistics-research';PREVIEW.mkdir(parents=True,exist_ok=True)
font=next(p for p in font_manager.findSystemFonts()if'NotoSansCJK-Regular'in p);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','svg.hashsalt':'suanlab-synchronization-statistics-research','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'axes.labelcolor':'#1e293b','text.color':'#1e293b','figure.facecolor':'white'})
BLUE,ORANGE,GREEN='#2563eb','#c2410c','#047857'
def save(fig,name,lang):
    target=OUT/f'synchronization-statistics-research-{name}.{lang}.svg';fig.savefig(target,bbox_inches='tight',pad_inches=.25,metadata={'Date':None});target.write_text('\n'.join(line.rstrip()for line in target.read_text().splitlines())+'\n');fig.savefig(PREVIEW/f'{name}.{lang}.png',dpi=130,bbox_inches='tight',pad_inches=.25);plt.close(fig)
for lang in ['ko','en']:
    ko=lang=='ko';fig,(a,b)=plt.subplots(2,1,figsize=(8,8.2),constrained_layout=True);angle=np.linspace(0,2*np.pi,1600)
    for k,color in zip([0.,.6,.96],[BLUE,ORANGE,GREEN]):
        q=np.sqrt(1-k*k);a.plot(angle/np.pi,q/(2*np.pi*(1-k*np.sin(angle))),color=color,lw=2,label=f'K={k:g}')
    a.set(xlabel='상대 위상 δ / π'if ko else 'Relative phase δ / π',ylabel='시간 점유 밀도 p(δ)'if ko else 'Time occupation density p(δ)',title='정해진 감속도 점유 봉우리를 만든다'if ko else 'Deterministic slowing produces an occupation peak');a.legend(fontsize=10);a.grid(alpha=.15)
    k=np.linspace(0,.99,500);b.plot(k,2*np.pi/np.sqrt(1-k*k),color=BLUE,lw=2)
    values=np.array([0.,.6,.8,.96]);b.scatter(values,2*np.pi/np.sqrt(1-values*values),color=ORANGE,zorder=5)
    b.set(xlabel='결합 K [1/s], Δ=1 [1/s]'if ko else 'Coupling K [1/s], Δ=1 [1/s]',ylabel='한 바퀴 시간 [s]'if ko else 'Full-turn duration [s]',title='경계 가까이서는 한 바퀴에도 긴 시간이 필요하다'if ko else 'A full turn takes longer near the locking boundary');b.grid(alpha=.15);save(fig,'slip-occupation',lang)
    fig=plt.figure(figsize=(8,7.7),constrained_layout=True);grid=fig.add_gridspec(2,3,height_ratios=[1,1.3]);theta=np.linspace(0,2*np.pi,300)
    configs=[([(1,0),(-1,0)],'반대 위상 · R2=1'if ko else 'Antiphase · R2=1'),([(1,0),(0,1),(-1,0),(0,-1)],'정사각형 · R2=0'if ko else 'Square · R2=0'),([(1,0),(-1,0),(.6,.8),(-.6,-.8)],'기울인 쌍 · R2=0.6'if ko else 'Tilted pairs · R2=0.6')]
    for index,(points,title)in enumerate(configs):
        a=fig.add_subplot(grid[0,index]);a.plot(np.cos(theta),np.sin(theta),color='#94a3b8',lw=1);a.scatter([p[0]for p in points],[p[1]for p in points],color=BLUE,s=45,zorder=5);a.set_aspect('equal');a.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),xticks=[],yticks=[],title=title);a.spines[['left','bottom']].set_visible(False)
        if index==0:
            for x,y in points:a.annotate('×2',(x,y),xytext=(-9,12),textcoords='offset points',fontsize=10)
    b=fig.add_subplot(grid[1,:]);r=np.linspace(0,1,300);b.plot(r,(1+r)/2,color=BLUE,lw=2,label='λ+/K');b.plot(r,(1-r)/2,color=ORANGE,lw=2,ls='--',label='λ-/K')
    for value in [0,.6,1]:b.scatter([value,value],[(1+value)/2,(1-value)/2],color=GREEN,s=35,zorder=5)
    b.set(xlabel='이차 질서도 R2, 정확 균형 R1=0'if ko else 'Second order R2, exact balance R1=0',ylabel='교란 성장률 / K'if ko else 'Disturbance growth rate / K',title='같은 영 질서도에서도 성장률은 다르다'if ko else 'Zero first order can hide different growth rates',ylim=(-.04,1.08));b.legend(fontsize=11);b.grid(alpha=.15);save(fig,'balanced-spectrum',lang)
