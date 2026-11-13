"""Render native bilingual book figures (Matplotlib/NumPy, fontconfig/Noto CJK).

Run at the repository root. Raster previews remain under .runtime. Plot colors
follow the book's blue/orange/green palette; no frontend dependency is added.
"""
from pathlib import Path
from subprocess import check_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

runtime=Path('.runtime/learning/heat-work-research');runtime.mkdir(parents=True,exist_ok=True)
font_path=check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True)
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),
 'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
colors=['#2563eb','#c2410c','#047857']

def save(fig,name,lang):
    path=Path(f'public/assets/learning/heat-work-{name}.{lang}.svg')
    # Keep native SVG titles and rotated labels inside the exported viewport.
    fig.savefig(path,bbox_inches='tight',pad_inches=.25)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(runtime/f'{name}-{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.25)
    plt.close(fig)

for lang in ['en','ko']:
    ko=lang=='ko'
    fig,axes=plt.subplots(2,1,figsize=(8.5,9.5),layout='constrained');fig.set_facecolor('white')
    time=np.linspace(0,12,400);decay=np.exp(-time/3)
    ax=axes[0];ax.plot(time,294+36*decay,color=colors[0],lw=2.5,label='T1')
    ax.plot(time,294-24*decay,color=colors[2],lw=2.5,label='T2')
    ax.axhline(294,color='#64748b',ls='--',label='Te = 294 K')
    ax.set(xlabel='t (s)',ylabel='T (K)',title='A. '+('모든 공통 척도에서 같은 온도' if ko else 'Same temperatures for every common scale'))
    ax.text(.97,.93,'s = 0.5, 1, 2',transform=ax.transAxes,ha='right',va='top');ax.legend(loc='center right');ax.grid(alpha=.2)
    ax=axes[1]
    for s,color in zip([.5,1,2],colors):ax.plot(time,s*72*(1-decay),color=color,lw=2.5,label=f's = {s:g}')
    ax.set(xlabel='t (s)',ylabel='Q (1→2) (J)',title='B. '+('온도에 보이지 않는 에너지 척도' if ko else 'Different energy scales hidden from temperatures'))
    ax.grid(alpha=.2);ax.legend(loc='lower right')
    fig.suptitle('온도 적합과 열적 척도 식별' if ko else 'Temperature fitting and thermal-scale identification',fontsize=16)
    save(fig,'contact-scale',lang)

    fig,axes=plt.subplots(2,1,figsize=(8.5,9.5),layout='constrained');fig.set_facecolor('white')
    u=np.linspace(0,1,400);ax=axes[0]
    for c,color in zip([-80,0,80],colors):
        temperature=300+30*u+40*u*(1-u)+c*u*(1-u)*(1-2*u)
        ax.plot(u,temperature,color=color,lw=2.5,label=f'c = {c:g} K')
        ax.scatter([.25],[315+3*c/32],color=color,s=35,zorder=5)
    ax.axvline(.25,color='#64748b',ls=':',lw=1)
    ax.scatter([.5],[325],color='#334155',s=40,zorder=6)
    ax.set(xlabel='u = log(V/V0) / log(V1/V0)',ylabel='T (K)',title='A. '+('b = 40 K: 같은 끝점·중간, 다른 이력' if ko else 'b = 40 K: same endpoints and midpoint, different histories'),ylim=(295,345))
    ax.legend(loc='upper left',ncol=3,fontsize=10);ax.grid(alpha=.2)
    ax=axes[1];cs=np.linspace(-100,100,200)
    for b,color in zip([0,40],colors[:2]):
        work=8.31446261815324*np.log(2)*(315+b/6)
        ax.plot(cs,np.full_like(cs,work),color=color,lw=2.5,label=f'b = {b:g} K')
    ax.set(xlabel='c (K)',ylabel='W (J)',title='B. '+('총 일에 보이지 않는 c 방향' if ko else 'The c direction is invisible to total work'),ylim=(1780,1890))
    ax.legend(loc='lower right');ax.grid(alpha=.2)
    fig.suptitle('총 열·일과 경로 관측의 영방향' if ko else 'Total heat/work and an observation null direction',fontsize=16)
    save(fig,'path-null-direction',lang)
