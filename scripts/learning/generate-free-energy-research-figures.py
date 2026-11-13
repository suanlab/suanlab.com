"""Native book figures; Matplotlib/NumPy and fontconfig/Noto CJK are required.

Run from repository root. Public SVGs use the existing book palette; raster
review previews stay in .runtime. No frontend package is added.
"""
from pathlib import Path
from subprocess import check_output
from math import log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

run=Path('.runtime/learning/free-energy-research');run.mkdir(parents=True,exist_ok=True)
font=check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,
 'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue,orange,green='#2563eb','#c2410c','#047857'
def save(fig,name,lang):
    p=Path(f'public/assets/learning/free-energy-{name}.{lang}.svg');fig.savefig(p,bbox_inches='tight',pad_inches=.2)
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    fig.savefig(run/f'{name}-{lang}.png',dpi=120,bbox_inches='tight',pad_inches=.2);plt.close(fig)

dt=.001;trace=[];x,v=1.,.8;d=.4
def rhs(x,v):return v,1/x-1-d*v
for step in range(12001):
    if step%10==0:trace.append((step*dt,x,v))
    a,b=rhs(x,v);c,f=rhs(x+dt*a/2,v+dt*b/2);g,j=rhs(x+dt*c/2,v+dt*f/2);k,l=rhs(x+dt*g,v+dt*j)
    x+=dt*(a+2*c+2*g+k)/6;v+=dt*(b+2*f+2*j+l)/6
tau,xs,vs=np.array(trace).T
for lang in ['en','ko']:
    ko=lang=='ko'
    fig,axes=plt.subplots(2,1,figsize=(8.5,9.5),layout='constrained');fig.set_facecolor('white')
    time=np.linspace(0,4,600);pa,pb=.25,.1;p=pb+(pa-pb)*np.exp(-time)
    gap=p*np.log(p/pb)+(1-p)*np.log((1-p)/(1-pb));initial=gap[0]
    axes[0].plot([-1,0],[pa,pa],color=blue,lw=2.5)
    axes[0].plot(time,p,color=blue,lw=2.5,label='p(t)')
    axes[0].plot([-1,0,0,4],[pa,pa,pb,pb],color=orange,ls='--',lw=2,label=''+('정준 목표' if ko else 'Canonical target'))
    axes[0].set(title='A. '+('점유는 연속, 목표는 점프' if ko else 'Continuous occupancy, a changing target'),xlabel='t (s)',ylabel='P');axes[0].legend()
    axes[1].plot([-1,0,0],[0,0,initial],color=blue,lw=2.5)
    axes[1].plot(time,gap,color=blue,lw=2.5,label='D ('+('현재 목표' if ko else 'current target')+')')
    axes[1].plot(time,initial-gap,color=green,lw=2.5,label='Σ / kB')
    axes[1].axhline(initial,color='#64748b',ls=':',label='Wex / (kB T)')
    axes[1].set(title='B. '+('남은 차와 누적 생성량' if ko else 'Remaining gap and accumulated production'),xlabel='t (s)',ylabel=''+('무차원' if ko else 'Dimensionless'));axes[1].legend()
    for ax in axes:ax.axvline(0,color='#64748b',ls=':',lw=1);ax.grid(alpha=.2)
    fig.suptitle('준위 변경과 열적 완화' if ko else 'A level quench and thermal relaxation',fontsize=16);save(fig,'quench-relaxation',lang)

    fig,axes=plt.subplots(2,1,figsize=(8.5,9.5),layout='constrained');fig.set_facecolor('white')
    potential=xs-1-np.log(xs);kinetic=vs*vs/2
    axes[0].plot(tau,xs,color=blue,lw=2.5,label='x = V / V*');axes[0].axhline(1,color='#64748b',ls='--',label=''+('압력 균형' if ko else 'Pressure balance'));axes[0].legend()
    axes[0].set(title='A. '+('초기 속도로 최소점을 지나치기' if ko else 'Initial velocity carries the piston past the minimum'),xlabel='τ = t / t*',ylabel='x')
    for curve,color,label in [(potential,blue,'h(x)'),(kinetic,orange,'K / A'),(potential+kinetic,green,'e = K/A + h')]:axes[1].plot(tau,curve,color=color,lw=2.5,label=label)
    axes[1].set(title='B. '+('퍼텐셜과 전체 에너지의 차이' if ko else 'Potential versus total available energy'),xlabel='τ = t / t*',ylabel=''+('무차원 에너지' if ko else 'Dimensionless energy'));axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('관성·감쇠와 자유 에너지 수지' if ko else 'Inertia, damping and the free-energy budget',fontsize=16);save(fig,'inertial-piston-budget',lang)
