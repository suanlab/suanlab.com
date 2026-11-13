"""Bilingual predictive belief and matched initial-layer scientific figures."""
from pathlib import Path
from math import exp,sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'public/assets/learning';PREVIEW=ROOT/'.runtime/learning/coarse-predictive-research';PREVIEW.mkdir(parents=True,exist_ok=True)
font=next(p for p in font_manager.findSystemFonts()if'NotoSansCJK-Regular'in p);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','svg.hashsalt':'suanlab-coarse-predictive-research','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'axes.labelcolor':'#1e293b','text.color':'#1e293b','figure.facecolor':'white'})
BLUE,ORANGE,GREEN='#2563eb','#c2410c','#047857'
def save(fig,name,lang):
    target=OUT/f'coarse-predictive-research-{name}.{lang}.svg';fig.savefig(target,bbox_inches='tight',pad_inches=.25,metadata={'Date':None});target.write_text('\n'.join(line.rstrip()for line in target.read_text().splitlines())+'\n');fig.savefig(PREVIEW/f'{name}.{lang}.png',dpi=130,bbox_inches='tight',pad_inches=.25);plt.close(fig)
def curves(eps,t):
    a,b,c,d,x0,y0=2.,1.,2.,3.,1.,0.;mu=-a+b*c/d;w0=y0-c*x0/d;disc=sqrt((d-eps*a)**2+4*eps*b*c);slow=-2*(a*d-b*c)/(d+eps*a+disc);fast=-(d+eps*a)/eps-slow;aslow=(-a*x0+b*y0-fast*x0)/(slow-fast);afast=x0-aslow;rate=mu/(1+eps*b*c/d**2);slip=eps*b*w0/d
    return aslow*exp(slow*t)+afast*exp(fast*t),x0*exp(mu*t),(x0+slip)*exp(rate*t)-slip*exp(-d*t/eps)
def errors(eps):
    times=sorted(set([4*i/4000 for i in range(4001)]+[eps*u/30 for u in range(401)if eps*u/30<=4]));rows=[curves(eps,t)for t in times];return max(abs(x-m)for x,m,c in rows),max(abs(x-c)for x,m,c in rows)
for lang in ['ko','en']:
    ko=lang=='ko';fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True);histories=['AA','BA','AAA','BAA'];x=np.arange(4);q=[.75,.25,.7,5/6];prob=[3/8,5/8,2/5,1/3]
    for axis,values in [(a,q),(b,prob)]:
        axis.bar(x,values,color=BLUE if axis is a else ORANGE,width=.55);axis.set_xticks(x,histories);axis.set_ylim(0,1);axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
        for i,v in enumerate(values):axis.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=10)
    a.set(ylabel='조건부 상태 0 비율'if ko else 'Conditional fraction in micro state 0',title='현재 A가 같아도 내부 구성은 이력에 따라 다르다'if ko else 'Same current A, different inferred composition')
    b.axhline(.5,color=GREEN,ls='--',label='고정 lift: 1/2'if ko else 'Fixed lift: 1/2');b.set(xlabel='관측 이력'if ko else 'Observed history',ylabel='다음 B 확률'if ko else 'Next B probability',title='AAA와 BAA는 마지막 두 표지도 같다'if ko else 'AAA and BAA even share their last two labels');b.legend(fontsize=10);save(fig,'conditional-state',lang)
    fig,(a,b)=plt.subplots(2,1,figsize=(8,8),constrained_layout=True);time=np.linspace(0,.8,500);values=np.array([curves(.25,float(t))for t in time])
    for j,color,label in [(0,BLUE,'전체 해'if ko else 'Full flow'),(1,ORANGE,'원래 Markov 비교'if ko else 'Original Markov'),(2,GREEN,'보정 합성식'if ko else 'Corrected composite')]:a.plot(time,values[:,j],color=color,lw=2,ls='-'if j==0 else'--'if j==1 else':',label=label)
    a.set(xlabel='물리 시간 [s]'if ko else 'Physical time [s]',ylabel='관측 x(t)'if ko else 'Observed x(t)',title='빠른 초기층이 남기는 느린 변위, ε=1/4'if ko else 'The fast initial layer leaves a slow displacement, ε=1/4');a.legend(fontsize=10);a.grid(alpha=.15)
    eps=[.25,.125,.0625,.03125];err=np.array([errors(e)for e in eps]);b.loglog(eps,err[:,0],'-o',color=ORANGE,label='Markov');b.loglog(eps,err[:,1],'-o',color=GREEN,label='합성식'if ko else 'Composite');b.set(xlabel='시간 비율 ε'if ko else 'Time-scale ratio ε',ylabel='격자 위 최대 절대 오차'if ko else 'Maximum absolute error on sampled grid',title='고정 준비·4초 창: 표집 오차 비교'if ko else 'Fixed preparation, 4 s window: sampled error comparison');b.legend(fontsize=10);b.grid(which='both',alpha=.15);save(fig,'initial-layer',lang)
