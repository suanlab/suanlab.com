"""Paired native SVGs for spectral cutoffs and exterior-block energy corrections."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'quantum-variational-research','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
font=next(p for p in font_manager.findSystemFonts()if'NotoSansCJK'in p);font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
P=Path('public/assets/learning');blue='#2563eb';orange='#c2410c';green='#047857'
for lang in ['en','ko']:
 en=lang=='en';q=np.linspace(0,.99,501);fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained')
 ax[0].plot(q,2*q,color=blue,label='Mean E = 2q' if en else '평균 E = 2q');ax[0].axhline(2,color=orange,linestyle='--',label='Verified cutoff beta = 2' if en else '검증한 하한 beta = 2');ax[0].set(title='The positive denominator is beta - E' if en else '양의 분모는 beta - E',ylabel='Energy' if en else '에너지');ax[0].legend()
 ax[1].plot(q,np.minimum(q/(1-q),1),color=orange,label='Capped v / (beta - E)²' if en else '1에서 자른 v / (beta - E)²');ax[1].plot(q,q,color=blue,linewidth=3,label='Actual leakage' if en else '실제 leakage');ax[1].plot(q,q,color=green,linestyle='--',label='One-sided variance bound' if en else '단측 분산 상계');ax[1].set(title='Sharp two-level bound; q = 1 is excluded' if en else '두 준위의 등호; q = 1은 제외',ylabel='Excited probability' if en else '들뜬 확률');ax[1].legend(fontsize=10)
 for a in ax:a.set(xlabel='Excited population q' if en else '들뜬 확률 q',xlim=(0,1));a.grid(alpha=.2)
 fig.suptitle('Variance needs a certified spectral sector' if en else '분산 해석에는 스펙트럼 영역의 인증이 필요',fontsize=15);fig.savefig(P/f'quantum-variational-research-excited-cutoff-certificates.{lang}.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
 fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained');x=np.linspace(0,2,501);root=np.sqrt(1+4*x*x);error=2*x*x/(1+root)
 ax[0].plot(x,error,color=blue,label='Exact energy error / Delta' if en else '정확 에너지 오류 / Delta');ax[0].plot(x,x*x,color=orange,linestyle='--',label='Looser upper bound x²' if en else '느슨한 상한 x²');ax[0].set(title='An exterior separation controls the correction' if en else '외부 영역의 분리가 보정을 좌우한다',ylabel='Normalized energy error' if en else '정규화 에너지 오류');ax[0].legend(fontsize=10)
 ax[1].plot(x,(1-1/root)/2,color=green,label='Exact exterior population' if en else '정확 외부 확률');ax[1].axhline(.5,color='#475569',linestyle=':',label='Large-coupling limit 1/2' if en else '큰 결합 극한 1/2');ax[1].set(title='Energy error and vector leakage are different' if en else '에너지 오류와 벡터 leakage는 다르다',ylabel='Exterior probability' if en else '외부 확률',ylim=(-.01,.54));ax[1].legend(fontsize=10)
 for a in ax:a.set(xlabel='Coupling ratio x = t / Delta' if en else '결합 비 x = t / Delta',xlim=(0,2));a.grid(alpha=.2)
 fig.suptitle('Retained Ritz value 0; full ground energy ≤ 0' if en else '보존 Ritz 값 0; 전체 바닥 에너지는 0 이하',fontsize=15);fig.savefig(P/f'quantum-variational-research-complement-schur-enrichment.{lang}.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
for p in P.glob('quantum-variational-research-*.svg'):p.write_text('\n'.join(s.rstrip()for s in p.read_text().splitlines())+'\n')
