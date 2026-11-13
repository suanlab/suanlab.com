"""Native SVGs comparing accumulated cutoff error and exact bond-two correlations."""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'quantum-representation-research','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
f=next(p for p in font_manager.findSystemFonts()if'NotoSansCJK'in p);font_manager.fontManager.addfont(f);plt.rcParams['font.family']=font_manager.FontProperties(fname=f).get_name();P=Path('public/assets/learning');blue='#2563eb';orange='#c2410c';green='#047857'
for lang in ['en','ko']:
 en=lang=='en';end=2*np.pi/np.sqrt(5);times=np.linspace(0,end,501);fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained')
 for cutoff,color in zip([1,3,6],[blue,orange,green]):
  a=np.diag(np.sqrt(np.arange(1,cutoff+1)),1);h=a.T@a+a.T+a;energies,vectors=np.linalg.eigh(h);distances=[]
  for t in times:
   finite=vectors@(np.exp(-1j*energies*t)*vectors[0,:].conj());alpha=-(1-np.exp(-1j*t));mu=abs(alpha)**2;coherent=np.array([alpha**n/np.sqrt(float(math.factorial(n)))for n in range(cutoff+1)])*np.exp(-mu/2);overlap=np.vdot(coherent,finite);distances.append(np.sqrt(max(0,1-abs(overlap)**2)))
  ax[0].plot(times/end,distances,color=color,label=f'C = {cutoff}')
 ax[0].set(title='Exact coherent state versus finite evolution' if en else '정확 coherent 상태와 유한 시간 전개',ylabel='Trace distance' if en else 'Trace distance',ylim=(-.02,1.04));ax[0].legend()
 theta=np.sqrt(5)*times/2;top=4/5*np.sin(theta)**2;bound=4*np.sqrt(2)/5*(1-np.cos(theta));ax[1].plot(times/end,top,color=blue,label='C=1 top probability' if en else 'C=1 top 확률');ax[1].plot(times/end,np.minimum(1,bound),color=orange,label='Capped accumulated bound' if en else '1에서 자른 누적 상계');ax[1].set(title='Endpoint top = 0 does not mean error = 0' if en else '끝점 top = 0이 오류 = 0을 뜻하지 않는다',ylabel='Two different diagnostics' if en else '서로 다른 두 진단',ylim=(-.02,1.04));ax[1].legend(fontsize=10)
 for a in ax:a.set(xlabel='Time tau / finite return time' if en else '시간 tau / 유한 귀환 시간',xlim=(0,1));a.grid(alpha=.2)
 fig.suptitle('A boundary history matters' if en else '경계의 이력이 중요하다',fontsize=15);fig.savefig(P/f'quantum-representation-research-boundary-history-dynamics.{lang}.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
 fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained');distance=np.arange(1,13)
 for p,color in zip([.25,.75,1],[orange,blue,green]):ax[0].plot(distance,(2*p-1)**distance,'o-',color=color,label=f'p = {p}')
 ax[0].axhline(0,color='#475569',linewidth=.8);ax[0].set(title='Connected ZZ: exact lambda to the r' if en else '연결 ZZ: 정확한 lambda의 r제곱',xlabel='Separation r (sites)' if en else '거리 r (격자 간격)',ylabel='Connected correlation' if en else '연결 상관');ax[0].legend()
 eps=np.geomspace(.001,.2,501);xi=-1/np.log(1-2*eps);ax[1].loglog(eps,xi,color=blue,label='Exact correlation length' if en else '정확 상관 길이');ax[1].loglog(eps,1/(2*eps),color=orange,linestyle='--',label='Small-epsilon 1 / (2 epsilon)' if en else '작은 epsilon의 1 / (2 epsilon)');ax[1].set(title='Bond dimension remains 2' if en else 'Bond dimension은 계속 2',xlabel='epsilon = 1 - p' if en else 'epsilon = 1 - p',ylabel='Correlation length (sites)' if en else '상관 길이 (격자 간격)');ax[1].legend(fontsize=9)
 for a in ax:a.grid(alpha=.2)
 fig.suptitle('A small bond does not fix a decay length' if en else '작은 bond가 감쇠 길이를 고정하지 않는다',fontsize=15);fig.savefig(P/f'quantum-representation-research-transfer-correlation-gauge.{lang}.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
for p in P.glob('quantum-representation-research-*.svg'):p.write_text('\n'.join(s.rstrip()for s in p.read_text().splitlines())+'\n')
