"""Native SVG comparisons for finite spin response and Majorana reconstruction limits."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'quantum-many-body-research','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
f=next(p for p in font_manager.findSystemFonts()if'NotoSansCJK'in p);font_manager.fontManager.addfont(f);plt.rcParams['font.family']=font_manager.FontProperties(fname=f).get_name();P=Path('public/assets/learning');blue='#2563eb';orange='#c2410c';green='#047857'
for lang in ['en','ko']:
 en=lang=='en';x=np.geomspace(.02,5,501);root=np.sqrt(1+4*x*x);gap=4*x*x/(root+1);chi=4*(root+1)/(root*gap);fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained')
 ax[0].loglog(x,gap,color=blue,label='Exact ordered gap / J' if en else '정확 순서 gap / J');ax[0].loglog(x,2*x*x,color=orange,linestyle='--',label='Small-field 2(h/J)²' if en else '작은 장의 2(h/J)²');ax[0].set(title='Two spins, h > 0: a small finite-system splitting' if en else '두 spin, h > 0: 작은 유한계 splitting',ylabel='Ordered gap / J' if en else '순서 gap / J');ax[0].legend(fontsize=10)
 ax[1].loglog(x,chi,color=green,label='Exact total J chi' if en else '정확 전체 J chi');ax[1].loglog(x,4/(x*x),color=orange,linestyle='--',label='Small-field 4 / (h/J)²' if en else '작은 장의 4 / (h/J)²');ax[1].set(title='A large derivative is not an infinite-system phase' if en else '큰 미분값이 무한계의 상을 뜻하지는 않는다',ylabel='Total susceptibility J chi' if en else '전체 susceptibility J chi');ax[1].legend(fontsize=10)
 for a in ax:a.set(xlabel='Transverse field h / J' if en else '횡장 h / J');a.grid(alpha=.2)
 fig.suptitle('Finite gap and static response' if en else '유한 gap과 정적 응답',fontsize=15);fig.savefig(P/f'quantum-many-body-research-finite-gap-response.{lang}.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
 fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained');locations=np.arange(2);labels=['n0 n1','Var(N)'];ax[0].bar(locations-.18,[.5,4],width=.36,color=blue,label='Pure four-mode cat' if en else '네 모드 순수 cat');ax[0].bar(locations+.18,[.25,1],width=.36,color=orange,label='Maximally mixed' if en else '최대 혼합');ax[0].set_xticks(locations,labels);ax[0].set(title='Same covariance Gamma = 0, different higher moments' if en else '같은 공분산 Gamma = 0, 다른 고차 모멘트',ylabel='Observable value' if en else '관측량 값');ax[0].legend(fontsize=10)
 ratio=np.linspace(0,1,201)
 for (s,r),color in zip([(-1,-1),(-1,1),(1,-1),(1,1)],[blue,orange,green,'#475569']):ax[1].plot(ratio,(s+r*ratio)/2,color=color,label=f'({s:+d} t {r:+d} u) / 2')
 ax[1].set(title='An even outer coupling splits the degeneracy' if en else 'Even 바깥 결합이 축퇴를 분리한다',xlabel='Outer coupling u / t' if en else '바깥 결합 u / t',ylabel='Energy / t' if en else '에너지 / t',xlim=(0,1));ax[1].legend(fontsize=9,ncol=2)
 for a in ax:a.grid(alpha=.2)
 fig.suptitle('Gaussian assumptions and finite Majorana zero modes' if en else 'Gaussian 가정과 유한 Majorana zero mode',fontsize=15);fig.savefig(P/f'quantum-many-body-research-majorana-covariance-limits.{lang}.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
for p in P.glob('quantum-many-body-research-*.svg'):p.write_text('\n'.join(s.rstrip()for s in p.read_text().splitlines())+'\n')
