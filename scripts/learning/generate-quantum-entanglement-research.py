"""Native scientific figures for product geometry, filtering and finite witness data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'quantum-entanglement-research','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
font=next(p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p);font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
P=Path('public/assets/learning');blue='#2563eb';orange='#c2410c';green='#047857'
for lang in ['en','ko']:
 en=lang=='en';q=np.linspace(0,.5,501);entropy=np.zeros_like(q);entropy[1:]=(-(1-q[1:])*np.log1p(-q[1:])-q[1:]*np.log(q[1:]))/np.log(2)
 fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained')
 ax[0].plot(q,2*q,color=blue,label='p(success) = 2q');ax[0].plot(q,1-2*q,color=orange,label='p(failure) = 1 - 2q');ax[0].set(title='Success becomes rare near a product' if en else '곱상태에 가까우면 성공이 드물다',ylabel='Branch probability' if en else '가지 확률')
 ax[1].plot(q,entropy,color=blue,label='Input h(q) / ln 2' if en else '입력 h(q) / ln 2');ax[1].plot(q,2*q,color=green,label='Flagged average = 2q' if en else '표지를 포함한 평균 = 2q');ax[1].set(title='A Bell success does not increase the average' if en else '성공 가지가 Bell이어도 평균은 증가하지 않는다',ylabel='Entanglement (bits)' if en else '얽힘 (bits)')
 for a in ax:a.set(xlabel='Smaller Schmidt weight q' if en else '작은 Schmidt 가중치 q',xlim=(0,.5),ylim=(-.03,1.05));a.legend();a.grid(alpha=.2)
 fig.suptitle('Retain the success flag and every attempt' if en else '성공 표지와 모든 입력 시도를 함께 기록',fontsize=15)
 fig.savefig(P/f'quantum-entanglement-research-product-distance-filter-flags.{lang}.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
 fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained');p=np.linspace(0,1,501);mean=(1-3*p)/4
 ax[0].plot(p,mean,color=blue,label='Witness mean (1 - 3p) / 4' if en else 'Witness 평균 (1 - 3p) / 4');ax[0].axhline(0,color='#475569',linewidth=1);ax[0].axvline(1/3,color=green,linestyle=':',label='Entanglement threshold' if en else '얽힘 경계');ax[0].axvline(1/np.sqrt(2),color=orange,linestyle='--',label='Projective CHSH threshold' if en else 'Projective CHSH 경계');ax[0].set(title='Werner source: different detection regions' if en else 'Werner 원천: 서로 다른 탐지 영역',xlabel='Werner parameter p' if en else 'Werner 파라미터 p',ylabel='Witness expectation' if en else 'Witness 기대',xlim=(0,1));ax[0].legend(fontsize=9)
 n=np.geomspace(100,10000,501);upper=-1/8+3/np.sqrt(n)
 ax[1].plot(n,upper,color=blue,label='Ideal upper bound' if en else '이상적 보정 상한');ax[1].plot(n,upper+3/40,color=orange,label='Upper bound + 3/40 calibration' if en else '상한 + 보정 여유 3/40');ax[1].axhline(-1/8,color=green,linestyle=':',label='Illustrative estimate = -1/8' if en else '설명용 점 추정 = -1/8');ax[1].axhline(0,color='#475569',linewidth=1);ax[1].set(xscale='log',xlabel='Fixed trial count n' if en else '사전 고정한 시도 수 n',ylabel='Witness upper bound' if en else 'Witness 상한',title='delta = exp(-8); illustrative count records' if en else 'delta = exp(-8); 설명용 계수 기록');ax[1].legend(fontsize=9)
 for a in ax:a.grid(alpha=.2)
 fig.suptitle('A negative estimate still needs a valid bound' if en else '음수 추정에도 유효한 신뢰 상한이 필요',fontsize=15)
 fig.savefig(P/f'quantum-entanglement-research-witness-finite-evidence.{lang}.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
for path in P.glob('quantum-entanglement-research-*.svg'):path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
