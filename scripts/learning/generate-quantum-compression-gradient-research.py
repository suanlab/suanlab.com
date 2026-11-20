"""Render analytic finite compression and gradient benchmarks in the existing book palette."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(font).exists():font_manager.fontManager.addfont(font)
plt.rcParams.update({'svg.fonttype':'none','font.family':'Noto Sans CJK JP' if Path(font).exists()else'DejaVu Sans','font.size':11})
root=Path('public/assets/learning');blue,orange,green='#2563eb','#c2410c','#047857'
for lang in ['ko','en']:
 def t(en,ko):return ko if lang=='ko'else en
 fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained');fig.suptitle(t('One cut is not an entire compression','한 cut은 전체 압축이 아니다'))
 x=np.arange(2);ax[0].bar(x-.17,[2/15,4/15],.34,label=t('Current discarded squared norm','현재 버린 노름 제곱'),color=blue);ax[0].bar(x+.17,[2/15,5/15],.34,label=t('Original-cut tail','원래 cut 꼬리'),color=orange)
 ax[0].set(xticks=x,xticklabels=['Cut 1','Cut 2'],ylabel=t('Squared norm / probability','노름 제곱 / 확률'),title='a,b,c,d = 3,2,1,1');ax[0].legend()
 c=np.linspace(0,2,200);N=14+c*c;E=(5+c*c)/N;bound=(5+2*c*c)/N
 ax[1].plot(c,E,color=blue,label=t('Actual sweep error E','실제 sweep 오차 E'));ax[1].plot(c,bound,color=orange,label=t('Original-tail sum bound','원래 꼬리 합 상계'));ax[1].set(xlabel='c  (a=3,b=2,d=1)',ylabel=t('Squared error / bound','오차 제곱 / 상계'),title=t('Current losses add exactly; original tails bound them','현재 손실은 정확히 더하고 원래 꼬리는 상계를 준다'));ax[1].legend()
 for a in ax:a.grid(alpha=.15);a.set_axisbelow(True);a.spines[['top','right']].set_visible(False)
 fig.savefig(root/f'quantum-compression-gradient-research-sequential-discarded-weight.{lang}.svg',bbox_inches='tight',pad_inches=.25);plt.close(fig)
 fig,ax=plt.subplots(2,1,figsize=(8,8),layout='constrained');fig.suptitle(t('A derivative rule follows the generator spectrum','미분 규칙은 생성자 spectrum을 따른다'))
 th=np.linspace(0,2*np.pi,200);f=lambda v:np.cos(v/2)
 four=(f(th+np.pi/2)-f(th-np.pi/2))/2+(1-np.sqrt(2))/4*(f(th+np.pi)-f(th-np.pi))
 ax[0].plot(th/np.pi,-.5*np.sin(th/2),color=blue,label=t('Analytic controlled derivative','제어 회전 해석적 미분'));ax[0].plot(th/np.pi,-np.sin(th/2)/np.sqrt(2),color=orange,label=t('Wrong ordinary two-shift rule','잘못 적용한 보통 두 이동'));ax[0].plot(th[::12]/np.pi,four[::12],'o',ms=4,color=green,label=t('Correct four-evaluation rule','올바른 네 번 평가'))
 ax[0].set(xlabel='theta / pi',ylabel='d <X control> / d theta');ax[0].legend()
 M=np.arange(20,1001,2)
 for theta,color in [(0,blue),(np.pi/2,orange)]:
  g=-3/5*np.sin(theta);ax[1].plot(M,(1-g*g)/M,color=color,label='theta = '+('0'if theta==0 else'pi/2'))
 ax[1].set(xscale='log',yscale='log',xlabel=t('Total independent shots M, equal allocation','독립 shot 총수 M, 균등 배분'),ylabel=t('Gradient estimator variance','Gradient 추정기 분산'),title='cos(phi) = 3/5');ax[1].legend()
 for a in ax:a.grid(alpha=.15);a.spines[['top','right']].set_visible(False)
 fig.savefig(root/f'quantum-compression-gradient-research-parameter-shift-frequency.{lang}.svg',bbox_inches='tight',pad_inches=.25);plt.close(fig)

# Keep generated SVG paths free of trailing whitespace for repository checks.
for path in root.glob("quantum-compression-gradient-research-*.svg"):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
