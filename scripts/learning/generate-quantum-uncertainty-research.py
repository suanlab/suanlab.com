"""Generate paired native scientific SVGs for the two research supplements."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'quantum-uncertainty-research','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
fonts=font_manager.findSystemFonts();cjk=next(p for p in fonts if 'NotoSansCJK' in p)
font_manager.fontManager.addfont(cjk);plt.rcParams['font.family']=font_manager.FontProperties(fname=cjk).get_name()
P=Path('public/assets/learning');blue='#2563eb';orange='#c2410c';green='#047857'
for lang in ['en','ko']:
    en=lang=='en';s=np.linspace(0,1,401)
    fig,axes=plt.subplots(2,1,figsize=(8,8),layout='constrained')
    axes[0].plot(s,1+s,color=blue,label='1 + s');axes[0].plot(s,1-s,color=orange,label='1 - s');axes[0].plot(s,1-s*s,color=green,label='1 - s²')
    axes[0].set_title('Complex Gram eigenvalues: rank 1 at purity' if en else '복소 Gram 고유값: 순수 상태에서 rank 1')
    axes[1].plot(s,np.ones_like(s),color=blue,label='1 (twice)' if en else '1 (중복 2)');axes[1].plot(s,1-s*s,color=green,label='1 - s²')
    axes[1].set_title('Real covariance eigenvalues: rank 2 at purity' if en else '실수 공분산 고유값: 순수 상태에서 rank 2')
    for ax in axes:
        ax.set(xlabel='Bloch radius s' if en else 'Bloch 반지름 s',ylabel='Eigenvalue' if en else '고유값',xlim=(0,1),ylim=(-.06,2.12));ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Three Pauli fluctuations' if en else '세 Pauli 관측량의 요동',fontsize=16)
    path=P/f'quantum-uncertainty-research-pauli-gram-purity.{lang}.svg';fig.savefig(path,metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8,8),layout='constrained');k=np.linspace(0,1,501);h=np.sqrt(1-k*k)
    axes[0].fill_between(k,0,h,color=blue,alpha=.12);axes[0].plot(k,h,color=blue,label='λ = sqrt(1 - κ²)');axes[0].scatter([.6,.9],[.8,.9],c=[green,orange],zorder=3)
    axes[0].annotate('(0.6, 0.8)',(.6,.8),xytext=(.24,.91));axes[0].annotate('(0.9, 0.9): incompatible' if en else '(0.9, 0.9): 양립 불가',(.9,.9),xytext=(.44,1.04),fontsize=10)
    axes[0].set(xlabel='Z strength κ' if en else 'Z 측정 강도 κ',ylabel='X strength λ' if en else 'X 측정 강도 λ',xlim=(0,1),ylim=(0,1.13),title='Common POVM: κ² + λ² ≤ 1' if en else '공동 POVM 조건: κ² + λ² ≤ 1')
    k=np.linspace(.03,.97,500);axes[1].plot(k,1/k**2-1,color=blue,label='εZ²');axes[1].plot(k,1/(1-k*k)-1,color=orange,label='εX²');axes[1].set_yscale('log');axes[1].set(xlabel='Z strength κ (boundary)' if en else '경계 위 Z 측정 강도 κ',ylabel='Squared calibrated error' if en else '보정된 오차의 제곱',title='Boundary errors: product = 1; endpoints excluded' if en else '경계 오차의 곱 = 1; 끝점 보정은 제외')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Two noisy readouts' if en else '두 noisy 측정 판독',fontsize=16)
    path=P/f'quantum-uncertainty-research-joint-unsharp-readouts.{lang}.svg';fig.savefig(path,metadata={'Date':None},bbox_inches='tight',pad_inches=.25);plt.close(fig)
for path in P.glob('quantum-uncertainty-research-*.svg'):path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
