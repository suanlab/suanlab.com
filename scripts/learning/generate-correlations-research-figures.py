"""Native bilingual figures for nuisance information, spatial modes and GLS."""
from pathlib import Path
import subprocess,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from importlib.util import spec_from_file_location,module_from_spec
sys.dont_write_bytecode=True
spec=spec_from_file_location('verify',Path(__file__).with_name('verify-correlations-research.py'));v=module_from_spec(spec);spec.loader.exec_module(v)
font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True);font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue,orange,green='#2563eb','#c2410c','#047857'
out=Path('public/assets/learning');preview=Path('.runtime/learning/correlations-research');preview.mkdir(parents=True,exist_ok=True)
def save(fig,name,lang):
    fig.tight_layout(pad=2);p=out/f'correlations-research-{name}.{lang}.svg';fig.savefig(p,bbox_inches='tight',pad_inches=.2);p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');fig.savefig(preview/f'{name}.{lang}.png',dpi=110,bbox_inches='tight',pad_inches=.2);plt.close(fig)
def information(k,h1,h2):
    features=np.array([[a,b,a*b] for a,b in v.STATES]);p=np.exp(features@np.array([h1,h2,k]));p/=p.sum();centered=features-p@features;full=centered.T@np.diag(p)@centered;c=full[:2,:2];eff=c-np.outer(full[:2,2],full[:2,2])/full[2,2];return c,eff
for lang in ('en','ko'):
    ko=lang=='ko';fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));hs=np.linspace(-2,2,150);known=[];unknown=[]
    for h in hs:
        c,e=information(.7,h,h);known.append(np.linalg.inv(c)[0,0]);unknown.append(np.linalg.inv(e)[0,0])
    ax[0].plot(hs,np.array(unknown)/known,color=blue);ax[0].axhline(1,color=orange,ls=':');ax[0].set(xlabel='같은 국소 편향 h1=h2' if ko else 'Equal local biases h1=h2',ylabel='h1 점근 분산 계수 비' if ko else 'h1 asymptotic variance coefficient ratio',title='참 K=0.7: 미지 결합 / 알려진 결합' if ko else 'True K=0.7: unknown / known coupling')
    c,e=information(.7,.8,-.3);theta=np.linspace(0,2*np.pi,300);circle=np.array([np.cos(theta),np.sin(theta)])
    for matrix,col,label in [(c,blue,'알려진 K' if ko else 'Known K'),(e,orange,'미지 K' if ko else 'Unknown K')]:
        eig,u=np.linalg.eigh(np.linalg.inv(matrix));curve=u@np.diag(np.sqrt(eig))@circle;ax[1].plot(*curve,color=col,label=label)
    ax[1].set(xlabel='척도화한 h1 오차' if ko else 'Scaled h1 error',ylabel='척도화한 h2 오차' if ko else 'Scaled h2 error',title='극한 정규 법칙의 공분산 등고선' if ko else 'Covariance contours of the limiting normal law',xlim=(-2.5,2.5),ylim=(-2,2.8));ax[1].set_aspect('equal');ax[1].legend(loc='upper center');save(fig,'interaction-nuisance-information',lang)

    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5))
    for n,col,marker in [(15,orange,'o'),(16,blue,'s')]:
        ls=np.arange(n//2+1);values=v.spectrum(n,-1.2)[:n//2+1];ax[0].scatter(2*ls/n,values,color=col,marker=marker,label=f'N={n}');ax[0].vlines(2*ls/n,0,values,color=col,lw=1,alpha=.6)
    ax[0].set(xlabel='허용 파수 q / pi (반사 절반 생략)' if ko else 'Allowed q / pi (reflected half omitted)',ylabel='정적 구조인자 S' if ko else 'Static structure factor S',title='K=-1.2: 홀수 고리는 정확한 pi 모드가 없음' if ko else 'K=-1.2: no exact pi mode on the odd ring',ylim=(0,12));ax[0].legend(loc='upper left')
    for parity,col in [(1,orange),(0,blue)]:
        ns=np.arange(3 if parity else 4,65,2);peaks=[max(v.spectrum(int(n),-1.2))/n for n in ns];cold=1/(ns**2*np.sin(np.pi/(2*ns))**2) if parity else np.ones(len(ns));label='홀수' if ko and parity else '짝수' if ko else 'Odd' if parity else 'Even';ax[1].plot(ns,peaks,'o-',ms=3,color=col,label=label+(' 유한 K' if ko else ', finite K'));ax[1].plot(ns,cold,color=col,ls='--',label=label+(' 저온 극한' if ko else ', cold limit'))
    ax[1].set(xlabel='고리 크기 N' if ko else 'Ring size N',ylabel='봉우리 S / N' if ko else 'Peak S / N',title='고정 유한 결합과 먼저 취한 저온 극한' if ko else 'Fixed finite coupling versus taking the cold limit first',ylim=(0,1.3));ax[1].legend(loc='upper right',fontsize=10);save(fig,'ring-fourier-modes',lang)

    fig,ax=plt.subplots(2,1,figsize=(8.5,9.5));n=9;x=np.arange(1,n+1)
    for rho,col in [(.8,blue),(-.8,orange)]:ax[0].plot(x,v.gls(n,rho),'o-',color=col,label=f'rho={rho}')
    ax[0].axhline(1/n,color=green,ls='--',label='균일 가중치' if ko else 'Uniform weights');ax[0].set(xlabel='관측 인덱스 i' if ko else 'Observation index i',ylabel='GLS 가중치' if ko else 'GLS weight',title='n=9: 정상 과정, 유한 관측 창의 경계' if ko else 'n=9: stationary process, finite observation boundaries',ylim=(0,.45));ax[0].legend(loc='upper center',fontsize=10)
    ns=np.arange(2,41)
    for rho,col in [(.8,blue),(-.8,orange)]:
        ratios=[]
        for n in ns:
            variance=(n+2*sum((n-k)*rho**k for k in range(1,n)))/n**2;efficient=(1+rho)/(n*(1-rho)+2*rho);ratios.append(variance/efficient)
        ax[1].plot(ns,ratios,'o-',ms=3,color=col,label=f'rho={rho}')
    ax[1].axhline(1,color=green,ls=':');ax[1].set(xlabel='관측 수 n' if ko else 'Observation count n',ylabel='보통 평균 분산 / GLS 분산' if ko else 'Ordinary mean variance / GLS variance',title='같은 관측·다른 추정량: 유한 효율 차이' if ko else 'Same observations, different estimators: finite efficiency',ylim=(.9,2.7));ax[1].legend(loc='upper right');save(fig,'ar1-efficient-mean',lang)
