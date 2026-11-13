"""Render the bilingual thermometer figure (Matplotlib, NumPy, fontconfig/Noto CJK).

Run from the repository root. SVG files are public book assets; raster previews
remain under .runtime and are not shipped. No frontend dependency is introduced.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import exp
import numpy as np
from matplotlib import font_manager
from subprocess import check_output

font_path=check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True)
font_manager.fontManager.addfont(font_path)
Path('.runtime/learning/thermometry-research').mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),'font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
for lang in ['en','ko']:
 ko=lang=='ko'
 fig,axes=plt.subplots(2,1,figsize=(8.5,9.5),layout='constrained')
 fig.set_facecolor('#ffffff')
 ax=axes[0];xs=np.linspace(.01,10,700);p=1/(1+np.exp(xs));cs=xs*xs*p*(1-p)
 ax.plot(xs,cs,color='#2563eb',lw=2.5);star=2.3993572805154677;peak=star*star*exp(-star)/(1+exp(-star))**2
 ax.scatter([star],[peak],color='#c2410c',s=60,zorder=5)
 ax.annotate('x = 2.399357\nC = 0.439229',xy=(star,peak),xytext=(4.5,.37),arrowprops={'arrowstyle':'->','color':'#c2410c'},color='#9a3412')
 ax.set(xlabel='x = Δ/T  (kB = 1)',ylabel='C = I(θ)  ('+('무차원' if ko else 'dimensionless')+')',title='A. '+('독립 준비 한 번의 상대 온도 정보' if ko else 'Fractional-temperature information per independent preparation'),xlim=(0,10),ylim=(0,.51));ax.grid(alpha=.2)
 ax=axes[1];ax.plot([.3,3.2],[.3,3.2],color='#047857',lw=2.5)
 ax.scatter([.5,1,2],[.5,1,2],color='#2563eb',s=60,zorder=5)
 for v in [.5,1,2]:ax.annotate(f'({v:g}, {v:g})',xy=(v,v),xytext=(10,-16),textcoords='offset points')
 ax.text(.04,.93,'x = 1; p = 0.268941; g = 1',transform=ax.transAxes,va='top')
 ax.text(.04,.83,'N → ∞: '+('척도 모호성 유지' if ko else 'scale ambiguity remains'),transform=ax.transAxes,va='top',color='#047857')
 ax.set(xlabel='T  ('+('선택한 에너지 단위' if ko else 'chosen energy unit')+')',ylabel='Δ  ('+('같은 에너지 단위' if ko else 'same energy unit')+')',title='B. '+('동일한 이진 관측 분포를 주는 서로 다른 물리적 쌍' if ko else 'Different physical pairs, identical binary observation law'),xlim=(0,3.4),ylim=(0,3.4));ax.grid(alpha=.2)
 fig.suptitle('열적 민감도와 절대 척도의 식별' if ko else 'Thermal sensitivity and identification of an absolute scale',fontsize=16)
 path=Path(f'public/assets/learning/thermometry-information-scale.{lang}.svg');fig.savefig(path)
 fig.savefig(f'.runtime/learning/thermometry-research/figure-{lang}.png',dpi=120);plt.close(fig)
