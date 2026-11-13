"""Bilingual current precision, initial-work and marked-heat scientific figures."""
from pathlib import Path
from fractions import Fraction as F
from math import exp, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path('public/assets/learning')
PREVIEW = Path('.runtime/learning/entropy-thermal-research')
PREVIEW.mkdir(parents=True, exist_ok=True)
font = next(p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                     'font.size': 14, 'svg.fonttype': 'none',
                     'svg.hashsalt': 'suanlab-entropy-thermal-research',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#1e293b', 'text.color': '#1e293b',
                     'figure.facecolor': 'white'})
BLUE, ORANGE, GREEN = '#2563eb', '#c2410c', '#047857'


def save(fig, name, lang):
    path = OUT / f'entropy-thermal-research-{name}.{lang}.svg'
    fig.savefig(path, bbox_inches='tight', pad_inches=.25, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(PREVIEW / f'{name}.{lang}.png', dpi=120, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)


for lang in ['en', 'ko']:
    ko = lang == 'ko'
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    for x in [np.linspace(-5, -.01, 400), np.linspace(.01, 5, 400)]:
        ax.plot(x, x/np.tanh(x/2), color=BLUE, linewidth=3)
    ax.axhline(2, color=GREEN, linestyle=':')
    ax.plot([0], [2], marker='o', markerfacecolor='white', markeredgecolor=BLUE, markersize=8,
            label='균형으로의 극한: 2' if ko else 'Limit at balance: 2')
    ax.set(xlabel='로그 율 비 x = ln(α/β)' if ko else 'Log rate ratio x = ln(α/β)',
           ylabel='정밀도 비용 C' if ko else 'Precision cost C',
           title='균형 근처의 비영 전류 정밀도 비용' if ko else 'Precision cost of a nonzero current near balance')
    ax.legend(fontsize=12)
    cases = [(3, 3, 3, 1, 1, 1), (2, 3, 4, 1, 1, 1), (1, 2, 5, 2, 1, 1), (1, 1, 1, 3, 3, 3)]
    variances = []
    for rates in cases:
        a, b, c, d, e, f = map(F, rates)
        z = c*d+d*e+b*c+a*e+a*c+e*f+b*f+a*b+d*f
        j = 3*(a*b*c-d*e*f)/z
        variances.append(float((9*(a*b*c+d*e*f)-2*sum(rates)*j*j)/z))
    bars = bx.bar(['균일', '비균일', '혼합', '반전'] if ko else ['Uniform', 'Unequal', 'Mixed', 'Reversed'], variances, color=[BLUE, ORANGE, BLUE, GREEN])
    bx.bar_label(bars, fmt='%.3f', padding=5)
    bx.set(ylim=(0, max(variances)*1.18), ylabel='계수 분산 비율 V [1/s]' if ko else 'Count variance rate V [1/s]',
           title='방향을 반전해도 분산은 음수가 아니다' if ko else 'Reversing direction does not make variance negative')
    ax.grid(alpha=.2)
    save(fig, 'current-precision-cost', lang)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    attenuation = exp(-1)
    c0, c1 = (21-5*attenuation)/36, (7+5*attenuation)/108
    optimum = sqrt(c1)/(sqrt(c0)+sqrt(c1))
    p = np.linspace(.03, .97, 500)
    variance = c0/(1-p)+c1/p-1
    minimum = (sqrt(c0)+sqrt(c1))**2-1
    ax.semilogy(p, variance, color=BLUE, linewidth=3)
    ax.scatter([optimum], [minimum], color=GREEN, s=65, zorder=4,
               label='초기 제안 최적값' if ko else 'Optimal initial proposal')
    ax.scatter([.5], [c0/.5+c1/.5-1], facecolors='white', edgecolors=BLUE, s=65, zorder=4,
               label='정준 초기화' if ko else 'Canonical initialization')
    ax.set(ylabel='보정 가중치 분산 Var[Z]' if ko else 'Corrected-weight variance Var[Z]',
           title='표집 정밀도와 물리적 준비는 다른 질문이다' if ko else 'Sampling precision and physical preparation are distinct')
    ax.legend(fontsize=12)
    raw = (1-p)*(9-attenuation)/6+p*(3+attenuation)/6
    bx.plot(p, raw, color=ORANGE, linewidth=3, label='무보정 평균 E[Y]' if ko else 'Uncorrected mean E[Y]')
    bx.axhline(1, color=GREEN, linestyle=':', label='보정 목표 E[Z] = 1' if ko else 'Corrected target E[Z] = 1')
    bx.set(xlabel='초기 상태 1 확률' if ko else 'Initial probability of state 1',
           ylabel='정규화 지수 평균' if ko else 'Normalized exponential mean',
           title='바뀐 준비에는 초기 중요도 보정이 필요하다' if ko else 'A changed preparation needs initial importance weights')
    bx.legend(fontsize=12)
    for panel in [ax, bx]:
        panel.set_xlim(0, 1)
        panel.axvline(optimum, color=GREEN, linestyle=':', alpha=.7)
        panel.grid(alpha=.2)
    save(fig, 'initial-work-reweighting', lang)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    t = np.linspace(0, 4, 400)
    variance_rate, activity = 231/1600, 27/80
    exact = variance_rate*t+(activity-variance_rate)/2*(-np.expm1(-2*t))
    ax.plot(t, exact, color=BLUE, linewidth=3, label='정확한 유한 분산' if ko else 'Exact finite-time variance')
    ax.plot(t, variance_rate*t, color=ORANGE, linestyle='--', linewidth=3, label='점근 근사 V_A T' if ko else 'Asymptotic approximation V_A T')
    ax.set(xlabel='관측 길이 T [s]' if ko else 'Observation duration T [s]',
           ylabel='채널 A 계수 분산' if ko else 'Channel-A count variance',
           title='짧은 기록에는 이완 보정이 남는다' if ko else 'Short records retain a relaxation correction')
    ax.legend(fontsize=12)
    affinity = np.log(3)
    chi = np.linspace(-affinity/2-1.5, -affinity/2+1.5, 500)
    psi = -1+np.sqrt(1+(9/40)*np.expm1(chi)+(3/40)*np.expm1(-chi))
    bx.plot(chi, psi, color=BLUE, linewidth=3)
    bx.axhline(0, color='#64748b', linewidth=1)
    for x in [-affinity, 0]:
        bx.axvline(x, color=GREEN, linestyle=':')
    bx.set_xticks([-2, -affinity, -affinity/2, 0, 1], ['−2', '−A', '−A/2', '0', '1'])
    bx.set(xlabel='채널 계수 기울임 χ · A = ln 3' if ko else 'Channel count tilt χ; A = ln 3',
           ylabel='장시간 로그 성장률 ψ [1/s]' if ko else 'Long-time log growth ψ [1/s]',
           title='영 성장률은 유한 시간 정규화가 아니다' if ko else 'Zero growth does not normalize a finite-time moment')
    for panel in [ax, bx]:
        panel.grid(alpha=.2)
    save(fig, 'reservoir-count-fluctuations', lang)
