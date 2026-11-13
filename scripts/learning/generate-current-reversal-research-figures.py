"""Bilingual scientific figures for time-arrow and finite-horizon path statistics."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path('public/assets/learning')
PREVIEW = Path('.runtime/learning/current-reversal-research')
PREVIEW.mkdir(parents=True, exist_ok=True)
font = next(p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                     'font.size': 14, 'svg.fonttype': 'none',
                     'svg.hashsalt': 'suanlab-current-reversal-research',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#1e293b', 'text.color': '#1e293b',
                     'figure.facecolor': 'white'})
BLUE, ORANGE, GREEN = '#2563eb', '#c2410c', '#047857'


def save(fig, name, lang):
    path = OUT / f'current-reversal-research-{name}.{lang}.svg'
    fig.savefig(path, bbox_inches='tight', pad_inches=.25, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(PREVIEW / f'{name}.{lang}.png', dpi=120, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)


for lang in ['en', 'ko']:
    ko = lang == 'ko'
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    delta = np.linspace(.001, 7, 800)
    rate = 4*np.sin(delta)**2/(delta*np.expm1(2*delta))
    area = 2*np.exp(-delta)*np.sin(delta)
    ax.axhline(2, color=GREEN, linestyle=':', label='연속 기록 비율 = 2/s' if ko else 'Continuous-record rate = 2/s')
    ax.plot(delta, rate, color=BLUE, linewidth=3, label='표본 정보 / 시간' if ko else 'Sampled information / time')
    ax.set(xlim=(0, 7), ylim=(-.06, 2.15), ylabel='상대 엔트로피 비율 [1/s]' if ko else 'Relative-entropy rate [1/s]',
           title='관측 간격이 시간 방향 정보를 숨긴다' if ko else 'Observation spacing can hide time-direction information')
    ax.legend(fontsize=12)
    bx.axhline(0, color='#64748b', linewidth=1)
    bx.plot(delta, area, color=ORANGE, linewidth=3, label='ω = +1 rad/s')
    bx.set(xlim=(0, 7), xlabel='관측 간격 Δ [s]' if ko else 'Observation interval Δ [s]',
           ylabel='쌍의 부호 넓이 평균 [m²]' if ko else 'Mean signed pair area [m²]',
           title='양의 각속도에서도 넓이 부호가 바뀐다' if ko else 'Pair-area signs can reverse at positive angular rate')
    for panel in [ax, bx]:
        panel.set_xticks([0, 1, 2, np.pi, 4, 5, 2*np.pi, 7], ['0', '1', '2', 'π', '4', '5', '2π', '7'])
        for x in [np.pi, 2*np.pi]:
            panel.axvline(x, color='#64748b', linestyle='--', alpha=.55)
        panel.grid(alpha=.2)
    bx.legend(fontsize=12)
    save(fig, 'sampled-time-arrow', lang)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    # Positive count weights, not renormalized. Tail beyond 80 is negligible for this plot.
    from math import exp
    plus, minus = [exp(-3)], [exp(-1)]
    for n in range(1, 81):
        plus.append(plus[-1]*3/n)
        minus.append(minus[-1]/n)
    counts = list(range(-8, 13))
    probabilities = [sum(p*m for i, p in enumerate(plus) for j, m in enumerate(minus) if i-j == k) for k in counts]
    ax.bar(counts, probabilities, color=[ORANGE if k < 0 else GREEN if k == 0 else BLUE for k in counts])
    ax.set(xlabel='방향 점프 수 N+ − N−' if ko else 'Oriented jump count N+ − N−',
           ylabel='확률' if ko else 'Probability',
           title='유한 시간에는 반대 방향 경로도 나타난다' if ko else 'Opposite-direction paths occur at finite time')
    ax.text(.98, .94, 'a = 3/s, b = 1/s, T = 1 s', transform=ax.transAxes, ha='right', va='top', fontsize=12)
    duration = np.linspace(.01, 2, 300)
    relative_se = np.sqrt(np.expm1(16*duration/3)/10000)
    bx.semilogy(duration, relative_se, color=ORANGE, linewidth=3)
    bx.set(xlabel='독립 경로당 관측 길이 T [s]' if ko else 'Duration per independent path T [s]',
           ylabel='지수 평균의 상대 표준오차' if ko else 'Relative SE of exponential mean',
           title='경로 10,000개: 길어질수록 커지는 오차' if ko else '10,000 independent paths: longer can be less precise')
    bx.grid(alpha=.2, which='both')
    save(fig, 'finite-time-path-fluctuations', lang)
