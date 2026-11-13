"""Generate bilingual scientific figures with text preserved in SVG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path('public/assets/learning')
PREVIEW = Path('.runtime/learning/fokker-research')
PREVIEW.mkdir(parents=True, exist_ok=True)
font = next(p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                     'font.size': 14, 'svg.fonttype': 'none',
                     'svg.hashsalt': 'suanlab-fokker-research', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': '#1e293b',
                     'text.color': '#1e293b', 'figure.facecolor': 'white'})
BLUE, ORANGE, GREEN = '#2563eb', '#c2410c', '#047857'


def save(fig, name, lang):
    svg_path = OUT / f'fokker-research-{name}.{lang}.svg'
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=.25, metadata={'Date': None})
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    fig.savefig(PREVIEW / f'{name}.{lang}.png', dpi=120, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)


for lang in ['en', 'ko']:
    ko = lang == 'ko'
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    z = np.linspace(.001, 3, 400)
    ax.axhline(1, color=GREEN, linestyle=':', label='정확한 계수 비 = 1' if ko else 'Correct coefficient ratio = 1')
    ax.plot(z, -np.expm1(-z)/z, color=BLUE, linewidth=3, label='완화율 γ_E / γ' if ko else 'Relaxation γ_E / γ')
    ax.plot(z, -np.expm1(-2*z)/(2*z), color=ORANGE, linewidth=3, linestyle='--', label='확산 D_E / D' if ko else 'Diffusion D_E / D')
    ax.set(xlim=(0, 3), ylim=(0, 1.12), xlabel='관측 간격 z = γΔ' if ko else 'Observation spacing z = γΔ',
           ylabel='추론한 계수 / 실제 계수' if ko else 'Inferred / true coefficient',
           title='정확한 OU 전이를 Euler로 해석한 편향' if ko else 'Interpreting exact OU transitions as Euler steps')
    ax.grid(alpha=.2)
    ax.legend(loc='upper right', fontsize=12)
    save(fig, 'sampled-ou-inference', lang)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    labels = ['왼쪽 조건부', '무조건부', '오른쪽 조건부'] if ko else ['Given left exit', 'Unconditional', 'Given right exit']
    vals = [5/12, 1/2, 3/4]
    bars = ax.barh(labels, vals, color=[BLUE, GREEN, ORANGE], height=.55)
    ax.bar_label(bars, labels=['5/12 s', '1/2 s', '3/4 s'], padding=7)
    ax.set(xlim=(0, 1), xlabel='평균 탈출 시간 [s]' if ko else 'Mean exit time [s]',
           title='왼쪽 셀 출발: 방향에 따라 달라지는 시간' if ko else 'Starting in the left cell: exit direction matters')
    cells = np.array([2, 4, 8, 16])
    widths = 2/cells
    bx.loglog(widths, widths**2/8, 'o-', color=BLUE, linewidth=3,
              label='격자 − 연속체 = Δx²/(8D)' if ko else 'Grid − continuum = Δx²/(8D)')
    bx.set(xlabel='셀 폭 Δx [m]' if ko else 'Cell width Δx [m]',
           ylabel='중심 출발 시간 오차 [s]' if ko else 'Center-start time error [s]',
           title='순수 확산: 시간 오차와 다른 공간 오차' if ko else 'Pure diffusion: spatial error after exact time integration')
    bx.grid(alpha=.2, which='both')
    bx.legend(fontsize=12)
    save(fig, 'grid-first-passage', lang)
