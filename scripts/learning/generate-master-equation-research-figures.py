"""Native bilingual figures for the three master-equation research supplements.

Run from the repository root with Matplotlib, NumPy and Noto CJK installed.
Raster previews stay under .runtime; only SVG assets are shipped.
"""
from pathlib import Path
from subprocess import check_output
import sys
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('master_research', Path(__file__).with_name('verify-master-equation-research.py'))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
run = Path('.runtime/learning/master-equation-research')
run.mkdir(parents=True, exist_ok=True)
font = check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK KR'], text=True)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                     'font.size': 11, 'svg.fonttype': 'none', 'svg.hashsalt': 'suanlab-master-research',
                     'axes.spines.top': False, 'axes.spines.right': False})
blue, orange, green = '#2563eb', '#c2410c', '#047857'


def save(fig, name, lang):
    path = Path(f'public/assets/learning/master-equation-research-{name}.{lang}.svg')
    fig.savefig(path, bbox_inches='tight', pad_inches=.25, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run/f'{name}.{lang}.png', dpi=120, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)


for lang in ['en', 'ko']:
    ko = lang == 'ko'
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9.5), layout='constrained')
    ax = axes[0]
    ax.axvspan(1.2, 2., color='#ffedd5', label='마지막 미완료 체류' if ko else 'Final unfinished hold')
    ax.step([0, .4, 1.2, 2], [0, 1, 0, 0], where='post', color=blue, lw=2.5)
    ax.scatter([.4, 1.2], [1, 0], color=orange, s=45, zorder=5)
    ax.axvline(2., color='#64748b', ls='--')
    for x, y, text in [(.18, .16, '0.4 s'), (.76, 1.16, '0.8 s'), (1.55, .16, '0.8 s')]:
        ax.text(x, y, text, ha='center')
    ax.set(xlabel='기록 시각 (s)' if ko else 'Recorded time (s)',
           ylabel='상태 표지' if ko else 'State label', ylim=(-.25, 1.65), xlim=(0, 2.08), yticks=[0, 1],
           title='A. '+('종료 시각까지의 체류도 증거다' if ko else 'A hold censored by the horizon remains evidence'))
    ax.legend(loc='upper left')
    ax = axes[1]
    times = np.geomspace(.005, 5, 350)
    exposures = np.array([model.expected_exposures(2., 1., t) for t in times])
    ax.loglog(times, 2*exposures[:, 0], color=blue, lw=2.5, label='I(log alpha)')
    ax.loglog(times, exposures[:, 1], color=orange, lw=2.5, label='I(log beta)')
    short = times[times <= .08]
    ax.loglog(short, 2*short, color=blue, ls=':', label='2T: '+('짧은 시간' if ko else 'short time'))
    ax.loglog(short, short**2, color=orange, ls=':', label='T²: '+('짧은 시간' if ko else 'short time'))
    ax.set(xlabel='기록 길이 T (s)' if ko else 'Recording duration T (s)',
           ylabel='로그 전이율의 기대 정보' if ko else 'Expected log-rate information',
           title='B. '+('0에서 시작: 복귀 전이율에는 새 노출이 필요하다' if ko else 'Start at 0: learning the return rate needs exposure'))
    ax.grid(alpha=.2)
    ax.legend(loc='lower right')
    save(fig, 'path-rate-information', lang)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9.5), layout='constrained')
    time = np.linspace(0, 2.5, 600)
    decay, phase = np.exp(-3*time), np.pi*time/2
    pa = (1+2*decay*np.cos(phase-2*np.pi/3))/3
    pb = (1+2*decay*np.cos(-phase-2*np.pi/3))/3
    ax = axes[0]
    ax.plot(time, pa, color=blue, lw=2.5, label='A: a > b')
    ax.plot(time, pb, color=orange, lw=2.5, label='B: a < b')
    ax.axhline(1/3, color='#64748b', ls=':', label='1/3')
    ax.axvline(2., color='#64748b', ls='--')
    common = (1+np.exp(-6))/3
    ax.scatter([2], [common], color=green, s=65, zorder=6)
    ax.annotate('2 s: '+('같은 kernel' if ko else 'same kernel'), xy=(2, common), xytext=(1.35, .48),
                arrowprops={'arrowstyle': '->', 'color': green}, color=green)
    ax.set(xlabel='물리 시간 t (s)' if ko else 'Physical time t (s)',
           ylabel='K_t(0,1)', ylim=(0, .6),
           title='A. '+('상태 0에서 시작한 서로 다른 조건부 법칙' if ko else 'Different conditional laws from state 0'))
    ax.grid(alpha=.2)
    ax.legend(loc='lower right')
    ax = axes[1]
    contrast = 2/np.sqrt(3)*decay*np.sin(phase)
    ax.plot(time, contrast, color=blue, lw=2.5, label='A')
    ax.plot(time, -contrast, color=orange, lw=2.5, label='B')
    ax.axhline(0, color='#64748b', lw=1)
    ax.axvline(2, color='#64748b', ls='--')
    ax.scatter([2], [0], color=green, s=60, zorder=6)
    ax.set(xlabel='물리 시간 t (s)' if ko else 'Physical time t (s)',
           ylabel='K_t(0,1) - K_t(0,2)',
           title='B. '+('중간 관측에는 방향 차이가 남는다' if ko else 'Intermediate observations retain a direction contrast'))
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, 'sampled-generator-alias', lang)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9.5), layout='constrained')
    ax = axes[0]
    positions = np.arange(3)
    ax.bar(positions-.18, [4/7, 2/7, 1/7], .36, color=blue,
           label='물리 시간 가중치' if ko else 'Residence weights')
    ax.bar(positions+.18, [1/3]*3, .36, color=orange,
           label='점프 방문 가중치' if ko else 'Jump-visit weights')
    ax.set(xticks=positions, xticklabels=['0', '1', '2'], ylim=(0, .7),
           xlabel='상태' if ko else 'State', ylabel='가중치' if ko else 'Weight',
           title='A. '+('전이율 1,2,4: 방문과 체류는 다른 평균이다' if ko else 'Rates 1,2,4: visits and residence use different weights'))
    ax.legend()
    ax = axes[1]
    scale = np.geomspace(.25, 4, 300)
    ax.loglog(scale, 8/(49*scale), color=blue, lw=2.5)
    points = np.array([.25, .5, 1, 2])
    ax.scatter(points, 8/(49*points), color=orange, s=45, zorder=5)
    ax.text(.95, .9, 'mu = 4/7: '+('변하지 않음' if ko else 'unchanged'), ha='right', transform=ax.transAxes)
    ax.set(xlabel='공통 전이율 배율 c' if ko else 'Common dimensionless rate multiplier c',
           ylabel='시간 평균 분산 계수 (s)' if ko else 'Time-average variance coefficient (s)',
           title='B. '+('초당 회로가 많으면 시간 평균 오차가 줄어든다' if ko else 'More circuits per second reduce time-average noise'))
    ax.grid(alpha=.2)
    save(fig, 'renewal-time-uncertainty', lang)
