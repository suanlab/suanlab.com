"""Generate bilingual response spectra and control-information scientific figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path('public/assets/learning')
PREVIEW = Path('.runtime/learning/response-research')
PREVIEW.mkdir(parents=True, exist_ok=True)
font = next(p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p)
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                     'font.size': 14, 'svg.fonttype': 'none',
                     'svg.hashsalt': 'suanlab-response-research',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#1e293b', 'text.color': '#1e293b',
                     'figure.facecolor': 'white'})
BLUE, ORANGE, GREEN = '#2563eb', '#c2410c', '#047857'


def save(fig, name, lang):
    path = OUT / f'response-research-{name}.{lang}.svg'
    fig.savefig(path, bbox_inches='tight', pad_inches=.25, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(PREVIEW / f'{name}.{lang}.png', dpi=120, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)


for lang in ['en', 'ko']:
    ko = lang == 'ko'
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    frequency = np.linspace(.001, 4, 500)
    z = 1+1j*frequency
    chi = z/(z*z+4)
    spectrum = 1/(1+(frequency-2)**2)+1/(1+(frequency+2)**2)
    naive = -2*chi.imag/frequency
    power = -frequency*chi.imag/2
    ax.plot(frequency, spectrum, color=BLUE, linewidth=3, label='실제 잡음 S00' if ko else 'Actual noise S00')
    ax.plot(frequency, naive, color=ORANGE, linewidth=3, linestyle='--', label='평형 대입' if ko else 'Equilibrium shortcut')
    ax.axhline(0, color='#64748b', linewidth=1)
    ax.set(xlim=(0, 4), ylabel='양측 스펙트럼 [m² s]' if ko else 'Two-sided spectrum [m² s]',
           title='회전 구동은 평형 주파수 관계를 바꾼다' if ko else 'Rotational driving changes the equilibrium spectral relation')
    ax.legend(fontsize=12)
    bx.plot(frequency, power, color=ORANGE, linewidth=3)
    bx.axhline(0, color='#64748b', linewidth=1)
    bx.axvline(np.sqrt(3), color=GREEN, linestyle=':', label='Ω = √3 /s')
    bx.set(xlim=(0, 4), xlabel='각주파수 Ω [rad/s]' if ko else 'Angular frequency Ω [rad/s]',
           ylabel='탐침 일률 / 힘² [m/(N s)]' if ko else 'Probe power / force² [m/(N s)]',
           title='음의 기여: 탐침에 돌아오는 일' if ko else 'Negative contribution: work returned to the probe')
    bx.legend(fontsize=12)
    for panel in [ax, bx]:
        panel.grid(alpha=.2)
    save(fig, 'thermal-response-spectrum', lang)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    eta = np.linspace(-1, 1, 400)
    forward, backward = 34/38, 11/38
    best = -(forward-backward)/(2*(forward+backward))
    info = forward*(eta+.5)**2+backward*(eta-.5)**2
    susceptibility = (135+138*eta)/1444
    ax.plot(eta, info, color=BLUE, linewidth=3)
    ax.scatter([best], [forward*backward/(forward+backward)], color=GREEN, s=65, zorder=5,
               label='최소: η* = −23/90' if ko else 'Minimum: η* = −23/90')
    ax.set(ylabel='초당 Fisher 정보 [1/s]' if ko else 'Fisher information per second [1/s]',
           title='같은 율 비율 변화, 다른 경로 정보' if ko else 'The same rate-ratio change carries different path information')
    ax.legend(fontsize=12)
    bx.plot(eta, susceptibility, color=ORANGE, linewidth=3)
    bx.axhline(0, color='#64748b', linewidth=1)
    bx.set(xlabel='교통량 손잡이 η' if ko else 'Kinetic traffic handle η',
           ylabel='정적 점유 감수율' if ko else 'Static occupancy susceptibility',
           title='최소 정보와 최대 응답은 다른 목적이다' if ko else 'Minimum information and maximum response are distinct aims')
    for panel in [ax, bx]:
        panel.set_xlim(-1, 1)
        panel.axvline(best, color=GREEN, linestyle=':', alpha=.8)
        panel.grid(alpha=.2)
    save(fig, 'control-path-information', lang)
