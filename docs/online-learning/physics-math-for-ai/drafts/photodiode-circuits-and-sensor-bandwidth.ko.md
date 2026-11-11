## 1. 빛에서 전류와 전압으로

실리콘 광다이오드가 광자를 흡수하면 움직일 수 있는 전자와 정공이 생길 수 있습니다. 공핍층의 전기장은 이들을 분리하고 외부 회로에서는 전류를 읽습니다. 파장을 고정하고 선형 동작 범위에서 광전류 변화를 $i_{\rm ph}=\mathcal{R}p$로 나타냅니다. $p$는 광 파워의 변화량이고, 응답도 $\mathcal{R}$의 단위는 A/W입니다. 응답도, 접합 정전용량, 암전류는 센서의 서로 다른 특성입니다. [Hamamatsu 실리콘 광다이오드 기술 자료](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/si_pd_kspd9001e.pdf)

## 2. 수동 읽기 회로의 식 유도하기

작은 신호 범위에서 광전류원을 부하 저항 $R$, 유효 정전용량 $C$와 병렬로 둡니다. $C$에는 접합과 회로 입력의 정전용량이 포함됩니다. 부하 저항은 센서 내부 병렬 저항보다 충분히 작고 바이어스 동작점은 고정되어 있다고 가정합니다. 축전기로 $C\dot v$, 부하로 $v/R$의 전류가 흘러 다음 식을 얻습니다.

$$
RC\,\dot v+v=R\mathcal{R}p.
$$

빛을 계단처럼 켜면 입력은 즉시 바뀝니다. 축전기에 전하가 쌓이면서 전압은 연속적으로 변합니다. 초기 전압이 0일 때 크기 $p_0$의 계단 입력에 대해 $v(t)=R\mathcal{R}p_0(1-e^{-t/(RC)})$입니다. $R$이 크면 최종 전압이 커지며, 최종값의 10%에서 90%까지 걸리는 시간은 $RC\ln 9$입니다.

## 3. 주파수에 따른 크기와 위상 읽기

광 파워를 주파수 $f$로 정현파 변조하면 광 파워 대비 전압의 전달함수는 다음과 같습니다.

$$
G(f)=\frac{R\mathcal{R}}{1+i2\pi fRC}.
$$

저주파 이득에 대한 진폭 비는 $1/\sqrt{1+(2\pi fRC)^2}$입니다. 차단 주파수 $f_c=1/(2\pi RC)$에서 진폭 비는 $1/\sqrt 2$로 약 −3 dB이며 위상은 −45°입니다. 저항을 두 배로 하면 저주파 이득은 두 배, 차단 주파수는 절반이 됩니다. 정전용량을 절반으로 줄이면 이 모형에서 저주파 이득은 같고 차단 주파수는 두 배가 됩니다.

$R=100\,\mathrm{k\Omega}$, $C=50\,\mathrm{pF}$, $\mathcal{R}=0.5\,\mathrm{A/W}$이면 시간 상수는 $5\,\mathrm{\mu s}$이고 차단 주파수는 약 $31.8\,\mathrm{kHz}$입니다. $10\,\mathrm{\mu W}$ 계단 입력의 최종 전압은 $0.5\,\mathrm V$이며 10–90% 상승에는 약 $11\,\mathrm{\mu s}$가 걸립니다.

## 4. 실제 회로에서 응답을 제한하는 부분 찾기

역방향 바이어스가 변하면 정전용량, 전하 수집 시간, 암전류도 달라질 수 있습니다. 실제 계단 응답에는 광원의 상승 시간, 전하 이동, 기생 정전용량, 디지타이저 필터도 반영됩니다. 입력 파형을 측정하거나 교정한 광원을 사용합니다. 바이어스, 파장, 부하, 이득 설정을 기록하고 한 번에 한 요소를 바꿔 차단 주파수가 RC 예측을 따르는지 확인합니다. 능동 트랜스임피던스 증폭기는 되먹임이 있어 대역폭과 안정성을 별도로 계산합니다. [Texas Instruments 광다이오드 증폭기 회로](https://www.ti.com/tool/CIRCUIT060018)

## 5. Python으로 시간 응답과 주파수 응답 비교하기

```python
from math import exp, isclose, log, pi, sqrt

responsivity = 0.5
optical_step = 10e-6

for resistance, capacitance in [(100e3, 50e-12), (200e3, 50e-12), (100e3, 25e-12)]:
    tau = resistance * capacitance
    cutoff = 1 / (2 * pi * tau)
    t10, t90 = -tau * log(0.9), -tau * log(0.1)
    final = resistance * responsivity * optical_step
    voltage = lambda t: final * (1 - exp(-t / tau))
    assert isclose(voltage(t10) / final, 0.1)
    assert isclose(voltage(t90) / final, 0.9)
    assert isclose(abs(1 / (1 + 2j * pi * cutoff * tau)), 1 / sqrt(2))
    print(f"final={final:.3f} V, cutoff={cutoff / 1e3:.1f} kHz, rise={(t90-t10)*1e6:.1f} µs")
```

## 6. AI 데이터 파이프라인의 교정 실험 설계하기

센서가 디지털화한 전압으로 입사광 펄스를 추정하는 상황을 생각해 봅니다. 짧은 두 펄스가 달라도 읽기 회로가 고주파 성분을 억제하면 출력이 비슷할 수 있습니다. 전압 기록과 함께 샘플링 주파수와 아날로그 전달함수를 저장합니다. 부하나 정전용량을 바꾼 조건에서 일반화 성능을 검사하면 측정 응답의 변화와 입사광의 변화를 함께 다루게 됩니다.
