## 1. 평균 신호와 그 주변의 변동 구분하기

빛을 일정하게 비추어도 전류는 평균값 주변에서 변동합니다. 파장과 바이어스를 고정하고 전하가 독립적으로 도착하는 포아송 과정으로 가정합니다. 평균 전체 전류 $I=I_{\rm photo}+I_{\rm dark}\ge0$에 대해 단측 백색 산탄 잡음의 전류 전력 스펙트럼 밀도는 $S_{\rm shot}=2qI$이며, $q$는 기본 전하입니다. 암전류의 평균을 빼더라도 암전류 변동은 분산에 남습니다. [Hamamatsu 실리콘 광다이오드 기술 자료의 잡음 특성](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/si_pd_kspd9001e.pdf)

## 2. 저항 열잡음을 전류 기준으로 더하기

절대온도 $T$의 부하 저항 $R$은 고전적인 존슨 열잡음을 만듭니다. 병렬 등가 전류로 나타내면 단측 밀도는 $S_{\rm th}=4k_B T/R$입니다. 두 전력 스펙트럼 밀도의 단위는 $\mathrm{A^2/Hz}$입니다. 평균이 0인 독립 변동의 분산은 더할 수 있으므로 전력 스펙트럼 밀도를 합한 뒤 제곱근을 구합니다. $\mathrm{A/\sqrt{Hz}}$ 단위의 진폭 밀도는 전력 스펙트럼 밀도의 제곱근입니다.

## 3. 측정 필터를 적분하기

수동 RC 읽기 회로는 전류를 $Z(f)=R/(1+i2\pi fRC)$로 전압으로 바꿉니다. 관심 주파수 범위에서 두 잡음이 백색이라고 가정하면 출력 분산은 다음과 같습니다.

$$
\sigma_v^2=R^2(S_{\rm shot}+S_{\rm th})B_{\rm eq},
\qquad
B_{\rm eq}=\int_0^\infty\frac{df}{1+(f/f_c)^2}=\frac{\pi}{2} f_c=\frac{1}{4RC}.
$$

등가 잡음 대역폭은 높이가 1이고 정규화한 전달함수 크기 제곱의 적분과 넓이가 같은 직사각형의 폭입니다. 필터가 차단 주파수 위쪽의 잡음도 일부 통과시키므로 −3 dB 차단 주파수보다 큽니다. 단측 밀도를 사용하므로 양의 주파수를 적분합니다.

## 4. 저항을 두 배로 바꾸어 검산하기

$B_{\rm eq}$를 대입하면 산탄 잡음 분산은 $qIR/(2C)$, 열잡음 분산은 $k_BT/C$입니다. $C$와 $T$를 고정하고 $R$을 두 배로 하면 열잡음 스펙트럼과 대역폭이 변해도 출력에서의 열잡음 분산은 같습니다. 전류가 같으면 산탄 잡음 분산은 두 배가 됩니다. $I=1\,\mathrm{\mu A}$, $R=100\,\mathrm{k\Omega}$, $C=50\,\mathrm{pF}$, $T=300\,\mathrm K$이면 산탄 잡음과 열잡음의 RMS 전압은 약 $12.7\,\mathrm{\mu V}$와 $9.10\,\mathrm{\mu V}$이며, 합친 RMS는 약 $15.6\,\mathrm{\mu V}$입니다.

## 5. 수치 적분으로 분산 재현하기

```python
from math import atan, isclose, pi, sqrt

q, k_B = 1.602176634e-19, 1.380649e-23
current, resistance, capacitance, temperature = 1e-6, 100e3, 50e-12, 300
cutoff = 1 / (2 * pi * resistance * capacitance)
s_shot = 2 * q * current
s_thermal = 4 * k_B * temperature / resistance

n = 20000
df = 100 * cutoff / n
partial = sum(df / (1 + (((j + .5) * df) / cutoff)**2) for j in range(n))
tail = cutoff * (pi / 2 - atan(100))
bandwidth = partial + tail
assert isclose(bandwidth, 1 / (4 * resistance * capacitance), rel_tol=1e-8)
variance = resistance**2 * (s_shot + s_thermal) * bandwidth
assert isclose(variance, q * current * resistance / (2 * capacitance) + k_B * temperature / capacitance, rel_tol=1e-8)
print(f"cutoff={cutoff/1e3:.2f} kHz, ENBW={bandwidth/1e3:.2f} kHz, RMS={sqrt(variance)*1e6:.2f} µV")
```

## 6. 잔차에서 모형에 없는 잡음 찾기

같은 바이어스, 샘플링 주파수, 읽기 설정에서 암상태와 여러 광 세기의 전압 파형을 측정합니다. 백색 잡음 모형에서는 평균 광전류가 늘면 출력 분산이 선형으로 늘어납니다. 저주파에 남는 잔차는 드리프트나 초과 저주파 잡음에서 올 수 있습니다. 포화와 양자화가 측정을 바꾸는 방식도 다르므로 잘림 여부, 입력 범위, 분해능을 함께 기록합니다. 관측된 변화가 장면의 변화인지 판단하기 전에 이런 측정값을 데이터셋의 불확실성 모형에 반영합니다.
