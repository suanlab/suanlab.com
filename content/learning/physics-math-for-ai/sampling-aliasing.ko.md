## 표본 사이의 신호는 숨겨질 수 있습니다

표본기는 t=n/fₛ에서 값을 기록합니다. 그 사이에 무슨 일이 있었는지는 직접 기록하지 않습니다. 주파수 f Hz와 위상 φ를 가진 cosine에서 fₛ의 정수배를 더하거나 빼면 표본 시각의 위상은 2π의 정수배만큼만 바뀝니다.

$$
x[n]=\cos(2\pi f n/f_s+\phi)
=\cos(2\pi(f-mf_s)n/f_s+\phi),\quad m\in\mathbb Z.
$$

이 항등식 때문에 alias 모호성이 생깁니다. 그래프를 더 촘촘히 그려도 표본화에서 잃은 정보를 되찾을 수 없습니다. 실험은 위상을 유지하면서 [−fₛ/2,fₛ/2)의 부호 있는 대표 주파수를 택합니다.

## 7 Hz를 초당 8번 관측하기

f=7 Hz, fₛ=8 samples/s이면 alias는 −1 Hz입니다. n=1에서 두 값은 cos(7π/4)=cos(−π/4)=√2/2이며 n=2에서는 둘 다 0입니다. 파란 7 Hz 연속 곡선과 주황 −1 Hz 곡선은 점 사이에서 다르지만 표본 표의 값은 일치합니다.

음수 주파수는 부호 있는 복소 지수 좌표입니다. 실수 cosine에서는 위상 부호를 반대로 하면 양수 주파수로도 표현할 수 있습니다. 실험이 부호를 유지하는 이유는 위상을 몰래 바꾸지 않고 같은 위상에서 성립하는 alias 항등식을 드러내기 위해서입니다.

## Nyquist 경계를 따로 살펴야 하는 이유

f=4, fₛ=8, 위상 quarter-turn을 1로 설정하면 φ=π/2입니다. 모든 표본은 0입니다.

$$
\cos(\pi n+\pi/2)=0,\qquad n\in\mathbb Z.
$$

0이 아닌 연속 파형이 이 관측 시각들에서는 영 신호와 구분되지 않습니다. readout의 strictly-below-Nyquist 표시는 등호에서 거짓입니다. 표본이 유한한 수로 계산되거나 특정 위상에서 잘 보인다는 이유만으로 복원 조건이 충족되는 것은 아닙니다.

## 복원 정리에 필요한 조건

이상 표본화 정리는 적절한 함수 공간의 대역 제한 신호 전체를 대상으로 합니다. 일반적인 모호성 없는 설정에서는 최대 주파수 두 배보다 엄격히 높은 표본률을 사용하며, 무한 표본열·정확한 관측·이상 보간을 가정합니다. 이 유한한 단일 tone 표는 모호성을 설명하는 예제이지 정리의 전체 복원 구현이 아닙니다.

음성·영상·센서 데이터를 downsampling하기 전에는 anti-alias filter로 대역 밖 성분을 억제해야 합니다. alias가 일어난 뒤의 필터만으로 원래 주파수를 일반적으로 식별할 수는 없습니다.

## 함수 공간을 정하고 sinc 복원을 증명하기

각주파수 관례에서 f∈L²(R)의 Fourier 변환이 [−πf_s,πf_s]에 지지된다고 합시다. 유한 지지의 L² 변환은 L¹이기도 하므로 역변환이 f의 연속 대표를 정의합니다. 영측도 집합에서 임의로 바꾼 값은 이 표본이 아닙니다. 이 대역의 직교 지수함수 e^(−iωn/f_s)로 F를 전개하면 역변환에 의해 계수는 f(n/f_s)/f_s입니다. 전개를 역변환하면

$$
f(t)=\sum_{n\in\mathbb Z}f(n/f_s)\operatorname{sinc}(f_st-n),
\qquad \operatorname{sinc}(u)=\frac{\sin(\pi u)}{\pi u},\quad\operatorname{sinc}(0)=1.
$$

대역 전개의 L² 수렴과 Plancherel로 시간에서의 L² 수렴을 얻습니다. 고정 유한 대역의 Cauchy–Schwarz는 역변환 오차의 점 평가도 제어하므로 대칭 부분합의 균등 수렴도 얻습니다. 완전한 무한 표본열과 정확한 대역 제한이 중요합니다.

임계 표본률 f_s=2B에서도 이 L² 대역 제한 함수족의 표본화는 단사입니다. 대역 끝점은 측도 0입니다. 기존 Nyquist sine 반례는 R 전체에서 에너지가 무한하여 이 함수족에 속하지 않습니다. 경계의 순수 tone이나 spectral 점 질량을 허용하면 문제가 달라집니다. 엄격한 guard band는 유용한 공학적 충분조건이지만 모든 버전의 표본화 정리에 대한 필요조건은 아닙니다.

## 실제 임계 표본률 복원을 잘라 계산하기

별도 신호 f(t)=sinc(2t−.3)는 L²이고 B=1 Hz, 표본률 f_s=2 Hz입니다. t=.25에서 정확값은 약 .935489284입니다. 코드는 직사각형 spectrum의 독립 적분과 sinc를 대조한 뒤 n=−N,…,N으로 절단합니다.

| 양쪽 절단 N | 복원값 | 절대 절단 오차 |
|---|---|---|
| 4 | .898916028 | .036573255 |
| 16 | .925550542 | .009938742 |
| 64 | .932947512 | .002541772 |
| 256 | .934850136 | .000639147 |

기존 1초 tone 표보다 많은 항을 쓰는 예제이며 유한 표를 정확한 무한 복원이라 부르지 않습니다. 실습의 alias는 f를 f_s로 나눈 나머지를 먼저 구한 후 부호 대표를 골라 10⁻²⁰ Hz 같은 작은 주파수도 유지합니다. Quarter-turn 위상은 정확한 회전으로 적용하여 π/2에 작은 각을 더할 때 사라지던 sinusoid 성분을 보존합니다. 정확한 Nyquist quarter-phase의 영 표본은 유지합니다.

**확인 문제:** L²라는 조건만으로 표본값이 정해지나요? **해설:** 아닙니다. 여기서는 compact spectral support가 연속 역변환 대표를 선택합니다.

```python
from math import sin, cos, pi

def sinc(u): return 1.0 if u==0 else sin(pi*u)/(pi*u)
def simpson(f,a,b,n=10000):
    h=(b-a)/n
    return h*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))/3

# f(t)=sinc(2t-0.3), band limit B=1 Hz, sampling rate fs=2 Hz.
t=.25
truth=sinc(2*t-.3)
# Inverse Fourier integral in cycles per second: spectrum is a shifted rectangle / 2.
reference=simpson(lambda nu:.5*cos(2*pi*nu*(t-.15)),-1,1)
assert abs(reference-truth)<1e-12
for N in [4,16,64,256]:
    recovered=sum(sinc(n-.3)*sinc(2*t-n) for n in range(-N,N+1))
    print('half-count, value, error:',N,round(recovered,9),round(abs(recovered-truth),9))
# sinc is L2 but not L1; use the L2 bandlimited theorem and its continuous representative.
```

## 직접 풀기

9 Hz를 초당 8번 표본화하면 alias는 +1 Hz입니다. 위상 0과 quarter-turn 1에서 확인하세요. 표시 해상도를 높이는 것과 실제 표본률을 높이는 것이 왜 다른지 설명해 보세요. 연구에서는 표본률·관측 시간·DFT bin 간격·그림 pixel 수를 구별해야 합니다.

추가 읽기: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
