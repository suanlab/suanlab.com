## 왜 파동을 더해 신호를 표현할까요?

신호를 시간별 값으로 읽으면 언제 변하는지 알 수 있고, 진동 성분으로 읽으면 어떤 변화 속도가 필요한지 알 수 있습니다. Fourier 해석은 내적을 이용해 좌표를 바꾸는 방법입니다. 좌표를 바꾼다고 비선형 상호작용까지 사라지는 것은 아닙니다.

먼저 주기 2π의 사각파를 생각합니다. (−π,0)에서는 −1, (0,π)에서는 +1이고 점프에서는 중간값 0을 택합니다. 홀함수이므로 cosine 계수는 0이며 sine과 적분하면 홀수 주파수만 남습니다.

$$
b_k=\frac1\pi\int_{-\pi}^{\pi}f(t)\sin(kt)\,dt,
\qquad S_M(t)=\frac4\pi\sum_{j=0}^{M-1}\frac{\sin((2j+1)t)}{2j+1}.
$$

항 하나의 꼭대기는 4/π≈1.273입니다. 네 항에는 주파수 1,3,5,7이, 열두 항에는 23까지 들어갑니다. 빠른 진동을 더해야 짧은 시간에 일어나는 급격한 변화를 표현할 수 있습니다.

## 불연속점과 그림을 정확히 읽기

파란 수평선은 대상 신호입니다. 빈 점은 제외한 한쪽 극한값, 채운 점은 점프의 중간값입니다. 주황 선은 유한 부분합입니다. t=0에서는 모든 sine 항이 0이므로 항 수와 관계없이 부분합도 0입니다. 이 구간별 매끄러운 주기 함수의 점프에서는 좌우 극한의 평균으로 수렴합니다.

점프를 포함한 균등수렴은 성립하지 않습니다. 항이 늘면 진동 구역은 좁아지지만 Gibbs 현상의 극한 overshoot는 없어지지 않습니다. 화면의 overshoot는 유한한 격자에서 계산한 최대값이며 그 극한 상수를 정확히 계산한 값이 아닙니다. 질의를 0 가까이 옮긴 뒤 항 수를 늘려 점별수렴과 균등수렴의 차이를 관찰하세요.

## 주기 없는 pulse의 Fourier 변환

Gaussian 모드에서는 각주파수 ω와 다음 적분 관례를 씁니다.

$$
f(t)=e^{-t^2/(2\sigma^2)},\qquad
F(\omega)=\int_{\mathbb R}f(t)e^{-i\omega t}\,dt
=\sigma\sqrt{2\pi}e^{-\sigma^2\omega^2/2}.
$$

σ=1이면 F(0)=√(2π)≈2.507이며 시간 신호의 전체 면적과 같습니다. σ가 커지면 pulse는 넓어지고 스펙트럼은 좁아집니다. 겹친 그림에서 파랑 가로축은 시간, 주황 가로축은 각주파수입니다. 같은 위치가 같은 물리량을 뜻하지 않습니다. 이 모드에서는 항 수 조작을 사용하지 않습니다.

## 양의 평균은 다른 수렴 명제를 증명한다

|k|≤n을 남기는 표준 Fourier 부분합 S_n에서 Fejér 평균 σ_N=(S₀+⋯+S_{N−1})/N을 정의합니다. |k|<N인 계수에 1−|k|/N을 곱하는 연산이며 convolution kernel은

$$
F_N(u)=\frac1N\left(\frac{\sin(Nu/2)}{\sin(u/2)}\right)^2,
\qquad F_N\ge0,\qquad\frac1{2\pi}\int_{-\pi}^{\pi}F_N=1.
$$

u=0의 값은 제거 가능한 극한 N입니다. 제곱을 유한 합으로 전개하면 적분 정규화를 확인합니다. δ≤|u|≤π에서는 F_N≤1/[N sin²(δ/2)]이므로 0의 고정된 이웃 밖 질량이 0으로 갑니다. Convolution 오차를 그 이웃과 나머지로 나누면 첫 부분은 균등연속성, 둘째는 줄어드는 꼬리로 제어됩니다. 따라서 연속 주기 함수의 Fejér 평균은 균등 수렴합니다.

사각파는 불연속입니다. 그래도 Fejér 평균은 −1과 1 사이에 있습니다. 그 범위의 함수를 비음수 정규화 가중치로 평균하기 때문입니다. 이는 화면의 원래 부분합과 다른 절차입니다. 연속 함수 열이 균등 수렴하면 극한도 연속이므로, 항 수만 늘려 불연속 극한의 이 장애를 없애지는 못합니다.

## 줄어드는 규모에서 사라지지 않는 Gibbs 초과량 유도하기

실습의 M개 홀수 항을 미분하면 S′_M(t)=(2/π)sin(2Mt)/sin t입니다. 0부터 적분하고 y=2Mt라 두면

$$
S_M\!\left(\frac y{2M}\right)=\frac2\pi\int_0^y
\frac{\sin u}{2M\sin(u/(2M))}\,du
\longrightarrow\frac2\pi\int_0^y\frac{\sin u}{u}\,du.
$$

첫 양의 최대는 t=π/(2M)에 있습니다. 극한 높이는 약 1.1789797445이며 위쪽 수준 1보다 .1789797445만큼 높습니다. 크기 2인 점프에 대한 비율은 .0894898722입니다. 점프의 17.9%라고 부르면 배율을 틀립니다.

| 홀수 항 수 M | 첫 최대값 | 초과량 / 점프 크기 |
|---|---|---|
| 1 | 1.27323954 | .13661977 |
| 4 | 1.18422513 | .09211257 |
| 16 | 1.17930541 | .08965270 |
| 64 | 1.17900009 | .08950005 |

![좌표 y=2Mt로 확대하면 M이 늘어도 사각파 부분합의 초과량이 유지되며 전이 모양은 sine integral의 극한 곡선에 접근한다.](/assets/learning/harmonic-gibbs-scale-ko.svg)

코드의 M=16·64와 고정 그림은 실습 최대 12를 넘는 별도 수식 예제입니다. 극한 sine integral은 수치 적분으로 계산하며, 실습의 정확한 점프 좌표 0·±π에서는 삼각함수 반올림 잔차 대신 부분합 0을 유지합니다.

**확인 문제:** 고정한 양의 t에서는 1로 수렴하는데 첫 최대값은 왜 그렇지 않나요? **해설:** 최대 위치가 M과 함께 점프 쪽으로 이동하기 때문입니다. 같은 점에서 취한 극한이 아닙니다.

```python
from math import sin, pi

def simpson(f,a,b,n=20000):
    h=(b-a)/n
    return h*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))/3

si_pi=simpson(lambda u: 1 if u==0 else sin(u)/u,0,pi)
limit=2*si_pi/pi
for m in [1,4,16,64]:
    t=pi/(2*m)
    peak=sum(4*sin((2*j+1)*t)/(pi*(2*j+1)) for j in range(m))
    print('m, first peak, overshoot / jump:',m,round(peak,8),round((peak-1)/2,8))
print('scaled limit:',round(limit,10))
# Standard Fejer mean: weights are 1-|k|/N, not an unweighted partial sum.
N=32
def fejer(t):
    return sum(4*(1-k/N)*sin(k*t)/(pi*k) for k in range(1,N,2))
assert max(abs(fejer(-pi+2*pi*j/2000)) for j in range(2001))<=1+1e-12
# The finite grid is a check; the nonnegative convolution kernel proves the global bound.
```

## 직접 풀기와 AI 연결

σ를 두 배로 하면 F(0)은 어떻게 될까요? 높이를 유지한 채 폭이 두 배가 되므로 면적과 F(0)도 두 배입니다. 수치 표로 확인하세요. 이어 유한 개의 매끄러운 sine 합이 사각파를 정확히 표현할 수 없는 이유를 설명해 보세요. AI의 spectral layer와 위치 특징도 진동 좌표를 활용하지만, 표본화·경계·학습 연산의 조건은 별도로 따져야 합니다. 적분 가능한 신호의 보통 변환, L² 확장과 distribution 변환의 존재 조건도 구별해야 합니다.

추가 읽기: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
