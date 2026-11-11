## Kernel을 명시적인 특징으로 바꾸기

큰 Gram 행렬 대신 작은 특징 벡터를 저장하면 선형 계산으로 kernel 방법을 근사할 수 있습니다. 이전 고차원 확률 단원에서 본 한 쌍의 특징 오차를 여기서는 함수 절편과 전체 작은 Gram 구조로 확장합니다. 근사 정확도와 PSD 구조는 다른 성질입니다.

RBF kernel은 이상적인 독립 주파수 ω~N(0,ℓ⁻²)에 대해 E cos(ω(x−z))입니다. Gaussian 특성함수에서 이 식이 나옵니다.

$$
k(x,z)=\exp\!\left(-\frac{(x-z)^2}{2\ell^2}\right)=E_\omega\cos(\omega(x-z)),\quad\omega\sim N(0,\ell^{-2}).
$$

## 짝지은 cos와 sin 특징

각 주파수에 cos와 sin 좌표를 붙이고 전체를 √m으로 나눕니다. 그러면 삼각함수 항등식으로 근사 kernel을 얻습니다.

$$
\begin{aligned}
\phi_m(x)&=\frac1{\sqrt m}(\cos\omega_jx,\sin\omega_jx)_{j=1}^m,\\
\widetilde k(x,z)&=\phi_m(x)^T\phi_m(z)=\frac1m\sum_j\cos(\omega_j(x-z)).
\end{aligned}
$$

m=4는 4차원이 아니라 8차원입니다. 대각은 cos²+sin² 덕분에 항상 1입니다. 세 입력의 특징 행렬을 Z라고 하면 근사 Gram은 ZZᵀ이며 cᵀZZᵀc=‖Zᵀc‖²≥0입니다. 참 kernel과 다르더라도 이 PSD 성질은 유지됩니다.

## 실험에서 비교하는 것

파랑은 정확한 RBF 절편 k(x,z), 주황은 같은 z의 특징 근사입니다. 표는 실제 주파수와 질의의 특징값을 보여줍니다. Gram Frobenius 오차는 −1,0,1의 세 입력에 대한 오차이고, 표시 격자 최대 오차는 유한한 121개 질의에 한정합니다. 두 수치를 영역 전체 supremum이나 학습 일반화 오차라고 부르지 않습니다.

seed는 재현 가능한 작은 의사난수 발생기의 고정 실행입니다. 이상적인 독립 Gaussian 표본에서 성립하는 불편성·집중 경계가 고정 seed 하나의 오차 상한을 보장하지는 않습니다. m을 늘려도 그 실행의 오차가 매 단계 단조 감소할 필요는 없습니다. 소표본 오차가 잠시 커지는 것은 이론의 모순이 아닙니다.

## 계산과 학습의 새로운 교환

특징 차원은 계산·저장 비용뿐 아니라 근사된 함수 공간을 바꿉니다. kernel ridge를 특징 ridge로 바꿀 때 정규화와 데이터 오차도 함께 달라질 수 있습니다. 특징 수가 충분한지 평가하려면 목표 kernel, 입력 영역, 표본 크기, 요구 오차와 확률 조건을 명시해야 합니다.

## 확률 공간을 명시하고 오차 명제를 유도하기

d=x−z를 고정하고 이상적인 독립 ω_j∼N(0,ℓ⁻²)를 뽑습니다. u=d²/ℓ²라 두면 특성함수와 cos²t=(1+cos2t)/2로

$$
E\widetilde k=k,\qquad
E(\widetilde k-k)^2=\frac{(1-e^{-u})^2}{2m}.
$$

독립성 때문에 한 주파수의 분산을 m으로 나눕니다. d=0이면 분산은 정확히 0이고 짝지은 특징의 대각은 매 실현에서 정확합니다. 아래 d=ℓ의 값은 주파수에 대한 기대 제곱 오차이며 실습 seed의 오차 보장이 아닙니다.

| 주파수 수 m | 이상적 기대 제곱 오차 |
|---|---|
| 4 | .04994705 |
| 16 | .01248676 |
| 64 | .00312169 |
| 256 | .00078042 |

Cosine은 [−1,1]에 있어 한 쌍의 절대 오차 꼬리에 Hoeffding의 2exp(−mε²/2)를 적용합니다. 고정한 n개 입력의 모든 Gram 원소에는 union bound 2n²exp(−mε²/2)를 얻습니다. n=3,ε=.2,δ=.05이면 이 보수적인 경계를 만족하는 충분한 m은 295입니다. 실습 최대 32를 넘는 별도 이론 비교입니다. 연속 영역 supremum이나 고정 의사난수 실행의 보장은 이 유한 집합 명제로 나오지 않습니다. 코드는 Gaussian cosine 모멘트를 독립 적분합니다.

## Kernel 오차를 정규화된 훈련 예측으로 연결하기

PSD Gram G,G̃와 λ>0에서 smoother의 resolvent 항등식은

$$
\widetilde S-S=\lambda(\widetilde G+\lambda I)^{-1}
(\widetilde G-G)(G+\lambda I)^{-1},
\qquad\|\widetilde S-S\|_{\rm op}\le\frac{\|\widetilde G-G\|_{\rm op}}\lambda.
$$

두 역행렬 노름이 각각 1/λ 이하이므로 훈련 예측 차이는 ‖G̃−G‖op‖y‖/λ 이하입니다. 원소별 오차 ε라면 nε‖y‖/λ가 충분한 경계입니다. 약한 정규화에서는 근사에 민감해집니다. 이 경계는 해당 훈련 입력의 예측을 제어하며 자료 밖 절편이나 모집단 일반화까지 제어하지는 않습니다. λ=0에는 이 논증으로 경계를 주지 못합니다.

**확인 문제:** 근사 Gram이 PSD이면 예측 오차도 작은가요? **해설:** 아닙니다. PSD는 유효한 이차형식을 보존하고, 가까움과 정규화된 해의 안정성에는 위의 추가 추정이 필요합니다.

```python
from math import exp, cos, sqrt, pi, ceil, log

def simpson(f,a,b,n=20000):
    h=(b-a)/n
    return h*(f(a)+f(b)+sum((4 if i%2 else 2)*f(a+i*h) for i in range(1,n)))/3

# Independent integration of the first and second moments at gap/length=1.
density=lambda w: exp(-w*w/2)/sqrt(2*pi)
first=simpson(lambda w: cos(w)*density(w),-10,10)
second=simpson(lambda w: cos(w)**2*density(w),-10,10)
var=(1-exp(-1))**2/2
assert abs(first-exp(-.5))<1e-11
assert abs(second-first*first-var)<1e-11
for m in [4,16,64,256]: print(m,round(var/m,8))
n,epsilon,delta=3,.2,.05
print('sufficient ideal iid frequencies:',ceil(2/epsilon**2*log(2*n*n/delta)))
# This bound is for n^2 fixed pairs, not all points in a continuum or a fixed PRNG seed.
```

## 확인 문제와 해설

1. 근사 kernel 값이 음수가 나오면 PSD가 아닌가요? **아니요.** Gram 이차형식의 비음수가 PSD 조건입니다.
2. m=32의 실제 좌표 수는? **64개입니다.** 짝지은 두 좌표를 셉니다.
3. 한 kernel 절편이 잘 맞으면 학습 결과도 같나요? **보장하지 않습니다.** 전체 Gram 오차와 정규화된 해의 안정성도 봐야 합니다.

추가 읽기: [Rahimi and Recht, Random Features](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf).
