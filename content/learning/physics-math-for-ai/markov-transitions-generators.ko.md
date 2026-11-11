## 조건부 전이 법칙부터 명시한다

Markov 모형은 현재 상태가 주어지면 다음 전이에 더 과거의 정보가 필요하지 않다고 말합니다. 이웃 상태가 독립이라는 뜻은 아닙니다. 상태 0·1에서 a는 0→1, b는 1→0의 확률입니다.

$$
P=\begin{pmatrix}1-a&a\\b&1-b\end{pmatrix},\qquad
p_{n+1}=(1-p_n)a+p_n(1-b).
$$

p_n은 state1의 주변 확률입니다. 행은 현재 상태에 대한 조건부 법칙이고 상태 확률 벡터는 현재의 불확실성입니다. 둘을 섞으면 행렬 모양은 맞아도 의미가 틀릴 수 있습니다.

## 수렴 곡선 대신 실제 법칙을 반복한다

이산 그림은 실제 확률 갱신입니다. a=0.2, b=0.4, 초기 state1이면 1→0.6→0.44입니다. a+b>0의 정상 확률은 a/(a+b)이지만 정상 법칙의 존재가 모든 경우의 수렴을 보장하지는 않습니다.

위 그림의 화살표는 선택한 구간의 P^n, 원 안은 현재 주변 확률입니다. n=0이면 화살표 행렬은 항등입니다. 항상 한 step의 전이라고 읽지 마세요. 주기성·기약성·시간 평균은 뒤의 ergodicity 단원에서 보강합니다.

## 연속시간에는 Generator가 필요하다

연속 모형에서 a,b는 step 확률이 아니라 시간 역수 단위의 jump rate입니다.

$$
Q=\begin{pmatrix}-a&a\\b&-b\end{pmatrix},\qquad P(t)=e^{tQ}.
$$

a+b>0에서 state1 확률은 pi1+(p0-pi1)exp(-(a+b)t)입니다. 구현은 시간 0 근처에서 expm1을 사용합니다. 두 rate가 0이면 모든 상태가 흡수되어 항등 전이를 유지하며 유일한 정상 법칙을 만들어 내지 않습니다.

## 역전 조건부 확률에는 Support가 필요하다

선택한 horizon 행렬 K와 초기 p에서 Bayes는 R_ji=p_i K_ij/pnext_j를 줍니다. 양 끝 상태의 joint law를 뒤집는 식입니다. pnext_j=0이면 이 joint law만으로 그 조건부 행을 정할 수 없습니다. 표는 0으로 나누거나 임의 행을 채우지 않고 비워 둡니다.

## Uniformization으로 연속 전이 구성하기

유한 상태 generator Q에서 γ≥maxᵢ(−Qᵢᵢ), γ>0을 택하고 U=I+Q/γ라 두면 U는 확률 행렬이다. I와 U가 가환하므로 지수 전개로

$$
e^{tQ}=e^{-\gamma t}\sum_{n=0}^\infty\frac{(\gamma t)^n}{n!}U^n.
$$

Poisson 후보 사건과 자기 전이를 허용하는 이산 전이로 연속 과정을 구성한다. 행렬 지수에 확률 의미를 가정하는 대신 비음수 원소와 행합 1을 증명한다. K항 뒤에서 자르면 빠진 행 질량은 Poisson 꼬리다. 잘린 행을 정규화하면 다른 kernel 근사다. Q=0이면 상수 과정이며 양의 사건률이 필요 없다.

두 rate a=b=1에서는 γ=1, U는 교대 전이다. 교대 횟수를 무작위화하면 양의 연속 시간의 결정론적 주기가 사라진다. 남을 확률은 (1+exp(−2t))/2다.

| t | 연속 P₁₁(t) |
|---|---|
| 0 | 1 |
| 1/2 | (1+e⁻¹)/2 |
| 1 | (1+e⁻²)/2 |

```python
from math import exp
for t in (0., .5, 1.):
    weight, same, other = exp(-t), 0., 0.
    for n in range(41):
        if n % 2: other += weight
        else: same += weight
        weight *= t/(n+1)
    assert abs(same-(1+exp(-2*t))/2) < 1e-14
    assert abs(same+other-1) < 1e-14
    print(t, same, other)
```

Poisson 혼합으로 닫힌 식을 독립 검산한다. 실제 Bayes 계산은 joint 곱의 underflow를 support 소실로 취급하지 않도록 수정했다. 주변 확률이 0으로 반올림되어도 양의 인자들이 남으면 로그 가중치를 스케일링해 조건부 행을 유지한다. 실제 support가 없는 끝점은 계속 비운다. 임의 정밀도 주변 확률을 제공한다는 뜻은 아니다.

## 직접 풀기: 서로 다른 세 행렬

a=b=1의 이산 전이는 교대하므로 P²=I입니다. 연속시간에서는 rate이므로 t=1에 state1에 남을 확률은 (1+exp(-2))/2이며 0이 아닙니다. Generator와 유한시간 전이를 같게 두면 왜 틀리는지 설명하세요. a=b=0, 초기 state1에서는 어떤 reverse 행을 조건부로 정의할 수 없는지도 구하세요.

더 읽기: [MIT 확률과정 노트](https://ocw.mit.edu/courses/18-445-introduction-to-stochastic-processes-spring-2015/).
