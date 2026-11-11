## 분포 에너지를 낮추는 gradient flow

Free energy F(ρ)=∫x²ρ/2+T∫ρlogρ, T≥0를 두자. 변분 미분의 속도는 v=−x−T∂ₓlogρ이고 연속식과 결합하면 Ornstein–Uhlenbeck 방정식이다.

$$
\partial_t\rho=\partial_x(x\rho)+T\partial_{xx}\rho.
$$

Gaussian 초기값에서는 m′=−m, (σ²)′=−2σ²+2T다. 따라서 m(t)=m₀e⁻ᵗ, σ²(t)=σ₀²e⁻²ᵗ+T(1−e⁻²ᵗ)이다. 시간 0 근처에서는 안정적인 지수 나머지로 계산한다.

## 밀도 혼합과 다른 변분 보폭

Jordan–Kinderlehrer–Otto 단계는

$$
\rho_{k+1}=\arg\min_\rho
\left[\frac{W_2^2(\rho,\rho_k)}{2h}+F(\rho)\right].
$$

이 문제에서는 이전 상태가 Gaussian이면 Gaussian 후보가 전체 최소도 달성한다. 평균·분산을 고정할 때 Gaussian은 음의 entropy를 최소화하고 W₂²의 모멘트 하한도 affine Gaussian coupling으로 달성하기 때문이다. 편의상 가족을 가정한 것과 구별하자.

상수 항을 생략한 에너지는 (m²+σ²)/2−Tlogσ다. 근접 목적을 최소화하면

$$
m^+=\frac{m}{1+h},\qquad
(1+h)(\sigma^+)^2-\sigma\sigma^+-hT=0.
$$

양의 근으로 다음 표준편차를 얻는다. 분산 미분방정식에 implicit Euler를 직접 적용한 식과는 다르다. 좌표와 Wasserstein 기하를 함께 읽어야 한다.

## 같은 물리 시간에서 비교

파란 선은 kh에서 정확한 연속 에너지, 주황 선은 실제 JKO 반복이다. 유한 변분 단계는 정확한 시간 적분이 아니다. 총시간을 고정하고 h를 줄여 에너지 감소와 시간 이산화 오차를 따로 비교할 수 있다.

m₀=σ₀=T=1,h=0.5,두 단계이면 σ=1, JKO 평균은 4/9다. 시간 1의 연속 평균은 e⁻¹≈0.367879라 W₂ 차이는 약 0.076565다. T=0이면 평균과 표준편차 모두 4/9가 되어 차이는 약 0.108279다.

## 에너지 감소로 지불하는 이동 비용

이전 법칙을 경쟁 후보로 넣으면

$$
F(\rho_{k+1})+W_2^2(\rho_{k+1},\rho_k)/(2h)\le F(\rho_k).
$$

합하면 누적 이동 비용을 전체 에너지 감소가 상계한다. 표와 수치에서 두 값을 따로 읽고, 근접 정상성 잔차도 검사한다.

## Gaussian 축소를 정당화하고 시간 오차 제한하기

평균 m,m₀, 표준편차 σ,σ₀의 임의 coupling에서 중심 공분산은 Cauchy–Schwarz로 σσ₀ 이하이다. E|X−Y|²를 전개하면 W₂²≥(m−m₀)²+(σ−σ₀)²다. 이 모멘트를 갖고 entropy가 유한한 밀도 ρ에서 KL(ρ||Gaussian(m,σ²))의 비음수성으로 Gaussian의 ∫ρlogρ가 가장 작음을 얻는다. Affine Gaussian coupling이 거리 하한을 달성하므로 두 하한을 함께 달성하여 앞의 전역 Gaussian 축소가 정당화된다. Entropy 상계에는 T≥0을 곱한다.

T=0에서는 JKO의 평균과 표준편차 모두 매번 1/(1+h)배다. 물리 시간 t=nh에서 h−h²/2≤log(1+h)≤h를 쓰면

$$
W_2(\rho_n,\rho(t))
\le\sqrt{m_0^2+\sigma_0^2}\,e^{-t}\operatorname{expm1}(th/2).
$$

δ=t−nlog(1+h)는 [0,th/2]에 속하고 실제 오차는 √(m₀²+σ₀²)e⁻ᵗ(expm1 δ)이기 때문이다. 그림 하나로 차수를 추측하는 대신 이 영 온도 Gaussian 모형의 고정 시간 일차 오차 상계를 증명한다. T>0에서도 평균 상계는 같지만 scale에는 별도 분석이 필요하다.

| h | n, t=1 | 비교 상태 |
|---|---:|---|
| 1/2 | 2 | (1+h)⁻ⁿ과 e⁻¹ |
| 1/4 | 4 | (1+h)⁻ⁿ과 e⁻¹ |
| 1/8 | 8 | (1+h)⁻ⁿ과 e⁻¹ |

```python
from fractions import Fraction as F
from math import exp, expm1, sqrt
for h in (F(1, 2), F(1, 4), F(1, 8)):
    n = int(1/h)
    state = F(1)
    movement = F(0)
    for _ in range(n):
        next_state = state/(1+h)
        movement += (next_state-state)**2/h  # two equal coordinates.
        state = next_state
    drop = 1-state*state
    error = sqrt(2)*(float(state)-exp(-1))
    bound = sqrt(2)*exp(-1)*expm1(float(h)/2)
    assert movement <= drop
    assert 0 <= error <= bound
    print(h, n, "W2 error/bound:", error, bound, "energy drop:", drop)
```

평형 scale과 작은 평균에서는 절대 free energy들이 같은 수로 반올림되어도 이동은 양수일 수 있다. 실습은 평균의 곱과 scale의 로그 나머지로 에너지 감소를 계산하여 평균 10⁻²⁰을 보존한다. 큰 절대 에너지들의 반올림된 차이를 소산 인증으로 사용하지 않는다. 계수 T−σ₀²에도 반올림된 제곱의 낮은 부분을 반영하여 평형 근처의 선도항을 유지한다. 절대 에너지는 별도로 표시하며 유한 정밀도의 상태·scale 반올림 한계는 남는다.

![같은 물리 시간 1에서 영 온도 Gaussian JKO의 시간 간격을 줄이면 W2 오차가 감소하며 증명한 상계 아래에 있습니다.](/assets/learning/transport-jko-error-ko.svg)

## 직접 풀기와 풀이

m₀=0,σ₀=T=1이면 표준 Gaussian이 평형이라 상태·에너지 변화와 이동 비용이 모두 0이다. T=0의 Dirac 집중은 시간 극한에서 생기며 유한 시간의 양의 표준편차를 폭 0 밀도로 바꾸면 안 된다.

Free-energy gradient flow, 고정 끝점 사이 최단 경로, 밀도의 선형 mixture가 각각 어떤 문제를 푸는지 설명하자. 모두 분포를 바꾸지만 같은 최적화는 아니다.

더 읽기: [Peyré와 Cuturi의 최적수송 교재](https://optimaltransport.github.io/book/).
