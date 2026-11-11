## 미분은 아직 gradient 벡터가 아니다

범함수의 미분은 섭동 h를 수 δJ[h]로 보내는 선형 함수다. Gradient는 선택한 Hilbert 내적에서의 Riesz 표현이다. 내적을 바꾸면 목적과 최소점은 같아도 gradient와 반복 경로가 달라질 수 있다.

실습은 끝점 0인 함수에서 적합도와 매끄러움 에너지를 사용한다.

$$
J[u]=\frac12\int_0^1\bigl((u-f)^2+\kappa(u')^2\bigr)dt,
\quad f=\sin\pi t+\tfrac12\sin(n\pi t).
$$

표시한 두 sine mode만 최적화하는 유한 Galerkin 공간이다. 그림의 표본점들을 무한 차원 수렴 정리로 해석하지 않고 모든 계산을 명시한다.

## 내적을 고르기 전에 미분하기

u=a₁sin πt+a₂sin nπt, 주파수 k=(π,nπ), 목표 계수 fᵢ=(1,0.5), Aᵢ=1+κkᵢ²로 두자. 직교성에서 계수 미분은 ∂J/∂aᵢ=(Aᵢaᵢ−fᵢ)/2다.

L2 질량행렬은 I/2이므로 함수의 L2 gradient 계수는 gᵢ=Aᵢaᵢ−fᵢ다. 계수 미분 자체와 다르다. 두 구현의 불일치가 아니라 Riesz 대응에서 생기는 배율이다.

## L2와 H1 기하 선택

H1형 내적은 ∫hv+ℓ²∫h′v′다. Mode 질량은 Mᵢ/2, Mᵢ=1+ℓ²kᵢ²이므로

$$
(\operatorname{grad}_M J)_i=\frac{A_i a_i-f_i}{M_i},
\qquad a_i^+=a_i-\eta\frac{A_i a_i-f_i}{M_i},
\qquad a_i^*=\frac{f_i}{A_i}.
$$

최소점은 metric과 무관하다. H1은 높은 주파수에 더 큰 내적 가중치를 주어 갱신의 전처리처럼 작동한다. ℓ=0이면 이 유한 공간에서 정확히 L2다.

충분히 정칙인 전체 함수에서는 강한 L2 gradient가 u−f−κu″이고 경계 처리도 필요하다. 임의 H₀¹ 함수는 L2 이계 도함수가 없을 수 있어 그 공간에서 L2 gradient가 존재하지 않을 수도 있다. 약한 미분과 H1 Riesz 표현은 여전히 의미 있을 수 있다. 유한 모형 밖에서는 함수 공간과 정칙성을 생략하면 안 된다.

## 선택한 내적에서의 안정성과 감소

Mode 오차 배율은 1−ηAᵢ/Mᵢ다. 화면은 δJ[−grad_M J]=−‖grad_M J‖²_M도 확인하며 제곱 노름은 Σgᵢ²/(2Mᵢ)다. 방향 미분이 음수라는 것과 임의의 큰 유한 보폭 감소는 다르다.

n=4,κ=0.1,η=0.3의 L2는 높은 mode 배율 약 −4.03741로 발산한다. H1 길이 0.2에서는 최악 배율이 약 0.572631이다. 두 최적 계수는 모두 약 (0.503281,0.029777)로 같다.

## 해상도가 높아져도 전처리가 유용한 조건

Mode 논증을 주파수 ω=nπ로 확장하자. κ>0, ℓ>0에서 r=κ/ℓ², s=ℓ²ω²라고 쓰면 일반화 곡률은 가중평균이다.

$$
\frac{1+\kappa\omega^2}{1+\ell^2\omega^2}
=\frac{1+rs}{1+s}\in[m,L],\qquad
m=\min(1,r)>0,\quad L=\max(1,r).
$$

고정 0<η<2/L에서 모든 mode 오차 배율의 절댓값은 ρ=max(|1−ηm|,|1−ηL|)<1 이하이다. Mode 에너지 오차를 합하면 유지한 mode 수에 무관하게 한 보폭마다 최대 ρ²로 수축한다. 무한 전개로 합을 확장하려면 초기 에너지가 유한해야 한다. 두 mode 그림의 표본에서 추정한 주장이 아니라 스펙트럼 논증이다.

ℓ²=κ이면 모든 일반화 곡률은 1이다. η=1의 정확한 metric 보폭은 이차 문제를 한 번에 푼다. 다만 metric 역의 적용 자체가 타원형 방정식 풀이이므로 비용 없는 알고리즘이라는 뜻은 아니다. 아래 프로그램은 실습 보폭 범위 안인 η=1/4을 사용한다.

반대 극한 두 가지가 조건을 설명한다. κ>0의 L2에서는 곡률이 ω²로 커져 임의로 높은 mode까지 안정한 양의 고정 보폭이 없다. κ=0, ℓ>0에서는 일반화 곡률이 0으로 가므로 각 유한 절단은 수축해도 모든 주파수에 공통인 1 미만의 수축 배율은 없다.

| 설정 | 고주파 곡률 | 균일한 결론 |
|---|---|---|
| κ>0, ℓ=0 | 무한히 증가 | 안정한 고정 보폭 없음 |
| κ=ℓ²>0 | 1 | 배율 1−η |
| κ=0, ℓ>0 | 0으로 접근 | 엄격한 공통 배율 없음 |

```python
from math import pi
eta, kappa = 0.25, 0.1
for n in (2, 8, 128):
    omega_squared = (n*pi)**2
    A = 1+kappa*omega_squared
    matched_metric = A  # ell^2=kappa, avoiding a rounded sqrt-and-square.
    matched_factor = 1-eta*A/matched_metric
    l2_factor = 1-eta*A
    unregularized_factor = 1-eta/matched_metric
    assert matched_factor == 0.75
    print(n, "L2:", round(l2_factor, 6), "matched:", matched_factor,
          "kappa=0,H1:", round(unregularized_factor, 6))
```

큰 주파수는 조절기 범위 2–8 밖에서 수행하는 별도의 해상도 연구다. Metric은 최소점을 바꾸지 않는다. 목적 정규화 κ와 metric 길이 ℓ는 일부러 맞추어도 서로 다른 역할이다.

![주파수가 커질 때 L2, 일치하는 H1, 정규화가 없는 H1의 일반화 곡률과 절대 오차 배율을 비교합니다.](/assets/learning/variational-metric-ko.svg)

## 직접 풀기와 풀이

H1을 12회 반복하면 목적 초과는 약 1.94×10⁻⁷이다. 내적만 L2로 바꾸어 고주파 계수가 커지는지 보고, H1 길이를 0으로 하여 같은 L2 결과를 재현하자. 적합 기준·정규화 매개변수·최소 함수는 바꾸지 않고 반복의 기하만 개선한 이유를 설명하자.

더 읽기: [MIT의 함수 내적·변분 강의](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
