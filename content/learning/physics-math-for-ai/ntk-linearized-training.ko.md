## 학습하는 함수의 국소 선형화

매개변수 θ를 바꾸면 네트워크 함수도 바뀝니다. 초기값 θ₀ 근처에서는 Jacobian J₀(x)=∂fθ(x)/∂θ를 사용해 함수 변화를 선형화할 수 있습니다. 현재 실습은 같은 초기화에서 실제 비선형 네트워크와 이 선형 모형을 각각 학습합니다.

$$
f_\theta(x)=\frac1{\sqrt m}\sum_{j=1}^m a_j\tanh(w_jx+b_j),\qquad f_{\rm lin}(x)=f_0(x)+J_0(x)\delta.
$$

a,w,b를 모두 학습하므로 Jacobian은 각 유닛당 세 좌표를 갖습니다. 1/√m 배율을 임의로 빼면 다른 kernel과 동역학이 됩니다.

## 경험적 NTK와 잔차 갱신

세 훈련점의 Jacobian 행렬을 J라고 하면 경험적 NTK는 JJᵀ입니다. 제곱 손실 L=∑rᵢ²/(2n)에 대한 parameter GD를 선형화 모형에 적용하면 다음 고정 kernel 동역학을 얻습니다.

$$
K_0=J_0J_0^T,\qquad r_{t+1}=(I-\eta K_0/n)r_t.
$$

이는 고정 Jacobian 모형에서 정확한 유한 단계 항등식입니다. 실제 비선형 모형은 θ가 변하며 J도 변합니다. 한 번의 작은 단계에서 두 함수 차이는 보통 2차 Taylor 항 규모지만 여러 단계 누적이나 큰 학습률에는 별도 분석이 필요합니다.

## 무엇을 실험하는가

입력은 −1,0,1, 표적은 1,0,1입니다. 폭 2와 seed 1의 같은 초기화에서 시작하면 단계 0의 예측 차이와 kernel drift는 정확히 0입니다. 네 단계 뒤에는 실제 Jacobian의 변화로 차이가 생깁니다. 파랑은 비선형 학습, 주황은 고정 초기 Jacobian 학습, 점은 표적입니다.

표에는 매 단계 두 손실, 훈련점 최대 예측 차이, ‖Kₜ−K₀‖F를 기록합니다. 이 값들은 유한한 세 입력에서의 측정입니다. 초기화는 재현 가능한 bounded 의사난수 예시이며 원 논문의 모든 무한 폭 가정을 재현했다고 주장하지 않습니다.

## 무한 폭 결론에는 가정이 필요하다

NTK 극한을 사용하려면 폭을 보내는 방식, 파라미터화, 초기화 분포, 활성화 정칙성, 학습률 또는 gradient flow, 유한 시간 범위와 데이터 조건을 확인해야 합니다. 폭 8의 그림 하나가 kernel의 상수화나 일반화를 증명하지 않습니다. 작은 drift가 모든 입력에서의 작은 함수 오차를 자동 보장하는 것도 아닙니다.

Kernel 관점은 일부 학습 메커니즘을 분석하기 좋지만 feature learning이 크게 일어나는 영역을 같은 고정 kernel로 설명하는 데에는 한계가 있습니다.

## 고유방향의 안정 조건과 영공간 한계를 유도하기

고정한 훈련 Jacobian J₀와 n개 입력에서 ‖r‖²/(2n)의 parameter GD는 δ_{t+1}=δ_t−ηJ₀ᵀr_t/n입니다. J₀를 곱하면 앞의 잔차 갱신을 얻습니다. K₀는 PSD이므로 고유값 μ_j≥0인 기저에서 각 잔차는

$$
r_{t,j}=\left(1-\frac{\eta\mu_j}{n}\right)^t r_{0,j}.
$$

양의 고유성분은 0<η<2n/μ_max에서 수축합니다. 상한 경계에서는 최상위 성분이 크기를 줄이지 않고 진동하고, 그 위에서는 그 성분이 커집니다. 고유값 0인 성분은 그대로입니다. range K₀에서 완벽히 수렴해도 훈련 손실이 0일 필요는 없습니다. 엄격 안정 조건에서 한계 손실은 ‖P_ker K₀ r₀‖²/(2n)입니다.

다음 별도 n=2 예제는 J₀=diag(1,0), r₀=(1,1)입니다. 기존 tanh 네트워크의 학습률 제어 최대 .5를 바꾼 것이 아닙니다.

| η | 4단계 뒤 첫 잔차 | 4단계 뒤 손실 |
|---|---|---|
| 1 | 1/16 | .250977 |
| 4 | 1 | .5 |
| 5 | 81/16 | 6.657227 |

![고정 kernel의 잔차가 안정 범위에서 수축하고 경계에서 진동하며 범위 밖에서는 증가하는 세 곡선. 영공간 잔차는 세 경우 모두 1로 남는다.](/assets/learning/approximation-ntk-stability-ko.svg)

## 무한 폭 정리가 추가하는 내용을 명시하기

원 논문의 고정 깊이 설정은 명시된 1/√width 파라미터화와 독립 Gaussian 초기화를 사용합니다. 학습 정리는 Lipschitz이고 두 번 미분 가능하며 2차 도함수가 유계인 활성화와, 확률적으로 유계인 유한 시간 학습 방향 적분을 가정합니다. 부록의 명제는 은닉 폭을 순차적으로 무한대로 보내 고정 시간 구간 전체의 kernel 수렴을 얻습니다. 해당 결과의 가정이지 모든 NTK 극한의 보편적인 정의는 아닙니다.

유한 실습은 tanh와 bounded seed 초기화를 쓰고 이산 GD 경로를 비교합니다. 위 유한 갱신은 무한 폭 없이 정확하지만, 비선형 모형으로 옮기려면 변하는 Jacobian과 Taylor 나머지를 제어해야 합니다. 매끄러움만으로 전역 frozen feature 정확도나 일반화를 증명하지 못합니다.

**확인 문제:** K₀가 singular일 때 η를 줄이면 영공간 잔차를 없앨 수 있나요? **해설:** 아닙니다. J₀ᵀ가 그 잔차를 0으로 보내므로 고정 특징 GD로는 갱신할 수 없습니다.

```python
from fractions import Fraction as F

# Separate frozen-Jacobian example: J=diag(1,0), K=diag(1,0), n=2.
# Loss is sum r_i^2/(2n); the second residual lies in ker K.
for eta in [F(1),F(4),F(5)]:
    r=[F(1),F(1)]
    for t in range(4):
        r=[r[0]-eta*r[0]/2,r[1]]
    assert r[0]==(1-eta/2)**4 and r[1]==1
    loss=sum(v*v for v in r)/4
    print(str(eta),str(r[0]),round(float(loss),6))
# Positive-eigenvalue stability: 0<eta<4. At 4 the component oscillates;
# above 4 it grows. No step size removes a nonzero nullspace residual.
```

## 확인 문제와 해설

1. η=0이면? **두 모형 모두 변하지 않고 drift도 0입니다.** 단계 수만 늘려 학습을 만들 수 없습니다.
2. m=2의 매개변수 수는? **6개입니다.** a,w,b 두 묶음입니다.
3. 고정 kernel에서 손실이 줄려면 아무 η나 되나요? **아니요.** K₀의 고유값과 n에 따른 안정 범위를 확인해야 합니다. 비선형 모형은 추가 조건도 필요합니다.

추가 읽기: [Jacot, Gabriel and Hongler, Neural Tangent Kernel](https://arxiv.org/abs/1806.07572).
