## Kernel 회귀의 목적함수를 먼저 고정하기

실습은 입력 −1,0,1, 관측값 1,c,1을 사용합니다. 합성 실험의 참조 함수는 x²이지만 학습기는 kernel과 세 관측값만 사용합니다. RBF kernel k(x,z)=exp(−(x−z)²/(2ℓ²))에서 ℓ은 length scale이고 amplitude는 1로 고정합니다.

$$
\min_{f\in H}\sum_i(f(x_i)-y_i)^2+\lambda\|f\|_H^2.
$$

합 손실 관례입니다. 앞에 1/n을 붙이는 평균 손실이면 같은 이름의 λ를 써도 선형계에 nλ가 나타날 수 있습니다. 논문과 코드의 배율을 먼저 맞추세요.

## 유한 Gram 시스템과 예측

Representer 정리로 f(x)=∑αᵢk(xᵢ,x)로 두고 다음을 풉니다.

$$
(K+\lambda I)\alpha=y,\qquad f(x)=k_x^T\alpha.
$$

코드는 역행렬을 직접 만들지 않고 Cholesky 분해와 삼각 해법을 사용합니다. 표시한 λ 외에 숨은 jitter를 더하지 않습니다. 현재 입력은 서로 다르고 ℓ이 유한 양수여서 λ=0에서도 RBF Gram은 양정입니다. 일반 kernel이나 중복 입력에서는 이 조건을 다시 검토해야 합니다.

## 수치 예제를 어떻게 확인할까

ℓ=1에서 K의 인접 상관은 exp(−1/2), 양끝 상관은 exp(−2)입니다. λ=0, c=0이면 세 관측을 수치 정밀도 안에서 정확히 통과하지만 x² 전체를 복원하는 것은 아닙니다. λ=.1이면 훈련점도 보통 완전히 통과하지 않고 노름과 잔차를 교환합니다. 표의 α와 예측값, 선형계 잔차를 함께 확인하세요.

$$
\|f\|_H^2=\alpha^TK\alpha,\qquad S=K(K+\lambda I)^{-1}.
$$

S는 관측값에서 훈련 예측값으로 가는 smoother입니다. trace(S)를 효과 자유도로 표시합니다. λ=0이면 3, 양의 λ이면 0과 3 사이입니다. 파랑은 합성 참조, 주황은 회귀 함수, 점은 관측값입니다.

## Length scale과 정규화는 다른 역할

작은 ℓ은 가까운 점에서만 비슷한 영향을 주고 큰 ℓ은 더 넓게 연결합니다. λ는 선택한 RKHS에서 큰 노름을 억제합니다. 두 값을 함께 바꿀 때의 모형 선택은 검증 자료나 명시한 확률 모형에 근거해야 합니다. 세 점에 맞았다는 이유만으로 새 분포의 성능이나 extrapolation을 보장할 수 없습니다.

큰 데이터에서는 Gram 저장 O(n²), 직접 분해 O(n³)가 부담이 됩니다. Random feature나 Nyström 근사는 계산을 줄이지만 그 근사 오차와 통계적 오차를 구분해야 합니다.

## 고유방향 하나씩 정규화를 읽기

실수 PSD 훈련 Gram K=U diag(s_j)Uᵀ와 λ>0에서 훈련 예측은 Sy이며 축소율은 q_j=s_j/(s_j+λ)입니다. 양의 고유방향은 줄고 영공간 성분은 버립니다. 효과 자유도 ∑q_j는 label 변화에 대한 smoother의 반응이며 통계적으로 독립인 관측 개수가 아닙니다.

입력을 고정하고 y=f+ε, Eε=0, Cov(ε)=σ²I라 둡니다. 기대값은 이 같은 입력에서 새로 뽑는 잡음에 대한 평균입니다. 고유기저 좌표 f_j=(Uᵀf)_j에서

$$
\frac1n E\|Sy-f\|^2
=\frac1n\sum_j(1-q_j)^2f_j^2
+\frac{\sigma^2}{n}\sum_j q_j^2.
$$

Sy−f=(S−I)f+Sε로 나누면 교차항의 기대값이 0이고 E‖Sε‖²=σ² tr(S²)이므로 증명됩니다. 잡음 없는 훈련 위치 함수값의 예측 오차이며 새로운 입력의 모집단 위험은 아닙니다. 같은 위치의 새로운 잡음 label을 예측하면 입력당 σ²가 더해집니다. λ=0이고 Gram이 singular이면 정의되지 않은 역행렬 대신 사영 관례를 정해야 합니다.

## 모의실험 가정 없이 교환 관계를 계산하기

K=[[1,1/2],[1/2,1]], f=(1,1), σ²=1/4를 택합니다. 고유값은 3/2와 1/2입니다. 코드는 독립 잡음 (±1/2,±1/2) 네 개를 평균합니다. 공분산이 정확히 σ²I이므로 Gaussian 잡음 가정 없이 항등식이 성립합니다.

| λ | 입력당 편향 제곱 | 입력당 분산 | 기대 오차 |
|---|---|---|---|
| 0 | 0 | .25 | .25 |
| 1/4 | .020408 | .147392 | .167800 |
| 1 | .160000 | .058889 | .218889 |
| 4 | .528926 | .010841 | .539766 |

별도 두 입력 모형의 수치이며 기존 RBF 제어를 바꾸지 않습니다. λ를 키우면 이 분산은 줄지만 두 항의 합이 단조로 줄지는 않습니다. 작은 해법 잔차로 좋은 λ나 이동한 입력 분포에서의 성능을 증명하지 못합니다.

**확인 문제:** 완벽히 최적화해도 ker K의 신호 성분에 편향이 남는 이유는? **해설:** Kernel 훈련 예측은 range K 안에 있어 선택한 kernel로 그 성분을 나타낼 수 없기 때문입니다.

```python
from fractions import Fraction as F

sigma_squared=F(1,4)
# K=[[1,1/2],[1/2,1]], f=(1,1): squared signal is 2 in the top eigendirection.
eigenvalues=[F(3,2),F(1,2)]
for lam in [F(0),F(1,4),F(1),F(4)]:
    shrink=[s/(s+lam) for s in eigenvalues]
    bias=2*(1-shrink[0])**2/2
    variance=sigma_squared*sum(q*q for q in shrink)/2
    # Independent enumeration of four zero-mean noise vectors of covariance sigma^2 I.
    risk=F(0)
    for e1 in [F(-1,2),F(1,2)]:
        for e2 in [F(-1,2),F(1,2)]:
            # Project onto unnormalised (1,1),(1,-1), then shrink and reconstruct.
            plus=shrink[0]*(1+(e1+e2)/2)
            minus=shrink[1]*(e1-e2)/2
            pred=[plus+minus,plus-minus]
            risk+=sum((v-1)**2 for v in pred)/8
    assert risk==bias+variance
    print(str(lam),round(float(bias),6),round(float(variance),6),round(float(risk),6))
```

## 확인 문제와 해설

1. λ를 키우면 언제나 참 함수에 가까워지나요? **아니요.** 분산을 줄이면서 편향을 늘릴 수 있고 실제 자료에 따라 달라집니다.
2. α가 음수이면 kernel이 PSD가 아닌가요? **아니요.** 계수의 부호와 kernel의 이차형식 조건은 다른 문제입니다.
3. 선형계 잔차가 작으면 모집단 위험도 작은가요? **보장하지 않습니다.** 수치 해법의 정확성과 통계 성능은 별도입니다.

추가 읽기: [Rasmussen and Williams, Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
