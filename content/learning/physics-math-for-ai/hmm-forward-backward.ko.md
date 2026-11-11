## 관측 뒤에 숨은 상태

상태 공간 모형은 관측 y와 보이지 않는 상태 x를 구분합니다. HMM은 상태가 이산인 경우입니다. 이 예제의 X₀는 공정하고 Xₜ가 이전 상태를 유지할 확률은 r, 관측이 현재 상태와 일치할 확률은 s입니다. 네 관측은 mask의 낮은 비트부터 읽습니다.

$$
p(x_{0:3},y_{0:3})=p(x_0)\prod_{t=1}^3p(x_t\mid x_{t-1})\prod_{t=0}^3p(y_t\mid x_t).
$$

## Forward는 과거, backward는 미래

αₜ(x)=p(y₀:ₜ,Xₜ=x)는 transition으로 이전 α를 합하고 emission을 곱합니다. βₜ(x)=p(yₜ₊₁:₃|Xₜ=x)는 미래를 거꾸로 요약합니다. 전체 evidence는 Σxα₃(x)입니다.

$$
\alpha_t(x)=p(y_t\mid x)\sum_u p(x\mid u)\alpha_{t-1}(u),\qquad
\beta_t(x)=\sum_v p(v\mid x)p(y_{t+1}\mid v)\beta_{t+1}(v).
$$

Filtering은 α만 정규화한 p(Xₜ|y₀:ₜ), smoothing은 αβ를 정규화한 p(Xₜ|전체 y)입니다. 기본 관측 0,0,1,1은 미래의 1들이 초기 상태 판단도 바꿉니다. 마지막 시점에서는 미래가 없으므로 filtering과 smoothing이 같습니다.

## 로그 영역과 불가능한 증거

작은 확률의 곱이 0으로 사라지는 것을 피하려고 log-sum-exp로 계산합니다. r=s=1인데 1,0,1,0을 관측하면 상태는 바뀔 수 없고 관측은 완벽하므로 evidence가 수학적으로 0입니다. 이때 log evidence는 −∞, 조건부 smoothing은 —입니다.

반면 r=1,s=10⁻³²⁰인 같은 관측은 매우 작은 양의 확률을 가집니다. exp(log evidence)는 숫자 범위 아래에서 0이 되어도 로그는 유한하고 조건부 확률은 계산할 수 있습니다. 화면은 가능 여부와 수치 범위 경고를 분리합니다.

## 연속 상태 공간과 Kalman의 작은 예제

연속 선형 Gaussian 모형은 Xₜ=FXₜ₋₁+wₜ, Yₜ=HXₜ+vₜ로 같은 구조를 가집니다. 스칼라 prior N(m,P), 관측 잡음 분산 R이면 K=PH/(H²P+R), 업데이트 평균 m+K(y−Hm), 분산 (1−KH)P입니다. m=0,P=1,H=1,R=1,y=1이면 posterior N(.5,.5)입니다. 이는 이산 표 합 대신 Gaussian 모멘트를 전달하는 예입니다. 일반 비선형 상태 공간에서 같은 닫힌식이 자동 성립하지 않습니다.

## 어느 시점에서 잘라도 같은 evidence

유한 상태·정규화한 전이와 관측 표, 양의 evidence를 가정합니다. Xₜ를 고정하면 Markov 인수분해상 과거와 미래의 관측 가중치가 αₜ와 βₜ로 분리됩니다. 따라서 어느 t에서나 Z=Σᵢαₜ(i)βₜ(i)입니다. 이는 forward/backward 구현을 검사하는 불변식입니다. 같은 논증을 인접 두 상태에 적용하면 edge posterior ξ를 얻습니다.

$$
\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}Z,
\qquad
\xi_t(i,j)=\frac{\alpha_t(i)A_{ij}B_j(y_{t+1})\beta_{t+1}(j)}Z.
$$

ξ의 j 합은 γₜ(i), i 합은 γₜ₊₁(j)와 같아야 합니다. γ는 상태의 예상 방문 횟수, ξ는 예상 전이 횟수를 만들므로 잠재 경로를 하나로 확정하지 않고 학습할 수 있습니다.

## 한 번의 EM을 정확분수로 확인하기

현재 이진 대칭 모형에서 prior는 공정하게 고정하고 stay r와 emission accuracy s만 추정합시다. 길이 T=4의 한 관측열에서 갱신은 다음과 같습니다. 분모 3은 전이 수, 4는 관측 수입니다.

$$
r_{\mathrm{new}}=\frac1{T-1}\sum_{t=0}^{T-2}\sum_i\xi_t(i,i),
\qquad
s_{\mathrm{new}}=\frac1T\sum_{t=0}^{T-1}\gamma_t(y_t).
$$

이전 posterior q(x)=pθold(x|y)를 고정하면 log Znew−log Zold=log E_q[pθnew(x,y)/pθold(x,y)]≥E_q log[pθnew/pθold]입니다. 유한 상태·양의 모수에서 Jensen이 적용되고, 위 갱신은 tied Bernoulli의 기대 로그우도를 최대화하므로 오른쪽은 음수가 아닙니다. 따라서 정확한 EM은 학습열 evidence를 감소시키지 않습니다. 이 사실은 전역 최적점이나 새 자료의 성능을 보장하지 않습니다.

$$
Q(\theta\mid\theta_{\mathrm{old}})
=\sum_x q(x)\log p_\theta(x,y),\qquad
\log Z_{\mathrm{new}}-\log Z_{\mathrm{old}}
\ge Q(\theta_{\mathrm{new}}\mid\theta_{\mathrm{old}})
-Q(\theta_{\mathrm{old}}\mid\theta_{\mathrm{old}})\ge0.
$$

| 단계, 관측 0,0,1,1 | r | s | evidence |
|---|---|---|---|
| 시작 | .800000 | .900000 | .058276 |
| 한 번 EM | .674656 | .909637 | .068564 |

코드는 γ,ξ의 주변 일치와 16개 잠재 경로의 합을 모두 분수로 대조합니다. 화면은 주어진 r,s에서의 추론을 계산하며, EM 갱신은 별도 실행 예제입니다. 일반 HMM에서 전이 행의 예상 방문 수가 0이면 그 행의 갱신은 식별되지 않아 별도 처리해야 합니다. 이 모형의 tied 전체 분모에는 그런 영 방문 행 나눗셈이 없습니다.

앞의 Kalman 식도 조건이 있습니다. 독립 Gaussian prior·관측 잡음과 P>0,R>0에서, 사후 로그밀도의 이차항 정밀도는 P⁻¹+H²/R이고 선형항은 m/P+Hy/R입니다. 완전제곱하면 제시한 평균과 분산을 얻습니다. 비선형 관측에서는 사후가 Gaussian으로 닫히지 않을 수 있습니다.

**직접 풀기:** Viterbi 경로의 전이 횟수로 ξ를 바꿔도 같은 EM 증명인가요? **해설:** 아닙니다. 전체 이전 posterior에 대한 기대를 한 경로로 바꾸면 위 Jensen 하한의 정확한 최대화가 아닙니다.

추가 근거: [Rabiner (1989), §III의 forward/backward·reestimation](https://www.fceia.unr.edu.ar/prodivoz/Rabiner_1989.pdf). 여기의 tied 두 모수 갱신과 분수 예제는 직접 유도했습니다.

```python
from fractions import Fraction as F
from itertools import product

y = [0,0,1,1]
def infer(r,s):
    transition = lambda i,j: r if i==j else 1-r
    emission = lambda t,i: s if i==y[t] else 1-s
    alpha = [[F(1,2)*emission(0,i) for i in [0,1]]]
    for t in range(1,4):
        alpha.append([emission(t,j)*sum(alpha[t-1][i]*transition(i,j)
                      for i in [0,1]) for j in [0,1]])
    beta = [[F(1),F(1)] for _ in range(4)]
    for t in range(2,-1,-1):
        beta[t] = [sum(transition(i,j)*emission(t+1,j)*beta[t+1][j]
                       for j in [0,1]) for i in [0,1]]
    Z = sum(alpha[-1])
    gamma = [[alpha[t][i]*beta[t][i]/Z for i in [0,1]] for t in range(4)]
    xi = [[[alpha[t][i]*transition(i,j)*emission(t+1,j)*beta[t+1][j]/Z
            for j in [0,1]] for i in [0,1]] for t in range(3)]
    for t in range(4):
        assert sum(alpha[t][i]*beta[t][i] for i in [0,1]) == Z
    for t in range(3):
        for i in [0,1]:
            assert sum(xi[t][i]) == gamma[t][i]
            assert sum(xi[t][j][i] for j in [0,1]) == gamma[t+1][i]
    # Independent complete-path enumeration checks both node and edge marginals.
    paths = []
    for x in product([0,1],repeat=4):
        w = F(1,2)
        for t in range(4):
            w *= emission(t,x[t])
            if t: w *= transition(x[t-1],x[t])
        paths.append((x,w))
    assert sum(w for x,w in paths) == Z
    for t in range(4):
        assert sum(w*x[t] for x,w in paths)/Z == gamma[t][1]
    for t in range(3):
        for i,j in product([0,1],repeat=2):
            assert sum(w for x,w in paths if x[t]==i and x[t+1]==j)/Z == xi[t][i][j]
    return Z,gamma,xi

r,s = F(4,5),F(9,10)
Z,gamma,xi = infer(r,s)
r_new = sum(xi[t][i][i] for t in range(3) for i in [0,1])/3
s_new = sum(gamma[t][y[t]] for t in range(4))/4
new_Z,_,_ = infer(r_new,s_new)
assert new_Z >= Z
for label,rr,ss,zz in [('old',r,s,Z),('one EM step',r_new,s_new,new_Z)]:
    print(label, 'r,s,evidence:', *(round(float(v),6) for v in [rr,ss,zz]))
```


## 확인 문제와 해설

1. 미래 관측까지 쓰는 것은? **Smoothing입니다.**
2. 관측 정확도 .5이면 정보는? **관측이 상태를 구분하지 못해 주변이 .5입니다.**
3. 노드별 주변 최빈값은 joint MAP 경로인가요? **일반적으로 아닙니다. Viterbi 같은 경로 목적과 구분합니다.**

추가 읽기: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
