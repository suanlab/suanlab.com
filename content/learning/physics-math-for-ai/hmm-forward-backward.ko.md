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

## 확인 문제와 해설

1. 미래 관측까지 쓰는 것은? **Smoothing입니다.**
2. 관측 정확도 .5이면 정보는? **관측이 상태를 구분하지 못해 주변이 .5입니다.**
3. 노드별 주변 최빈값은 joint MAP 경로인가요? **일반적으로 아닙니다. Viterbi 같은 경로 목적과 구분합니다.**

추가 읽기: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
