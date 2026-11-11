## 해의 위치뿐 아니라 구조를 계산하기

진동자는 q′=p, p′=-ω²q이며 q(0)=1, p(0)=0으로 시작합니다. 정확해 q=cos(ωt), p=-ω sin(ωt)는 위상 평면에서 닫힌 타원을 그립니다. 위치 오차가 작아 보이는 짧은 관측만으로는 장시간 에너지와 기하 구조를 알기 어렵습니다. 이 단원은 실제 갱신점, 물리 에너지, 다른 보존 이차량을 나누어 표시합니다.

위 그림은 무차원 q,p에 같은 축척을 사용합니다. 따라서 그림이 정사각형으로 확대되어도 원래 두 좌표의 길이 비율을 바꾸지 않습니다. 아래 파랑은 실제 계산 에너지 H=(p²+ω²q²)/2이며 주황은 초기 에너지입니다. 연결선은 유한한 점의 순서이며 연속 Hamiltonian 궤적 자체는 아닙니다.

## 세 갱신식의 차이

Forward Euler는 q와 p를 모두 이전 값에서 갱신합니다. Kick-then-drift symplectic Euler는 p부터 바꾼 뒤 새 p로 q를 움직입니다. Verlet는 반 kick, 한 drift, 반 kick을 적용합니다.

$$
p_{1/2}=p_n-\tfrac h2\omega^2q_n,\qquad
q_{n+1}=q_n+h p_{1/2},\qquad
p_{n+1}=p_{1/2}-\tfrac h2\omega^2q_{n+1}.
$$

선형 2차원에서 forward Euler의 determinant는 1+(ωh)²이고 나머지 두 방법은 1입니다. 여기서는 determinant 1이 symplectic 조건과 연결되지만 고차원에서는 determinant만 확인해서 충분하지 않습니다. 일반 조건은 MᵀJM=J입니다.

## 면적·물리 에너지·수정 에너지

Symplectic이 정확한 H를 보존한다는 뜻은 아닙니다. 이 선형 진동자에서 symplectic Euler는 H-hω²qp/2를, Verlet는 다음 이차량을 정확한 산술에서 보존합니다.

$$
H_* = \tfrac12\{p^2+\omega^2[1-(\omega h)^2/4]q^2\}.
$$

ωh<2에서 이 이차형은 양의 정부호입니다. 경계 또는 그 이상에서는 면적을 보존해도 해가 유계라는 결론을 낼 수 없습니다. 실험에서 step 수를 줄여 이런 차이를 보세요. 유한 정밀도의 미세한 수정 에너지 drift와 물리 H의 진동도 구별해야 합니다.

## 에너지가 유계여도 위상 오차는 자란다

고정 step Verlet의 z=ωh<2에서 고유값은 exp(±iφ),cosφ=1−z²/2다. 따라서 위상률은 ωₕ=2asin(ωh/2)/h이며 전개하면 ωₕ−ω=ω³h²/24+O(h⁴)다. 물리 시간 T 뒤 위상 전진 오차는 T(ωₕ−ω)다. 수정 에너지가 유계여도 장시간 위상 오차가 균일하게 작다는 결론은 나오지 않는다.

![안정한 세 고정 보폭의 해석적 Verlet 위상률 편향을 동일한 물리 시간에 누적한 그림.](/assets/learning/scientific-phase-ko.svg)

| ω=1의 h | 위상률 초과량 근삿값 |
|---|---:|
| 1/2 | 0.0107210 |
| 1/4 | 0.00262265 |
| 1/8 | 0.000652189 |

```python
from fractions import Fraction as F
from math import asin
def matrix(h):
    return [[1-h*h/2,h],[-h*(1-h*h/4),1-h*h/2]]
def mm(A,B):return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
M=mm(matrix(F(19,10)),matrix(F(1,2)))
trace=M[0][0]+M[1][1]
assert trace==F(-299,125) and M[0][0]*M[1][1]-M[0][1]*M[1][0]==1
print('alternating trace:',trace)
for h in (.5,.25,.125):
    bias=2*asin(h/2)/h-1
    assert bias>0
    print(h,bias,'leading approximation:',h*h/24)
```

가변 보폭은 또 다른 반례를 준다. 각각의 Verlet 사상이 symplectic이면 곱도 symplectic이다. 하지만 고정-h 수정 이차식은 h에 따라 바뀐다. ω=1에서 h₁=1/2,h₂=19/10을 교대하면 두-step 사상은 determinant 1,trace −299/125<−2다. 각 보폭은 따로 안정해도 교대 곱의 반복은 불안정하다. 코드로 유리수 trace를 정확히 검산한다. Symplectic만으로 적응 step 안정성을 보장하지 못하며 live 실습은 고정 h를 유지한다.

## 손계산과 직접 풀기

ω=1, h=0.2이면 forward Euler 첫 상태는 (1,-0.2)이고 H=0.52입니다. Symplectic Euler는 p=-0.2를 먼저 만든 뒤 q=0.96이 되어 H=0.4808입니다. 교차항 -hqp/2=0.0192를 더하면 수정 이차량은 정확히 0.5입니다.

직접 풀기: 같은 조건의 Verlet 첫 상태와 에너지를 구하세요. 풀이: p_half=-0.1, q_next=0.98, p_next=-0.198입니다. H=(0.198²+0.98²)/2=0.499802입니다. 원래 H는 0.5와 다르지만 H_*는 초기의 0.495와 같습니다. 이 결과는 같은 h를 유지하는 현재 고정 step 모형의 성질입니다.

## AI 연구에서 확인할 구조

Hamiltonian neural network나 HMC 제안은 적분 오차와 구조 보존에 민감할 수 있습니다. 상태 관측 loss만 줄이는 것과 올바른 장시간 거동을 갖는 것은 서로 다른 검증 과제입니다. 단계 순서, step 크기, 에너지 drift, 면적 또는 symplectic 잔차를 기록하세요. HMC에서는 별도의 Metropolis 보정과 확률적 정상성 조건도 필요합니다. 에너지 그림 한 장이 샘플러의 정확성을 증명하지는 않습니다.

참고: [MIT 수치 방법 강의](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
