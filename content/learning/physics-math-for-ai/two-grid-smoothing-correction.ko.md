## 미세 격자에서 느린 오차가 남는 이유

반복 선형 해법에서 오차의 크기만이 아니라 공간 모양이 수렴 속도를 좌우합니다. 이웃을 평균내는 relaxation은 급격히 교대하는 성분을 빠르게 줄이지만 천천히 변하는 성분에는 거의 손대지 못할 수 있습니다. 다중격자는 미세 격자에서 매끄러운 오차를 더 거친 격자에서 저렴하게 표현해 보정합니다.

현재 정확해는 0이며 초기 오차 e(x)=sin(mπx)를 둡니다. 따라서 별도의 정확한 PDE solver를 참값으로 삼지 않아도 오차 자체를 알 수 있습니다. 파랑은 실제 cycle 뒤의 오차이고 주황은 초기 오차입니다. 표에는 h가중 L2 오차와 잔차를 따로 기록합니다.

## Smoothing과 잔차 방정식

Dirichlet Poisson의 A는 대각 2/h², 이웃 -1/h²입니다. Weighted Jacobi는 다음과 같습니다.

$$
e\leftarrow e-\omega D^{-1}Ae,\qquad r=-Ae.
$$

저주파 sine의 고유값은 작아 한 번의 갱신이 조금만 바뀝니다. 고주파에서는 더 큰 보정이 생기며 weight를 적절히 골라야 합니다. 가장 큰 weight가 항상 가장 좋은 smoothing을 의미하지 않습니다. 여기서는 전후 smoothing 횟수를 똑같이 설정합니다.

## 실제 Coarse Operator 조립

P는 coarse 절점 사이를 선형 보간하고 R=Pᵀ/2는 full weighting입니다. Coarse 행렬을 임의의 다른 stencil로 대체하지 않고 현재 A와 transfer로 계산합니다.

$$
A_H=RA_hP,\qquad A_H e_H=Rr,\qquad
e\leftarrow e+Pe_H.
$$

Coarse 식은 pivoting을 사용하는 작은 직접 해법으로 풉니다. 이어 post-smoothing을 적용하며 cycle 수만큼 실제 반복합니다. Restriction과 prolongation의 차원, 경계 0의 처리, 잔차 부호가 모두 중요합니다. 잘못된 부호는 오차를 제거하는 대신 더합니다.

## Coarse 보정은 에너지 사영이다

A가 대칭 양의 정부호이고 P가 full column rank이면 R=Pᵀ/2의 상수 인자는 정확한 coarse 보정에서 소거된다. 오차 연산자는 C=I−P(PᵀAP)⁻¹PᵀA다. 직접 곱하면 C²=C,CᵀA=AC,PᵀAC=0이므로

$$
\|Ce\|_A^2=\|e\|_A^2-\|e-Ce\|_A^2.
$$

Euclidean 거리가 아니라 A-에너지에서 가장 가까운 coarse 성분을 제거한다. 정확한 coarse 보정의 비확장은 증명하지만 적절한 smoothing·근사 논증 없이 균일 two-grid 수렴까지 보이지는 않는다.

Weighted Jacobi의 고주파 대역 θ∈[π/2,π]에서 최악 인자는 max(|1−ω|,|1−2ω|)다. 양 끝 오차를 균형 맞추면 minimax ω=2/3,최악 인자 1/3이다. 균일 1D Poisson의 이 대역 상수이며 모든 연산자의 최적 weight가 아니다.

![공간 주파수별 weighted Jacobi 배율과 고주파 대역에서 minimax weight의 의미를 비교하는 그림.](/assets/learning/scientific-smoothing-ko.svg)

| ω | 지정 고주파 대역의 최악 배율 |
|---|---:|
| 1/2 | 1/2 |
| 2/3 | 1/3 |
| 1 | 1 |

```python
from fractions import Fraction as F
A=[[F(2),F(-1),F(0)],[F(-1),F(2),F(-1)],[F(0),F(-1),F(2)]]
p=[F(1,2),F(1),F(1,2)];e=[F(1),F(-1),F(2)]
mv=lambda v:[sum(a*b for a,b in zip(row,v)) for row in A]
dot=lambda a,b:sum(x*y for x,y in zip(a,b))
coefficient=dot(p,mv(e))/dot(p,mv(p))
removed=[coefficient*x for x in p]
corrected=[x-y for x,y in zip(e,removed)]
assert dot(p,mv(corrected))==0
assert dot(e,mv(e))==dot(corrected,mv(corrected))+dot(removed,mv(removed))
for weight in (F(1,2),F(2,3),F(1)):
    print(weight,max(abs(1-weight),abs(1-2*weight)))
```

유리수 사영으로 세 fine 미지수의 직교성과 피타고라스를 확인한다. 실제 solver는 Galerkin coarse 행렬을 조립하고 전후 smoothing을 수행한다. Neural pooling도 연산자·norm·보정 식을 명시해야 같은 의미의 그림으로 해석할 수 있다.

## 손계산과 직접 풀기

m=1인 매끄러운 sine와 m=7인 진동 sine를 같은 fine 격자에서 비교해 보세요. Jacobi의 mode multiplier는 1-ω(1-cos(mπh))입니다. 작은 mπh에서는 거의 1이라 변화가 느리고, 선택된 큰 주파수에서는 절댓값이 작아질 수 있습니다. Coarse correction은 smoothing이 남긴 매끄러운 부분을 주로 줄입니다.

직접 풀기: fine 오차가 정확히 Pz 형태이고 smoothing이 없으면 한 coarse correction 뒤 어떻게 되나요? 풀이: r=-APz, e_H=(RAP)^{-1}R(-APz)=-z이므로 e+Pe_H=0입니다. 반대로 coarse 공간에 없는 모든 성분까지 한 번에 사라진다는 결론은 낼 수 없습니다.

## 다중스케일과 AI 연결

Mesh 기반 network의 pooling·upsampling이 multigrid와 시각적으로 비슷하더라도 같은 잔차 방정식을 푼다는 뜻은 아닙니다. 어떤 operator와 norm을 보정하는지 확인해야 합니다. 물리 homogenization은 또 다른 문제로, 작은 구조의 유효 계수를 얻으려면 scale separation, cell problem 등의 가정이 필요합니다. 여기서는 실제 두 수준 선형 보정까지만 계산하며 물리적 coarse-graining 전체를 구현했다고 주장하지 않습니다.

검증에서는 Galerkin 항등식, coarse 공간 제거, 여러 격자의 오차 감소와 실제 작업량을 함께 봅니다. 한 격자의 작은 잔차만으로 격자 독립 수렴이나 O(N) 복잡도를 증명하지 않습니다.

참고: [MIT 수치 PDE 강의](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
