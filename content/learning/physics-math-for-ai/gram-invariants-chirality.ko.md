## 불변 특징은 정보를 의도적으로 버릴 수 있습니다

R³의 ordered 벡터 v₀,v₁,v₂의 Gram 행렬은 모든 쌍별 내적을 담습니다. 직교 g는 gᵀg=I라 이를 보존하고 회전·반사 모두 직교입니다. 따라서 proper 회전만 동등하게 취급해야 하는 문제에서 Gram은 일반적으로 full-rank 거울상을 구별하지 못합니다.

v₀=ex,v₁=ey,v₂=(0.3,0.4,h)에 z 회전·y 회전·선택적 z 반사를 적용합니다. 연산과 내적은 모두 실제 3차원이며 위 그림만 비등각 투영입니다. 아래 그림은 독립 Gram 성분 여섯 개를 비교합니다.

$$
V=[v_0\ v_1\ v_2],\quad G=V^TV,\qquad
(gV)^T(gV)=G.
$$

## Gram이 결정하는 범위

같은 Gram을 가진 두 ordered 벡터 집합 사이의 생성자 대응은 모든 선형 결합의 norm을 보존하므로 span 사이의 잘 정의된 등거리 사상입니다. 보완 공간의 정규직교 기저를 확장하면 R³의 직교 변환을 얻습니다. 따라서 전체 Gram은 이 ordered 구성의 O3 궤도를 정합니다.

길이 세 개만 쓰면 쌍별 각도도 잃습니다. 임의의 학습 불변 특징이 O3에 대해서도 완전한 것은 아닙니다. 여기의 완전성 논증은 전체 Gram과 명시한 ordered 작용을 사용합니다.

## 부호 있는 부피와 방향성

χ=detV는 방향 있는 부피이며 다음과 같이 변합니다.

$$
\chi'=\det(g)\chi,\qquad \det G=(\det V)^2=\chi^2.
$$

h=1이면 원래 부피는 1입니다. proper 회전은 유지하고 반사에서는 −1이지만 Gram은 같습니다. full-rank triple에서는 Gram에 부피 부호를 더해야 O3 궤도 안의 두 SO3 방향류를 구별합니다.

h=0이면 span은 평면이고 부피는 0입니다. 비영 부피의 handedness 부호가 없어 화면에서 미정의로 남깁니다. 퇴화 상태에 양의 방향을 억지로 부여하지 않습니다.

## 수치 분해의 차이

서로 상쇄하는 Gram 성분으로 determinant를 전개하면 정확한 값 0에도 작은 음수 roundoff가 나올 수 있습니다. 수치 표는 그 raw 전개값과 안정적인 detG=χ²를 구별합니다. 정확한 Gram이 indefinite하거나 제곱 부피가 음수라는 뜻이 아닙니다. 오차를 숨기려고 raw 값을 몰래 0으로 clamp하지 않습니다. 변환 부피도 알려진 직교 determinant 법칙으로 안정적으로 구하고 전개 부동소수점 값은 따로 보여, 퇴화 근처의 가짜 방향 판정을 피합니다.

## Rank가 반사 모호성의 존재를 결정합니다

3차원을 모두 생성하는 triple에서는 V를 W로 보내는 직교 사상이 유일하므로 그 행렬식이 같은 SO3 궤도인지 결정합니다. span 차원이 3보다 작으면 그 span을 점별로 고정하면서 보완 방향을 반사할 수 있습니다. 이 반사를 합성하면 입력 벡터의 상은 유지한 채 행렬식 부호를 뒤집습니다. 따라서 rank가 부족한 ordered 구성에서는 전체 Gram만 같아도 SO3 궤도가 같습니다. 퇴화점에는 비영 부피의 두 chirality 류가 존재하지 않습니다.

Handedness에는 조건수 가정도 필요합니다. 가역 V와 아래 조건을 만족하는 섭동 E에서는 V+tE 전체가 가역입니다. 영벡터가 생긴다고 가정하면 최소 특이값 하한과 모순되기 때문입니다. 따라서 경로를 따라 determinant 부호가 바뀌지 않습니다.

$$
\|E\|_2<\sigma_{\min}(V)\quad\Longrightarrow\quad
\operatorname{sign}\det(V+E)=\operatorname{sign}\det V.
$$

V=diag(1,1,h)의 거울상은 h를 −h로 바꾸므로 비용은 2|h|이고 Gram은 정확히 같습니다. h가 0으로 가면 반대 handedness도 좌표상 임의로 가까워집니다. 따라서 신뢰할 방향 분류에는 수학적으로 비영인 determinant뿐 아니라 rank margin과 잡음 모형도 필요합니다. 극도로 작은 h의 제곱 부피는 binary64에서 underflow할 수 있으므로 반올림된 제곱 0만으로 rank를 판정할 수 없습니다.

| h | 최소 특이값 | 거울상 변위 | Gram 대각 |
|---:|---:|---:|---|
| 1 | 1 | 2 | 1,1,1 |
| 0.1 | 0.1 | 0.2 | 1,1,0.01 |
| 0.001 | 0.001 | 0.002 | 1,1,0.000001 |

정확 유리수 예제는 Gram determinant 소거 오차와 섞지 않고 조건수 효과만 분리합니다.

## Python으로 계산 재현하기

```python
from fractions import Fraction as F
for h in [F(1),F(1,10),F(1,1000)]:
    gram=[F(1),F(1),h*h]
    displacement=2*h
    for perturb in [h/2,-h/2]: assert h+perturb > 0
    assert (-h)*(-h)==h*h
    print('h:',h,'sigma_min:',h,'mirror cost:',displacement,'Gram:',gram)
# A reflection of the unused z direction fixes every vector in the xy plane.
for v in [(1,0,0),(0,1,0),(3,4,0)]: assert (v[0],v[1],-v[2])==v
```

## 직접 풀기와 풀이

h=1에서 proper·반사 경우를 비교하면 여섯 Gram 성분은 같고 부피만 +1에서 −1로 바뀝니다. h=0이면 rank 2, 안정적 Gram determinant 0, 방향 부호 미정의입니다. 회전만 바꿔 같은 불변량이 유지되는지 확인하세요.

## 연구와의 연결

분자·점 집합의 불변 특징은 과제의 동치류에 맞아야 합니다. 반사 불변성을 강제하면 화학·기하에 유용한 chirality를 지울 수 있고, 반사가 무관한 과제에서는 불필요한 방향 구별을 넣을 수도 있습니다. O3와 SO3, 순서·순열, 병진 중심화, 퇴화 조건을 명시한 뒤 형상 정보를 완전히 담았는지 논증하세요.

추가 읽기: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
