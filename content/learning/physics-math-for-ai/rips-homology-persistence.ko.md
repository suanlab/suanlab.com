## 이웃 Graph를 넘어 복합체를 만듭니다

유한 점 집합 자체가 연속 표면은 아닙니다. 규모별 구조를 보려면 filtration을 선택해야 합니다. 여기서는 모든 꼭짓점 쌍의 거리가 ε 이하일 때 Vietoris–Rips simplex를 넣습니다. filtration 값은 지름입니다. ε는 간선 거리 임계값이며 반지름 관례의 추가 2배를 사용하지 않습니다.

직사각형 꼭짓점 네 개는 점·간선·삼각형·마지막 사면체를 만듭니다. 점 위치가 평면에 그려져도 추상 복합체의 차원이 2로 제한되지 않습니다. 구조 그림의 음영은 포함된 삼각형이고 사면체는 평면 투영이 모호하므로 개수를 따로 읽습니다.

## 경계로 기하를 선형대수로 바꾸기

GF(2)의 chain은 simplex들의 부분집합이며 두 번 더하면 지워집니다. 경계사상은 codimension-one face를 모두 더합니다. face의 face는 두 번씩 나타나 ∂²=0이고 모든 boundary는 cycle입니다. 따라서 homology를 다음과 같이 정의합니다.

$$
H_k=\ker\partial_k/\operatorname{im}\partial_{k+1},\qquad
\beta_k=n_k-\operatorname{rank}\partial_k-\operatorname{rank}\partial_{k+1}.
$$

구현은 실제 경계행렬을 만들고 binary 소거를 수행합니다. 그림의 모양이나 미리 적은 Betti 목록에서 값을 추측하지 않습니다.

## 단위 정사각형의 임계값 따라가기

ε<1이면 네 고립점이라 β₀=4, β₁=0입니다. ε=1에서 네 변이 들어와 한 성분과 하나의 미충전 cycle이 됩니다. 간선 경계 rank가 3이므로 β₁=4−3=1입니다. ε=√2에서는 대각선·삼각형 4개·사면체가 들어와 전체 simplex가 되고 β₁=β₂=0입니다.

간선만 남기면 완전 graph에도 graph cycle이 남습니다. 삼각형까지만 넣고 사면체를 빼면 사면체 표면이 가짜 H₂를 만듭니다. 의도한 full Rips filtration에는 높은 차원의 coface도 필요합니다.

## Persistence는 생성과 소멸을 짝짓습니다

simplex를 filtration 값과 차원으로 정렬해 face가 먼저 오게 한 뒤 경계 열을 축약합니다. 영 열은 class를 만들고 나중의 pivot 열이 소멸을 짝짓습니다. 정사각형의 양수 수명 H₁ interval은 다음과 같습니다.

$$
[1,\sqrt2),\qquad \text{lifetime}=\sqrt2-1.
$$

diagram은 birth와 death를 좌표로 그리고 수명 0 쌍은 생략합니다. 무한 H₀는 표에 별도로 기록합니다. death 임계값에서는 이미 해당 interval이 살아 있지 않습니다. 같은 filtration 값의 순서가 수명 0 pairing을 바꿀 수 있어도 의미 있는 persistent module은 바꾸지 않습니다.

## 직접 풀기와 해석

가로 1, 세로 2로 바꾸면 구멍은 2에서 태어나 √5에서 죽고 수명은 약 0.236입니다. 숫자와 실제 채워진 simplex를 확인하세요. 모든 거리를 s배 하면 birth·death도 s배이고 Betti 패턴은 재조정한 threshold에서 같습니다.

이 결과는 해당 거리 filtration의 성질입니다. 미지 모집단 다양체를 추론하려면 표본·잡음·복원 가정이 더 필요합니다. 오래 사는 구멍도 유용한 근거일 수 있지만 diagram 하나만으로 그 존재의 과학적 원인까지 인증되지는 않습니다.

추가 읽기: [GUDHI, Rips Complex](https://gudhi.inria.fr/doc/latest/group__rips__complex.html).
