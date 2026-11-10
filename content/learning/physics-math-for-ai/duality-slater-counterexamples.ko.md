## 하한과 값의 등호 정리는 다릅니다

가능한 부등식 승수의 Lagrangian infimum은 모든 primal 가능값의 하한입니다. 이를 최대화하면 d*≤p*라는 약 쌍대성입니다. 강 쌍대성은 최적값의 등호이며 유한 승수에서 supremum을 달성하는지는 별도입니다.

Slater는 유한 최적값의 볼록 문제에 대한 충분 qualification입니다. 적절한 relative interior의 가능점에서 비선형 부등식을 엄격히 만족하고 affine 등식 조건을 갖춥니다. 충분조건이 실패했다는 사실만으로 값의 등호 여부는 정해지지 않습니다.

## Strict 가능성과 달성 승수

첫 모드는 min (x−2)²/2, x≤1입니다. x=0은 strict 가능하고 primal 최적값은 x=1에서 0.5입니다.

$$
d(\lambda)=\lambda-\lambda^2/2,\quad\lambda\ge0,
\qquad\lambda^*=1,\quad d^*=0.5.
$$

dual 곡선은 primal 기준 높이에 닿습니다. 이 정상 예제에서는 witness 크기를 사용하지 않습니다.

## 값은 같지만 유한 최적 승수가 없는 경우

min x, x²≤0은 가능점 0뿐이라 p*=0이며 strict 가능점이 없습니다. λ>0에서 x+λx²를 최소화하면 다음과 같습니다.

$$
x_\lambda=-1/(2\lambda),\qquad d(\lambda)=-1/(4\lambda).
$$

λ=0의 infimum은 x=−M으로 −∞입니다. 양수 λ에서는 λ→∞에서만 d→0이라 d*=p*=0이지만 유한 최적 승수는 없습니다. 최적점의 KKT 정상성은 1+2λ·0=0이어야 하므로 불가능합니다. 볼록성만으로 승수 존재를 얻지는 못했습니다.

## 실제 양의 간극

세 번째는 Boyd·Vandenberghe 연습문제 5.21을 명시해 사용합니다.

$$
\min_{x\in\mathbb R,\ y>0} e^{-x}
\quad\text{subject to}\quad x^2/y\le0.
$$

perspective quadratic x²/y는 y>0에서 볼록·비음입니다. 가능성은 x=0을 강제해 primal 값 1을 달성합니다. 모든 유한 λ≥0에서 L은 비음이고 x=M,y=M⁴이면 e⁻ᴹ+λ/M²→0이므로 d(λ)=0, 최적값 간극은 1입니다.

이 sequence는 함수 정의역 안이지만 primal 불가능점입니다. 더 좋은 가능 해가 아니라 무제약 L infimum의 witness입니다. 여기서는 dual supremum은 달성하지만 내부 L infimum은 달성하지 않습니다. 세 달성 주장을 구별하세요.

## 직접 풀기와 해석

세 번째 모드에서 M을 4에서 16으로 늘리면 witness L은 0으로 내려가고 primal 값은 1입니다. 이것이 약 쌍대성 반례가 아닌 이유를 설명하세요. 값의 간극은 0이지만 승수 달성이 없는 두 번째와 비교하세요.

AI 최적화 인증에는 primal·dual 정의역, 볼록성, qualification을 명시해야 합니다. 유한 그림만으로 무한 영역의 infimum을 확정할 수 없으며 명시한 sequence와 하한 논증이 그 근거입니다.

추가 읽기: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
