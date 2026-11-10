## 생성자는 연속 운동을 정합니다

행렬 지수는 무한소 생성자를 유한 군 원소로 바꿉니다. 고정 각벡터 ω에 대해 Ω=hat(ω), 병진 생성자 v를 묶어 ξ̂=[[Ω,v],[0,0]]로 둡니다. T(t)=exp(tξ̂)는 Ṫ=ξ̂T, T(0)=I를 만족합니다. 이는 일정한 spatial twist 관례이며 body-frame 관례와 구별해야 합니다.

ω가 단위축일 필요는 없습니다. 노름에 t를 곱하면 회전각이며 비영일 때 방향이 축입니다. 실험은 각생성자 세 성분·병진 생성자 세 성분·시간을 모두 조작하므로 명시한 유한 범위에서 임의의 일정한 se3 twist를 계산합니다.

## Rodrigues와 결합 병진

W=tΩ, θ=||tω||로 두면 다음과 같습니다.

$$
R=I+\frac{\sin\theta}{\theta}W+
\frac{1-\cos\theta}{\theta^2}W^2,
\qquad
p=J(t\omega)\,tv,
$$

$$
J=I+\frac{1-\cos\theta}{\theta^2}W+
\frac{\theta-\sin\theta}{\theta^3}W^2.
$$

이동은 ∫₀ᵗexp(sΩ)v ds이기도 합니다. 운동 중 생성자의 병진 방향도 회전하므로 Ω만 지수화하고 tv를 붙이면 대개 다른 변환입니다. 파랑은 정확한 원점 궤적의 xy 투영, 주황은 단순 직선 이동입니다. 전체 xyz·방향은 수치와 비등각 frame 그림에서 읽습니다.

## Quarter-turn 운동 손계산

ω=(0,0,π/2),v=(1,0,0),t=1이면 적분 결과는 p=(sinωz/ωz,(1−cosωz)/ωz,0)=(2/π,2/π,0)≈(0.637,0.637,0)입니다. 단순 tv=(1,0,0)와 차이는 약 0.733이며 회전·병진 결합 효과입니다. 수치 적분 오차가 아닙니다.

## 영 각도는 정칙 극한입니다

공식에는 θ로 나누는 항이 있지만 계수 극한은 유한합니다.

$$
\frac{\sin\theta}{\theta}=1-\theta^2/6+\cdots,\quad
\frac{1-\cos\theta}{\theta^2}=1/2-\theta^2/24+\cdots,\quad
\frac{\theta-\sin\theta}{\theta^3}=1/6-\theta^2/120+\cdots.
$$

영 각생성자에서는 R=J=I라 p=tv입니다. 작은 비영 각에서도 급수가 소거 오차를 막습니다. 구현은 독립적인 4×4 행렬 지수 급수와 exp(sΩ)v의 별도 적분으로 검산합니다.

## 직접 풀기와 풀이

ω=0,v=(1,0,0)에서 곡선이 직선 단위 이동이 되는지 확인하세요. 다음 ω=(0,0,2π),v=0이면 생성자가 비영인데 최종 회전은 항등입니다. 따라서 지수 좌표는 전역 단사가 아닙니다. 같은 한 바퀴에서 v=(1,0,0)이면 횡방향 병진은 평균해 0이 되며 앞으로 한 단위가 아닙니다.

## 연구와의 연결

지수 갱신은 pose 최적화·연속 시간 운동 모형에 유용하지만 frame·단위·곱 순서를 명시해야 합니다. principal 회전 logarithm에는 branch 선택과 π 근처의 주의가 필요합니다. 이 실험은 지수를 계산하며 전역적으로 유일한 역 logarithm을 가정하지 않습니다.

추가 읽기: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
