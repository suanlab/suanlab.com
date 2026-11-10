## Kernel을 명시적인 특징으로 바꾸기

큰 Gram 행렬 대신 작은 특징 벡터를 저장하면 선형 계산으로 kernel 방법을 근사할 수 있습니다. 이전 고차원 확률 단원에서 본 한 쌍의 특징 오차를 여기서는 함수 절편과 전체 작은 Gram 구조로 확장합니다. 근사 정확도와 PSD 구조는 다른 성질입니다.

RBF kernel은 이상적인 독립 주파수 ω~N(0,ℓ⁻²)에 대해 E cos(ω(x−z))입니다. Gaussian 특성함수에서 이 식이 나옵니다.

$$
k(x,z)=\exp\!\left(-\frac{(x-z)^2}{2\ell^2}\right)=E_\omega\cos(\omega(x-z)).
$$

## 짝지은 cos와 sin 특징

각 주파수에 cos와 sin 좌표를 붙이고 전체를 √m으로 나눕니다. 그러면 삼각함수 항등식으로 근사 kernel을 얻습니다.

$$
\begin{aligned}
\phi_m(x)&=\frac1{\sqrt m}(\cos\omega_jx,\sin\omega_jx)_{j=1}^m,\\
\widetilde k(x,z)&=\phi_m(x)^T\phi_m(z)=\frac1m\sum_j\cos(\omega_j(x-z)).
\end{aligned}
$$

m=4는 4차원이 아니라 8차원입니다. 대각은 cos²+sin² 덕분에 항상 1입니다. 세 입력의 특징 행렬을 Z라고 하면 근사 Gram은 ZZᵀ이며 cᵀZZᵀc=‖Zᵀc‖²≥0입니다. 참 kernel과 다르더라도 이 PSD 성질은 유지됩니다.

## 실험에서 비교하는 것

파랑은 정확한 RBF 절편 k(x,z), 주황은 같은 z의 특징 근사입니다. 표는 실제 주파수와 질의의 특징값을 보여줍니다. Gram Frobenius 오차는 −1,0,1의 세 입력에 대한 오차이고, 표시 격자 최대 오차는 유한한 121개 질의에 한정합니다. 두 수치를 영역 전체 supremum이나 학습 일반화 오차라고 부르지 않습니다.

seed는 재현 가능한 작은 의사난수 발생기의 고정 실행입니다. 이상적인 독립 Gaussian 표본에서 성립하는 불편성·집중 경계가 고정 seed 하나의 오차 상한을 보장하지는 않습니다. m을 늘려도 그 실행의 오차가 매 단계 단조 감소할 필요는 없습니다. 소표본 오차가 잠시 커지는 것은 이론의 모순이 아닙니다.

## 계산과 학습의 새로운 교환

특징 차원은 계산·저장 비용뿐 아니라 근사된 함수 공간을 바꿉니다. kernel ridge를 특징 ridge로 바꿀 때 정규화와 데이터 오차도 함께 달라질 수 있습니다. 특징 수가 충분한지 평가하려면 목표 kernel, 입력 영역, 표본 크기, 요구 오차와 확률 조건을 명시해야 합니다.

## 확인 문제와 해설

1. 근사 kernel 값이 음수가 나오면 PSD가 아닌가요? **아니요.** Gram 이차형식의 비음수가 PSD 조건입니다.
2. m=32의 실제 좌표 수는? **64개입니다.** 짝지은 두 좌표를 셉니다.
3. 한 kernel 절편이 잘 맞으면 학습 결과도 같나요? **보장하지 않습니다.** 전체 Gram 오차와 정규화된 해의 안정성도 봐야 합니다.

추가 읽기: [Rahimi and Recht, Random Features](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf).
