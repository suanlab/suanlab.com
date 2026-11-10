# 구성방정식 구현 전 유도 메모

이후 [계산 모델·상세 한영 원고](continuum-constitutive-foundation.md)를 작성하고 검증했다. 아래는 사전 검토·유도 당시 기록이다. 실습 화면 통합은 남았다.

2026-11-06. [두 단원 설계](continuum-constitutive-authoring-plan.md)의 자체 유도와 검증 계약이다. 제품 구현·학생용 상세 원고·고정밀 참조·화면 검증은 아직 하지 않았다. [검토한 자료](continuum-constitutive-source-audit.md)와 아래 직접 계산을 구분한다.

## 같은 매끄러운 변형 이력에서 두 점탄성 모델 비교

일축 공학 변형률 ε=ε₀sin(ωt)를 지정한다. E>0은Pa, η>0은Pa s, ω는rad/s다. 변형률과 내부 탄성 변형률 모두 작은 범위로 제한한다. 시작 응력은 σ₀=E e₀로 두면 초기 내부 탄성 변형률 e₀를 명시할 수 있다. 처음 전체 변형률은0이므로 Maxwell의 초기 점성 변형률은−e₀다.

Kelvin–Voigt는 두 요소의 변형률이 같고 응력을 더한다.

$$
\sigma_{KV}=E\varepsilon+\eta\dot\varepsilon,
\qquad \psi_{KV}=\frac12E\varepsilon^2,
\qquad \mathcal D_{KV}=\eta\dot\varepsilon^2,
\qquad \sigma_{KV}\dot\varepsilon=\dot\psi_{KV}+\mathcal D_{KV}.
$$

Maxwell은 두 요소의 응력이 같고 변형률을 더한다. 점성 요소의 변화율은σ/η이고 탄성 요소의 변화율은σ̇/E다.

$$
\dot\sigma+\frac{\sigma}{\tau}=E\dot\varepsilon,
\qquad \tau=\frac\eta E,
\qquad \psi_M=\frac{\sigma^2}{2E},
\qquad \mathcal D_M=\frac{\sigma^2}{\eta},
\qquad \sigma\dot\varepsilon=\dot\psi_M+\mathcal D_M.
$$

순간 기계 전력이 음수여도 수동성이 깨졌다는 뜻은 아니다. 저장 에너지가 되돌아갈 수 있다. 비음 조건은 소산에 적용하며 총 입력 일과 저장 변화·소산 적분을 비교한다. 단위는ψ가J/m³, 전력과소산률이W/m³다.

## Maxwell의 초기 과도와 정상 주기 응력

x=ωτ, a=Eε₀ x²/(1+x²), b=Eε₀ x/(1+x²)라고 놓자. 실제 해와 미분은 다음과 같다.

$$
\sigma(t)=a\sin(\omega t)+b\cos(\omega t)+(\sigma_0-b)e^{-t/\tau},
$$

$$
\dot\sigma=a\omega\cos(\omega t)-b\omega\sin(\omega t)
-\frac{\sigma_0-b}{\tau}e^{-t/\tau}.
$$

σ(0)=σ₀이고 실제 도함수를 원래 ODE에 대입해 확인한다. 독립 참조에서는 이 식을 그대로 복사하지 말고 초기 기억 항과 지수 커널 convolution을 고정밀로 적분한다. 수치 미분과 창의 기계 일·저장 변화·소산 적분도 별도로 검사한다. 정상 주기 해의 초기값은b이며, σ₀=0의 첫 주기는 일반적으로 응력–변형률 경로가 닫히지 않는다.

$$
\sigma(t)=\sigma_0e^{-t/\tau}
+E\int_0^te^{-(t-s)/\tau}\dot\varepsilon(s)\,ds.
$$

ω=0이고 위의 사인 이력을 유지하면 변형률은 항등적으로0이다. Maxwell은 초기 응력만 완화하며 외부 전력0에서 저장 에너지가 소산으로 바뀐다. 이 경우 유한한 주기와 주기 손실을 억지로 부여하지 않는다. η=0/E=0은 τ 매개화의 범위 밖이며 별도의 구성 극한과 구분한다.

## 정상 주기의 저장·손실과 첫 주기의 차이

사인 변형률 기준으로 응력을 ε₀[E' sin(ωt)+E'' cos(ωt)]라고 정의한다.

$$
E'_{KV}=E,\qquad E''_{KV}=\eta\omega,
\qquad E'_M=\frac{Ex^2}{1+x^2},\qquad E''_M=\frac{Ex}{1+x^2}.
$$

$$
\oint \sigma\,d\varepsilon=\pi\varepsilon_0^2E''\ge0.
$$

마지막 식은 ω>0의 정상 주기에 적용한다. 초기 과도가 남은 첫 주기의 실제 일은 별도로 적분해야 하며 저장 에너지 차이가 일반적으로0이 아니다. loop 면적·기계 일·열 소산을 구별한다. 정상 해를 과도 해라고 부르거나, 닫히지 않은 첫 경로를 닫힌 loop라고 부르지 않는다.

손계산 기준 E=2Pa,η=1Pa s,ε₀=0.002,ω=1rad/s이면 τ=0.5s,x=0.5,a=0.0008Pa,b=0.0016Pa다. Maxwell의 저장·손실 탄성률은0.4Pa·0.8Pa다. Kelvin–Voigt는2Pa·1Pa이며 두 모델의 순환 손실은 각각π·0.002²에 해당 손실률을 곱한다. 이는 차원을 명시한 교육용 값이며 특정 실제 고분자의 계수라고 주장하지 않는다.

## 비압축 neo-Hookean 단축 경로

기준 체적당 에너지 W=μ(I₁−3)/2, μ>0을 사용한다. λ>0인 주신장과 그 후의 강체 회전R을 둔다.

$$
F=R\operatorname{diag}(\lambda,\lambda^{-1/2},\lambda^{-1/2}),
\qquad \det F=1,\qquad B=FF^T,
\qquad \sigma=\mu B-pI.
$$

회전된 측면도 함께 회전시켜 자유 측면 traction 조건을 적용하면 p=μ/λ다. 축방향 Cauchy 응력과 기준 단면적당 nominal 응력은 다르다.

$$
\sigma_{ax}=\mu(\lambda^2-\lambda^{-1}),
\qquad P_{ax}=\mu(\lambda-\lambda^{-2}),
\qquad A=\frac{A_0}{\lambda},
\qquad A\sigma_{ax}=A_0P_{ax}.
$$

같은 수치의 면적을 두 응력에 곱하면 힘이 다르게 나오는 오류를 일부러 대조한다. 압력을 재료 상수로 고정하지 않으며 측면 경계를 바꾸면 압력 결정도 달라짐을 설명한다. 공간 회전 뒤에 응력은RσRᵀ로 바뀌지만 저장 에너지·주응력·실제 대응 면의 traction 크기는 바뀌지 않는다. 좌표별 응력 성분의 변화와 물성의 변화는 다르다.

## 안정적인 작은 변형과 일축 접선의 범위

$$
W=\frac\mu2(\lambda^2+2\lambda^{-1}-3)
=\frac{\mu(\lambda-1)^2(\lambda+2)}{2\lambda},
\qquad \frac{dW}{d\lambda}=P_{ax},
\qquad \frac{dP_{ax}}{d\lambda}=\mu(1+2\lambda^{-3})>0.
$$

log stretch ℓ을 기본 입력으로 삼으면 λ=exp(ℓ)가1로 반올림되어도 expm1(ℓ)의 비영 변화와 안정 에너지를 별도로 계산할 수 있다. 실제 기계 일에는 기준 체적당 W와 공액인P dλ를 사용한다. Cauchy 응력에 dλ를 곱해 같은 에너지라고 부르지 않는다. 닫힌 일축 경로의 적분은0이지만 부동소수점 진단을 임의 경로에 대한 엄밀한 증명으로 확대하지 않는다.

λ=2,μ=2Pa이면 W=2J/m³,P=3.5Pa,σ=7Pa이고 현재 면적은A₀/2다. λ가1에 가까우면 P≈3μ(λ−1)이므로 일축 작은 변형 계수는3μ다. 위 양의 접선은 이 경로의 성질이다. 일반 다축 변형의 강한 타원성·전역 안정성·시편 좌굴과 같은 주장에는 별도 조건과 해석이 필요하다.

## 독립 검증의 최소 범위

점탄성은 고정밀 convolution·실제 시간 미분·시간창의 일과 소산, 초기 기억 제거 대조, 정상 주기와 첫 주기의 구별을 포함한다. 초탄성은 고정밀 에너지 미분·실제 기준/현재 면적의 힘·회전 공변성·작은 신장과 로그 관측·변형 순환 일을 포함한다. 모델과 검증에 같은 잘못된 응력·면적 정의를 복제하지 않는다. 이 기록만으로 새 단원 등록이나 입문 검증 수를 올리지 않는다.
