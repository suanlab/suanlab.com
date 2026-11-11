# 근사이론·커널·표현력 원주제 대조

2026-11-11 기준. `approximation-kernels`의 10개 원주제를 기존 한영 원고와 `learning-approximation.ts`에 대조했다. 기존 입문·실습·연습을 보존하며 구성 증명, 공간·측도 조건, 유한 표본과 최적화의 서로 다른 보장을 보강했다. 이 기록을 포함한 대조 범위는 19영역·185개 원주제이며 전체 51영역·509개 일반 이론과 연구 심화의 완료를 뜻하지 않는다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{ko,en}.md`이다.

| 원주제 | ID | 정의·가정·보강 근거 |
|---|---|---|
| 다항·spline 근사 | polynomial-spline-approximation | 연속성 계수·이항 분산의 Bernstein 균등 수렴 증명, Rolle 보간 나머지, 정확 오차와 일반 상계 |
| 보편 근사 조건 | universal-approximation-conditions | 모든 1차원 연속 함수의 ReLU spline 구성, Lipschitz 경계·매듭 반례·bias 제거의 제한 |
| RKHS·Mercer | rkhs-mercer-features | PSD 유한합의 몫·완비화·연속 평가, full support·측도 정규화와 support 밖 복원 반례 |
| Representer 정리 | representer-span-penalty | 유한 span의 존재·유일성 조건, singular Gram에서 같은 함수의 여러 계수 |
| Kernel regression | kernel-ridge-regression | 고유방향 축소·고정 입력의 편향/분산 항등식, 네 잡음의 정확 평균 |
| Gaussian process | gaussian-process-posteriors | 독립 Gaussian 잔차의 조건화 증명, 동시/점별 구간, 작은 양의 분산 |
| Random features | random-feature-kernel-approximation | 정확 기대 제곱 오차·유한 집합 Hoeffding, PSD 근사와 resolvent의 예측 안정성 |
| NTK·극한 조건 | ntk-linearized-training | PSD 고유방향의 수축·경계 진동·발산·영공간 한계, 원 논문의 유한 시간·초기화 조건 |
| 표현력·표본 효율 | expressiveness-sample-efficiency | 고정 유한족의 균일 경계·ERM 위험, 16개 가설과 모든 iid 입력 열의 전수 계산 |
| 세 종류의 오차 | approximation-estimation-optimization-errors | 경험 위험 삽입의 상계 증명, 모든 이항 자료에서 GD 경로의 기대값·초기화 효과 |

## 독립 예제와 그림

Python 10종을 한영에서 같은 코드로 실행한다. 표·코드의 별도 예제를 기존 실습의 새로운 제어라고 부르지 않는다. 다항·hinge·Gram·singular 계수·편향/분산·고정 kernel·분류·GD 평균은 정확분수 또는 독립 계산을 사용한다.

모서리 |x−1/3|의 균등 spline은 n=2·3·4·6에서 2/9·0·1/9·0 오차를 갖는다. 비포함 격자의 매 단계 개선은 보편 근사의 존재 명제가 아니다. 상수 kernel은 계수 (4/5,0),(0,4/5),(2/5,2/5)가 같은 함수와 목적함수를 주지만 마지막만 표준 선형계를 만족한다.

고정 두 입력 회귀의 네 잡음 평균으로 λ=0·1/4·1·4의 기대 오차 .25·.167800·.218889·.539766을 확인한다. Random feature는 Gaussian cosine의 두 모멘트를 별도 적분한다. n=3,ε=.2,δ=.05의 유한 집합 충분 경계 m=295를 실습 최대 m=32와 구별한다. 고정 seed의 인증이나 연속 영역 supremum은 주장하지 않는다.

한영 그림 두 종류는 n=2의 고정 NTK 안정성/영공간과 16개 공정 부호 자료에서 기대 위험/경험 최적화 초과량을 설명한다. NTK 패널의 세로 축은 서로 다르고 기대 위험 그림 위쪽은 t≥3을 확대한다. 기존 비선형 tanh 실습과 구별한다.

네 입력·16개 이진 가설을 고정한 ERM 예제는 모든 4ⁿ iid 열을 세어 기대 분류 오차 (1/2)(3/4)ⁿ을 확인한다. 이는 n마다 차수를 키우는 기존 소멸 다항식과 다른 고정 함수족 학습 곡선이다. 일반 제곱 손실에 [0,1] 유계 손실 경계를 가정 없이 적용하지 않는다.

## 수치 수정과 범위

기존 GP 구현은 length=1,σ²=10⁻³⁰,x=0의 양의 분산과 σ²=0,x=10⁻⁸ 또는 10⁻¹⁰⁰의 분산을 0으로 잃었다. 가까운 무잡음 관측을 먼저 조건화하고 expm1으로 공분산 차를 계산한다. 거리로 나눈 조건부 공분산을 계산한 뒤 배율을 곱하고 제곱해 조기 underflow를 줄인다. 나머지 두 점의 양정 해법과 resolvent 항등식으로 잡음 효과를 더한다. 숨은 jitter는 없다.

RKHS의 c²/ρ를 먼저 제곱하면 ρ=10⁻³⁰⁰,c=10⁻²⁰⁰의 유한 비용 10⁻¹⁰⁰을 잃었다. c/√ρ를 계산한 뒤 hypot·제곱을 사용한다. 공간 안의 유한 노름이 수치 범위 밖이면 —·안내로 처리하며, ρ=0에서 공간 밖인 함수의 ∞와 구별한다. GP의 표현 범위 아래 양의 분산도 —·안내로 표시하고 정확 무잡음 관측의 실제 0은 유지한다.

다항 보간의 질의 오차도 두 높이의 차 대신 해석적인 양의 오차 식으로 계산한다. n=16,x=1−2⁻⁵³의 약 6.94×10⁻¹⁸를 0으로 잃지 않는다. Spline은 반올림된 n·x만으로 셀을 결정하지 않고 저장 매듭과도 대조한다. 양수의 범위 손실은 —·안내와 질의점 보류로 표시한다.

`finiteRiskErrors`는 현재 상수에서 ERM을 빼는 대신 해석적 변위를 유지하고 경험 초과량 d², 부호 있는 모집단 차이 d(2μ+d)를 계산한다. K=12,c₀=.5+2⁻⁵³,η=.5,t=16에서 현재 상수가 .5로 반올림되어도 변위 2⁻⁶⁹와 초과량 2⁻¹³⁸을 유지한다. 범위 손실의 —·안내와 실제 영 오차·반대 상수의 같은 모집단 위험을 구분한다.

## 출처와 후속 범위

- [MIT 9.520, RKHS/Mercer 강의](https://ocw.mit.edu/courses/9-520-statistical-learning-theory-and-applications-spring-2003/resources/class14/): RKHS·특징의 배경. 측도 네 종류와 support 밖 반례는 직접 계산했다.
- [Schölkopf–Herbrich–Smola, §2](https://alex.smola.org/papers/2001/SchHerSmo01.pdf): 직교 성분에 대한 representer 논증. 유한 span 존재 증명과 singular 계수 예제는 별도로 전개했다.
- [Rasmussen–Williams, Chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf): GP 조건화의 배경. 고정 입력 편향/분산·안정적인 세 점 조건화·고정밀 비교는 별도로 전개했다.
- [Rahimi–Recht, §3](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf): 짝지은 Fourier 특징과 한 쌍의 집중 경계. 정확 RBF 분산·유한 n² union bound·resolvent 비교는 본문에서 유도했다.
- [Jacot–Gabriel–Hongler, §§2·4·5 및 Appendix A.2](https://arxiv.org/html/1806.07572v4): 초기화·극한·시간 조건의 근거. 유한 이산 안정성 예제는 원 논문의 무한 폭 실험을 재현했다는 주장이 아니다.

다차원 정량 보편 근사·최적 깊이/폭, 일반 Mercer 전개의 완전 증명, GP hyperparameter 불확실성·동시 band, 연속 영역 random feature와 Nyström 통계 경계, 비선형 feature learning·일반 NTK 극한, 무한 함수족 복잡도와 minimax 표본 효율은 후속 심화다. 원주제 대조를 해당 전체 연구 이론의 완료로 확대하지 않는다.

## 검증 기록

- 근사이론 계산 검사 35개·기존 원고 보존 검사 7개·선택한 한영 reader 렌더링 검사 10개를 통과했다. 이 52개 검사를 전체 테스트 모음의 통과로 확대하지 않는다.
- Python 코드 20개의 최종 SHA·출력을 대조해 재실행했다. 10개 단원의 한영 절 수·표시 수식·실행 코드가 일치하며 영어 원고의 한글과 두 언어 원고의 일본어 가나가 없다. 320px에서 새 표 20개와 최종 SVG 4개의 배치를 확인했다.
- GP 분산 33개 입력을 독립적인 650자리 Gaussian 조건화와 대조했다. 실제 0·극소 양수·subnormal·표현 범위 손실을 구분했다. RKHS 노름 4개 입력과 질의/최적화 오차 16개 입력은 binary 입력값의 정확분수로 별도 대조했다. 유계 length·query·noise 격자 3,872개에서도 분산의 비음수·사전 분산 경계를 검사했다. n=1~16의 spline 매듭과 양쪽 이웃 binary64 값에서 셀·오차의 비음수도 확인했다.
- 실제 정적 빌드의 경계 선행 검사는 25개 상태·10개 동작 그룹이 통과했다. `npm run check:approximation`의 한영·320/768/1440px·다크 모드 103개 접근성·배치 상태, 12개 동작 그룹, 안내 예제 30개·퀴즈 20개를 통과했다. 표·코드의 키보드 스크롤, 정수 draft·Escape 복구, 저장·언어 전환·초기화, 이전 계산 회귀도 확인했다.
- `npm run lint`, `npm run build`, `npm run typecheck`, `npm run validate:content`, `npm run check:export`가 통과했다. 빌드 1,893개 경로와 내보낸 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 확인했다.
- 소스 1,882개·한영 원고 1,300개·설정 8개·릴리스 파일 31개를 빌드 후보와 바이트 대조했고, 내보낸 SVG 4개도 원본과 같다. 공개 반영 완료는 commit/push 이후 GitHub Pages 배포 결과로 별도 확인한다.
