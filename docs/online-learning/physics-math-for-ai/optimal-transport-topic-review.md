# 최적수송과 분포의 기하 원주제 대조

2026-11-12 기준. `optimal-transport`의 8개 원주제를 기존 한영 원고·실습·계산과 대조했다. 누적 26영역·253/509개 원주제의 대조 기록을 연결한다. 요청 당시 입문 48개 및 후속 입문 362/362 완료와 전체 연구 심화는 구분한다.

## 원주제와 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`, 계산은 `src/lib/learning-transport.ts`다.

| 원주제 | 단원 ID | 보강한 논증·실행 |
|---|---|---|
| Monge·Kantorovich | monge-kantorovich-mass | 유한 가능집합의 compact성과 최적해 존재; 연속 출발의 map이 원자 극한에서 계획 분할로 바뀌는 유리수 적분 |
| 계획·쌍대 | transport-plans-dual-certificates | 최대 부등식 위반량으로 potential 복구; slack 간선의 질량 상계·유리수 원/쌍대 인증 |
| Wasserstein | wasserstein-quantile-metrics | Minkowski 삼각부등식·제곱 비용 교차 제거; 약수렴하지만 W₂가 수렴하지 않는 꼬리 원자 |
| Entropy·Sinkhorn | sinkhorn-entropic-couplings | 양의 유일 해·가능 계획의 목적 차와 KL 등식·Pinsker 범위; 80자리 Decimal 검산 |
| 동적 수송 | dynamic-transport-continuity | 모멘트 Bessel·시간 Cauchy–Schwarz 하한; 시간 속도 초과 작용과 다항식 적분 |
| Displacement | wasserstein-displacement-paths | 임의 두 시간의 geodesic 거리 증명; 모든 coupling의 mixture/이동 분산 차 항등식 |
| Gradient flow | wasserstein-gradient-flow-jko | 모멘트 거리·entropy 하한의 Gaussian 축소; T=0의 고정 시간 일차 오차·이동 비용 검산 |
| 근사 오차 | transport-regularization-grid-errors | entropy에서 비용 편향 상계 유도; 거리 단위의 이산화+제곱근 인증과 정지 오차 예산 |

기존 설명·연습과 여덟 실습의 기저·조절기·안내·퀴즈를 보존하고 표준 라이브러리 Python 8종을 한영으로 추가했다. 시간 속도·JKO 시간 오차의 한영 SVG 네 개는 별도 해석·해상도 연구이며, 기존 조절기의 반복 수나 시간표를 확장했다고 주장하지 않는다.

거리 오차 예산 예제의 ε는 현재 실습 최솟값보다 작다. 이론의 충분 조건과 현재 UI에서 가능한 실행을 구별하고, 지정한 gap 목표가 고정 반복 수로 자동 달성되지 않음을 밝힌다. 정칙성·소실 경계항을 전제로 한 동적 모멘트 증명과, T=0에 특화된 시간 오차 상계를 일반 PDE 수렴 정리로 확대하지 않는다.

## 수치 계산 보완

두 원자 엔트로피 기준 해의 비정규화 초과 비용은 작은 교차 질량 s에서 2Ls로 직접 계산한다. L=3,ε=0.05에서 약 2.63×10⁻²⁶을 유지하며, 절대 수송 비용은 같은 값으로 반올림될 수 있다. 절대 비용 차감과 해석적으로 중심화한 비용의 평가를 구별한다.

JKO 감소는 평균 차·합의 곱과 scale 변화의 로그 나머지로 평가한다. 평형 scale 1·T=1과 평균 ±10⁻²⁰에서 절대 에너지를 빼면 0이지만, 양의 감소·이동 비용을 보존한다. 평형 부근의 T−σ₀²는 보상 제곱으로 낮은 부분을 복구하여 작은 감소의 상대 정확도도 유지한다. Scale 상태의 유한 정밀도 반올림 한계는 남으며 작은 음수를 임의로 0에 clipping하지 않는다.

## 출처·보존

[Peyré·Cuturi 저자 교재](https://optimaltransport.github.io/book/)의 수송·쌍대·엔트로피·경로·분포 흐름 배경을 대조했다. JKO의 변분 시간 이산화는 [Jordan·Kinderlehrer·Otto 원 논문](https://epubs.siam.org/doi/10.1137/S0036141096303359)의 틀에 연결한다. 작은 유리수·clock·Gaussian·꼬리 원자·오차 예산은 본문에서 직접 유도하며 원 논문의 실험 재현을 주장하지 않는다.

영문 원고 두 곳의 수식만 언어 중립적인 membership 표기와 마침표 정리로 한영에 맞추었다. 이 두 정규화를 제외한 원래 본문·연습의 보존을 검증했다. 변경한 16개 원고의 이전 fixture SHA·현재 SHA·이 기록을 reviewed ledger에 연결하고 immutable fixture 네 개는 바꾸지 않는다.

## 검증

관련 계산·원고 보존 검사 58개와 타입·lint·콘텐츠 검증이 통과했다. 한영 Python 16개 블록의 코드 SHA·출력·표시 수식 일치를 확인했다. 독립 140자리 계산의 엔트로피 편향 54건과 JKO 에너지 감소 144건에서 최대 상대 오차는 각각 6.8×10⁻¹⁵·6.1×10⁻¹⁶ 이하다. 기준은 입력과 실제 계산 상태를 정확한 binary64 값으로 읽은 에너지이며, JKO 상태 궤적 전체를 140자리로 풀었다는 뜻은 아니다.

1,893개 경로 정적 빌드와 1,888개 HTML의 내부 링크·자산·fragment·sitemap·canonical 검사를 통과했다. 실제 export의 한영·320/768/1440px·명암 82개 상태에서 접근성 위반·브라우저 오류가 0이고 상호작용 10묶음·안내 24개·퀴즈 16개·이전 계산 28개·원주제 연결 8개를 확인했다. 정수 초안의 blur 오류 표시·모형 보존·Escape 복원, 표·코드의 키보드 스크롤과 최종 320px 그림 네 개도 확인했다.

작업 소스 1,882개·학습 원고 1,300개·릴리스 파일 27개·설정 8개의 후보 byte 일치와 SVG 네 개의 export 일치를 확인했다. 전체 CI 및 공개 배포 완료는 이 로컬 검증과 구별한다.

## 후속 범위

일반 map 정리의 전체 증명·고차원 비정렬 수송, 임의 비용의 Sinkhorn 복잡도·작은 ε의 continuation, 비정칙 밀도의 동적 수송, 일반 자유 에너지의 JKO 수렴, 비균등·비균형 수송의 근사 인증은 후속 심화 범위다. 이번 원주제 대조가 해당 분야의 모든 연구 이론 완료를 뜻하지 않는다. 다음은 수치계산과 자동 미분의 10개 원주제다.
