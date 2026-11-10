# 통계장론: 일차 자료와 신규 집필 범위

2026-11-11. 원주제 `statistical field theory`는 진행 중이다. 기존 상관함수 세 단원이나 경로적분 네 단원을 다시 세어 완료 처리하지 않는다. 현재 등록 601단원, 검증된 입문 원주제 316/362, 미완료 46개를 유지한다.

## 직접 확인한 자료

- [David Tong §1.1.1, §1.3](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S1): 고정 자화에 대한 미시상태 합, 배치별 자화와 평균의 차이, 온도 의존 유효 자유에너지, 국소 질서변수와 gradient 항을 확인했다. 유한 합을 적분으로 바꾸는 단계는 근사다. 이 교재의 절대 정규화에서는 측도 상수를 유지한다.
- [Tong §2, §2.1](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2): Gaussian 요동, 실수 장의 Fourier 공액 조건, determinant와 열 미분을 확인했다. 작은 곡률에서 사차항을 버린 결과를 임계현상의 확정값으로 사용하지 않는다.
- [Berges–Tetradis–Wetterich, hep-ph/0005122, §1 식 (1.9)–(1.12)](https://arxiv.org/html/hep-ph/0005122): 연결 생성함수의 미분과 Legendre 유효 작용의 정의를 HTML 본문에서 확인했다. 차원 없는 source 표기를 열역학 source와 구별한다. 이 검토가 전체 FRG 유도·수치해법을 읽고 검증했다는 뜻은 아니다.

Cambridge PDF의 실패한 요청은 읽은 자료에 포함하지 않는다. 아래 유한 모형과 열 미분 반례는 직접 유도하고 별도 Python Fraction 계산과 대조한다.

## 기존 내용과 추가할 내용

| 기존 단원 | 이미 있는 내용 | 새로 필요한 내용 |
|---|---|---|
| finite-gaussian-field-sources-connected-correlations | 평균·공분산·원천·제약 측도·기호 정규화 | 물리 격자 간격, 절대 자유에너지와 열 미분, 온도 의존 계수 |
| spatial-spectra-decay-and-correlation-length | DFT·경계·상관 길이·정련 | 실수 모드 수와 절대 측도의 Jacobian |
| interacting-nongaussian-connected-correlations | 이산 상호작용·누적량 | 연속 scalar 상호작용의 적분 가능성과 실제 적분/근사의 구별 |
| scalar-field-action-boundary-variation | 장·변분·격자 작용 | 국소 질서변수의 통계적 구성과 숨긴 미시상태의 엔트로피 |

## 집필할 네 처리

1. **유한 장의 측도와 열 자유에너지** — 첫 단원 ID `finite-field-measure-and-thermal-free-energy`. 1/2차원 격자, 유도 측도, 절대 Z, 온도 의존 quadratic 모형, 에너지·엔트로피·열용량의 구별. [모형 계약](thermal-field-model-contract.md)에 검증 범위를 고정한다.
2. **미시상태에서 국소 질서변수로** — block 상태별 정확 합·퇴화도·유효 자유에너지, 국소 gradient 가정의 근사성. 전체 자화 하나로 공간 구조를 잃는 예를 포함한다. ID와 계산 범위는 첫 단원 이후 확정한다.
3. **상호작용 scalar Gibbs 장** — 양의 사차항의 안정화, 유한 부피 대칭, 국소 Hessian과 실제 모멘트, 이산화·꼬리·적분 오차의 구별. 이산 Ising 합을 연속 phi4 적분 결과로 표시하지 않는다.
4. **연결 생성함수와 유효 작용** — source 단위, Legendre 변환, 공분산과 곡률 역관계, 제한 확률 자유에너지와 볼록 유효 작용의 구별. 평균장·loop 근사의 적용 조건을 설명한다.

통계장론 원주제는 네 처리의 실제 내용과 입문 범위를 함께 감사한 뒤 판정한다. 후속 원주제인 임계성·보편성, 재규격화, 유효 이론, 척도 분리는 별도로 남는다. fermion·gauge 측도와 비섭동 연구 심화도 위 네 처리만으로 완료되지 않는다.

## 구현 순서와 현재 상태

수학 기반과 독립 검산 → 상세 한영 원고 → 입력/계산 응답 검증 → Worker 수명 → 기존 홈페이지 색과 스타일의 그림·시각화·실험·해설 문제 → 실제 reader 접근성·언어·반응형 점검 → 단원 등록과 범위 감사 순서다. 준비 단계 파일은 홈페이지 완료 항목이나 등록 수에 더하지 않는다.

## 후속 관찰: 두 번째 처리 준비

첫 문단의601은 검토 시작 시점의 등록 수다. 이후 첫 처리 `finite-field-measure-and-thermal-free-energy`가 검증·통합되어 현재602단원이며 미완료 원주제46개는 유지한다. 두 번째 처리 ID는 `microscopic-blocks-and-local-order-parameters`로 확정했다. [새 모형 계약](block-order-model-contract.md)과 [기반 검증 기록](block-order-foundation.md)을 따른다. 한영 활성 초안 한 단원이 생겼지만 등록 수에는 더하지 않는다.

Kadanoff 원문 PDF의 pp.264–265를 텍스트와 p.265 이미지에서 확인했다. 두 값의 셀 변수와 최근접 셀 상호작용을 유지하는 것은 명시적인 모형 가정이다. 이 검토로 이후 페이지의 임계 scaling 유도 전체를 검증했다고 주장하지 않는다. 이 단원의 작은 유한 반례는 그 가정과 정확한 주변화를 구별하기 위한 독립 유도다.

## 후속 관찰: 두 번째 처리 등록

두 번째 처리의 [실제 reader와 범위 검토](block-order-reader-integration.md)가 완료되어 현재603단원이다. 첫 번째 측도/열 자유에너지와 두 번째 국소 질서변수를 검증했다. 다음은 상호작용 scalar Gibbs 장의 일차 자료·안정화·적분/꼬리 오차 범위를 고정하는 일이다. 연결 생성함수·Legendre 유효 작용과 원주제 감사도 남았다. 검증316/362, 미완료46 원주제와 전체 연구 심화 범위는 그대로다.
