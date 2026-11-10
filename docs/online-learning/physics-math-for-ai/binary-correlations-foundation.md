# 상관·요동 1: 두 스핀 연결 공분산 기반 작업

2026-11-07 현재 상태: 세 단원을527개 등록 교재에 연결하고 실제 읽기 검증을 완료했다. [최신 통합 기록](correlations-integration-verification.md)을 우선한다. 아래의 준비·미등록 표기는 당시 단계의 기록이다.

> 현재 단계(2026-11-07): 세 가족의 [학습 화면 준비와 별도 native 검증](correlations-teaching-components.md)을 마쳤다. 아래는 해당 준비 단계의 기록이며 다음은 실제 Interactive Book 등록·Next 자산·읽기 검증이다.

2026-11-07. `binary-connected-correlations-field-hessian`의 수치 모델과 상세 한영 본문을 작성했다. 원래 `상관·fluctuation` 주제의 첫 가족이며 **아직 교재에 등록하지 않았다**. 등록된 단원 수 524개·안내 실험 1,572개와 미완료 입문 주제 75개는 이 준비로 바뀌지 않는다.

## 학습 내용

네 이름 붙은 상태에서 원시 곱과 연결 공분산을 구별하고, 두 국소 장의 Hessian을 유도한다. 음의 교차 원소와 양의 정부호성, 공통 장의 스핀당 반응, source 척도, 물리적 단위, 조건부 평균과 인과 해석의 차이, Fisher 좌표 변환을 설명한다. 같은 방향·반대 방향의 손계산과 해설 문제 6개를 포함한다. 영어 약 2,576단어, 한글 음절 4,998개이며 각각 13절·수식 13개·상태 표를 실제 Markdown 파이프라인으로 검증했다.

## 근거

- 모델: `src/lib/learning-binary-correlations.ts`. 정확한 입력 유리수와 방향성 지수·로그 구간, 분산의 양의 쌍 합, 행렬식의 양의 삼중 합을 사용한다.
- 독립 참조: `scripts/learning/generate-binary-correlations-oracle.py`와 `scripts/fixtures/binary-correlations-oracle.json`. 실제 binary64 입력의 Decimal 중심화 모멘트 21사례·420관측을 대조했다. 400/900자리와 추가 80자리 확인, 생성 파일 바이트 재현을 확인했다.
- 새 모델 30개·본문 7개 테스트 통과. 이전 상전이 등록 회귀 5개를 포함한 묶음 **42개 통과**.
- 최소 양수 결합의 양·음 부호, 정밀도 상한의 미해결 표시, 선형 범위 아래의 양의 분산, 대칭·독립·척도·유한차분 반응을 검사했다.
- 자세한 유도는 [계산 계약](binary-correlations-derivation-notes.md), 실행 기록은 `.runtime/learning/binary-correlations/verification.json`을 따른다.

## 계속할 작업

같은 원래 주제의 주기 Ising 고리와 상관된 AR(1) 평균 오차 모델·한영 원고가 남았다. 이후 세 가족의 고정 개념 단계·시각화·안내 실험·퀴즈, 취소 가능한 worker와 실제 읽기 연결 검증을 순서대로 진행한다. 현재 모델 테스트와 본문 렌더링은 신규 단원의 브라우저 사용성·등록·공개 배포를 증명하지 않는다. 전체 교육과정과 연구 심화도 계속 미완료다.
