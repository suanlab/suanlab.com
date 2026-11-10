# Uncertainty·measurement: 수치 두 가족과 상세 한영 본문

2026-11-08. `finite-state-covariance-uncertainty`와 `qubit-povm-instrument-backaction`의 상세 한영 원고4개를 작성했다. 수치 기반에 설명·유도·직관적 그림 읽기·손계산·반례·해설 문제를 연결했다. 아직 Worker·교수 화면·reader에 등록하지 않았으므로560단원·1680안내·305/362입문 검증·57미완료를 유지한다.

## 준비한 원고

- [준비 공분산과 불확정성: 한국어](finite-state-covariance-uncertainty-chapter.ko.md), [English](finite-state-covariance-uncertainty-chapter.en.md): 중심화·Hilbert–Schmidt Gram·Schrödinger/Robertson 유도·포화·복소 사영·실수 공분산 타원·ensemble 분해·Gaussian 위치/운동량·표본 오차와 장치 오차의 구별.
- [POVM과 측정 후 상태: 한국어](qubit-povm-instrument-backaction-chapter.ko.md), [English](qubit-povm-instrument-backaction-chapter.en.md): 효과·Kraus/instrument·조건부/비선택 상태·같은 효과의 세 장치·Bloch 압축/피드백·순차 결과·raw/보정 pointer·RMS 오차/교란·상태 거리·Ozawa 보정 관계·Choi.

각 원고는12절·해설6문제·표2개다. 네 원고 전체48절·해설24개·표8개·표시 수식34개이며, 영어 두 원고는 각각2079단어·2176단어다. 두 언어의 표시 수식과 문제 순서를 일치시켰다. 기존 홈페이지 스타일을 사용하는 읽기 화면은 다음 통합 단계에서 연결한다.

## 내용 검증

새17검사는 네 원고를 실제 remark/GFM/math→KaTeX→HTML 파이프라인으로 렌더링한다. KaTeX 오류·경고, 손상된 표의 열 수, 미치환 수식, 영어 원고의 한글 혼입을 검사한다. 문자열 검사를 내용 정확성의 유일한 근거로 삼지 않고 본문 예제를 생산 모델 및 독립 계산과 대조한다.

- 순수 qubit의 공분산 하한144/625와0인 교환자 하한을 구별하고, 순수 qutrit의 엄격한 slack1을 확인한다.
- Var(A)=0일 때 null/비유일 사영 계수와 최소 잔차1을 유지한다. 같은 상태의 서로 다른 ensemble이 분산 분해를 바꾸는 예도 확인한다.
- 같은 POVM의 세 장치, κ=0·3/5·1의 Bloch/교란 표, raw 오차4/5와 보정 오차16/9를 대조한다.
- Sharp-A 반복성과 같은 unsharp 기록 반복 확률(1+κ)/2를 구별하고, 피드백 뒤 반복 실패·η²=4를 확인한다.
- I/2의 상태 거리0과η(X)²=2, Choi rank1/4, εη 곱 하한의 실패와 보정된 왼쪽√2를 비교한다.
- 별도 수치 적분으로 Gaussian의 정규화·위치/운동량 분산·최소 곱ℏ/2를 검산한다. 비가환 유한 행렬의 교환자 trace0도 직접 계산해 정준 commutator의 유한 차원 장애를 설명한다.

기존768검사를 포함한785검사·타입·전체 lint를 확인한다. 수치 기반의 독립91사례/88유일 입력·정확78128대조·방향성23176대조와 참조 재생성 일치는 [두 수치 가족 검증](quantum-povm-instrument-foundation-status.md) 및 해당 runtime 기록에 보관한다. 새 원고 검사로 독립 수치 참조 수를 부풀리지 않는다.

## 남은 순서와 범위

다음은 두 입력 가족의 Worker 계약·오류·취소·교체·종료와 실제 native Worker 검증이다. 이어 고정 개념·자유 실험·시각화·예측·안내·퀴즈·저장·모션을 만들고, Interactive Book에 등록한 뒤 실제 읽기·한영·모바일·접근성·정적 출력을 검증한다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 교재 카드를 넣지 않는다.

이번 단계는 원고 준비이며 새 build·브라우저·homepage 등록·공개 배포·commit·push는 없다. 기존 수치·Worker·교수 화면·본문과4366개 정적 파일은 보존한다. 원래 주제의 검증 완료나 연구 깊이 완료로 세지 않는다. 일반 차원·무한 차원 정의역·일반 POVM·복합계·엔트로피와 시간 관련 불확정성·추정·연구 과제는 전체51영역509주제와 함께 남는다.

근거: `.runtime/learning/quantum-uncertainty-manuscripts/`의 `new-tests.log`, `combined-tests.log`, `rendering.json`, `typecheck.log`, `lint.log`, `verification.json`, `preservation.json`.
