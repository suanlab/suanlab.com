# Quantum variational principle: 두 수치 가족 준비

2026-11-09. 일반 복소 비직교 시험 공간의 `nonorthogonal-ritz-subspace-certificates`를 구현·검산했다. 앞선 [Rayleigh 에너지 가족](quantum-variational-foundation-status.md)과 함께 두 수치 기반이 준비됐다. 상세 한영 본문·Worker·학습 화면·실제 교재 등록은 남아 있으며564개 한영 단원·1692개 안내, 원래 입문 주제307/362검증·55미완료를 유지한다.

## 구현한 내용

전체 차원1…4, 독립적인 시험 열1…n의 일반 복소 기저와 Hermitian Hamiltonian을 받는다. 63개 소유 유계 정수에서 정확 Gram/역·투영 Hamiltonian·일반화 특성 다항식·물리 공간 사영·선택한 상태의 에너지/분산·전체/평면 안/평면 밖 잔차를 계산한다.

각 Ritz 고유공간의 물리 사영과 Galerkin 잔차, 실제 기저 열 prefix의 최소 에너지, 전체 스펙트럼과의 interlacing을 반환한다. 복소 shear와 비영 실수 scale로 좌표를 바꿔도 상태·공간·특성 다항식이 유지되는지 정확하게 검사한다. Gram 조건수는 좌표 민감도를 보여주며 계수 크기를 물리 확률이라고 부르지 않는다.

축퇴 공간은 전체 사영으로 다룬다. Ritz 정류와 전체 공간의 고유상태를 구별하며, 수치 고유벡터를 계산하는 기능은 제공하지 않는다. 상세 계약은 [수치 감사](quantum-ritz-source-audit.md), 학생을 위한 손계산은 [공간·좌표·잔차 예제](quantum-ritz-worked-example-notes.md)에 기록했다.

## 검증한 근거

- 독립 SymPy/mpmath400자리 기준41개·41유일 준비가 모든10개 n×k 조합을 포함한다. 정확 행렬·모멘트·좌표/공간 비교와 고정밀 사영·Ritz/Gram/전체 스펙트럼을 대조했다. 검사 전체에서 정확 유리수 검산9040건·방향성 관측4208건을 확인했다. 기준 비교 허용 오차는10⁻²⁰⁵, 사례별 출력 구간 폭 기준은 max(1,|기준|)×10⁻¹²⁵다. 두 수를 모든 허용 입력의 일률적인 폭 보장으로 확대하지 않는다.
- 새 Ritz/구간 검사53개·타입·전체 lint를 통과했다. 별도 구간 연산 검사에는 부호 있는 유리수 연산9240건, 복소 켤레·행렬 항등식·잘못된 분모·영점 포함 나눗셈·확률 범위 검사가 포함된다. 기준 파일을 재생성해 byte 동일성을 확인했다.
- 4096비트에서 기존 재귀 근 분리의 호출 스택 초과를 발견해 명시적 작업 스택으로 수정했다. 정밀도·왼쪽 우선 탐색·dyadic 근 처리·Sturm 개수는 유지했다. 10⁻³⁰⁰ 간격과 중복 근의 새4096비트 검사를 포함해 기존 Rayleigh/root/순수 텐서/혼합 얽힘213개 검사를 통과했다.
- 최대 복소 입력·매우 나쁜 Gram 조건수·정확 rank 결손·동일 평면의 좌표 변경·전체 공간·상수 Hamiltonian·Ritz 정류이지만 전체 고유상태가 아닌 사례를 확인했다. 동일 계산 안의 완전히 같은 특성 다항식은 재사용하며, 이를 적용한 최종 코드에서도53개 검사를 다시 통과했다.
- 손계산 예제의 최소 에너지(4−√7)/3, 평면 밖 잔차 방향과 복소 기저 변경을 별도 SymPy로 검산했다. 기존 정적 출력과 공개 사이트에는 새 계산기를 등록하지 않았다.

근거: `.runtime/learning/quantum-ritz-foundation/`의 `verification.json`, `tests-final.log`, `shared-root-regression.log`, `reference-checks.json`, `reference-reproduction.json`, `interval-tests-final.log`, `hand-example-check.json`, `lint-final.log`, `typecheck-final.log`, `preservation.json`.

## 다음 순서와 전체 범위

다음은 두 가족의 상세 한영 원고다. 일반 대학원생이 따라갈 수 있도록 가정·직관·직접 유도·해설 예제·문제를 작성하고, 이후 Worker 취소/검증·개념 그림·자유 실험·예측·안내·퀴즈·저장·실제 읽기와 정적 출력을 연결한다. 진입은 **도서 → Interactive Book**으로 유지한다.

전체51영역509주제·영역별 감사·증명과 연구 깊이·공용 그림 후속은 미완료다. 양자 변분 원리 원래 주제도 아직 검증 완료로 세지 않는다. 공개 배포·commit·push는 하지 않았다.
