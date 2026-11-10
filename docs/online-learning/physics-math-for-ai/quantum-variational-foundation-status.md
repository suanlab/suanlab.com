# Quantum variational principle: 첫 수치 가족 준비

2026-11-08. 일반1…4차원 복소 Hermitian Hamiltonian의 `finite-state-rayleigh-energy-variance` 수치 기반을 구현했다. 양자 변분 원리 전체·교재 등록·브라우저 검증 완료로 표시하지 않는다. 현재564개 한영 단원·1692개 안내, 원래 입문 주제307/362검증·55미완료를 유지한다.

## 구현한 내용

36개 소유 유계 정수에서 정확 에너지·이차 모멘트·분산·잔차·방향1/2차 미분·유한 경로점을 계산한다. 일반 signed 스펙트럼의 square-free/Sturm 인증, 정확 중복도와 전체 고유공간 사영, 바닥 공간의 중첩·누출·초과 에너지·gap 기반 누출 상한을 반환한다.

들뜬 고유상태의 영분산과 바닥 상태를 구별한다. 축퇴된 바닥 공간에서는 한 벡터가 아닌 공간 전체에 대한 중첩을 쓴다. 상수 H의 들뜬 gap과 영벡터 경로의 모멘트는 `null`이다. 확률0/1의 알려진 범위와 구간을 교차하며 상태 재정규화·작은 양수/gap의 임계값 제거는 하지 않는다.

코드는 `src/lib/learning-rayleigh{,-input}.ts`와 `learning-quantum-variational-math.ts`다. 상세 입력·수학·정밀도와 한계는 [수치 계약](quantum-rayleigh-source-audit.md)에 기록했다.

## 검증한 근거

- 독립 Python/SymPy/mpmath250자리 기준56개·56유일 준비, 정확 유리수 대조820건·방향성 관측3686건을 통과했다. fixture의 최대 상대 구간 폭은10⁻¹²⁵, 기준 대조 오차는10⁻²⁰⁵다.
- 새65개와 기존 root4개, 합계69개 검사·타입·전체 lint를 통과했다. 고유공간 완전성·멱등성·Hermitian 성질·고유공간 방정식·trace/rank 및 확률·모멘트 재구성도 구간으로 확인했다.
- 기존 Hilbert·준비 불확정성·밀도·텐서곱·혼합 얽힘 회귀366개를 통과했다. 기존 소스와 생산 reader는 수정하지 않았다.
- 최대 입력 계수, 가까운 구분된 네 고유값, 미소 양의 분산, 위상/크기 방향의 평탄성, 음수 Hamiltonian 배율에서의 순서 반전, 잘못된/상속/accessor 입력을 확인했다. 참고 자료를 다시 생성해 byte 동일성도 확인했다.

근거는 `.runtime/learning/quantum-variational-foundation/`의 `verification.json`, `reference-checks.json`, `tests-expanded.log`, `previous-quantum-regression.log`, `typecheck-final.log`, `lint-final.log`, `reference-reproduction.json`, `preservation.json`이다.

## 다음 순서와 미완료 범위

다음은 `nonorthogonal-ritz-subspace-certificates`다. 정확 Gram 양성/열독립성·일반화 고유값·Galerkin 잔차·좌표 변경·공간 포함·조건 문제를 구현하고 독립 검증한다. 이후 상세 한영 원고·Worker·시각화·안내·퀴즈·저장·실제 reader와 정적 출력을 순서대로 연결한다. 진입은 **도서 → Interactive Book**으로 유지한다.

전체51영역509주제·영역 감사·증명과 연구 깊이·공용 그림 후속은 미완료다. 새 계산기는 아직 메뉴·registry·검색에 등록하지 않았다. 기존 정적 미리보기와 읽기 주소를 유지하며 공개 배포·commit·push는 하지 않았다.
