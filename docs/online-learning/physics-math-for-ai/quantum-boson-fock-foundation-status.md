# 다체계 세 번째 가족: Boson raw-state cutoff 기반

2026-11-09. `many-body·tensor network`의 세 번째 수치 가족에서 **원시 상태의 유한 투영 대수와 관측**을 구현·검산했다. coherent 상태·tail 인증이 남으므로 이 가족 전체와 예정 단원은 아직 준비 완료가 아니다. 기존 **566개 한영 단원·1,698개 안내, 원래 입문 308/362개 검증·54개 미완료**, 전체 51영역·509주제와 연구 깊이를 유지한다.

## 구현한 범위

`learning-boson-fock.ts`는 1…2모드·모든 20개 cutoff 조합(각 0…3)에서 정확 생성·소멸 행렬과 유한 Hamiltonian을 조립한다. Gaussian-integer 복소 준비, onsite·self/cross 밀도·복소 hopping·drive, 평균·분산·잔차, 국소 밀도·quadrature·입자 수·경계 교환자를 다룬다. 유한 대각 스펙트럼의 바닥 판정을 무한 공간의 바닥으로 표현하지 않는다.

`learning-boson-algebraic.ts`는 √2·√3·√6을 포함하는 **Q(√2,√3)의 정확 네 좌표**를 유지한다. exact sign과 영 판정, field inverse는 유리수 연산과 대수적 conjugation으로 결정한다. 표시용 구간에는 계수 크기에 따른 추가 비트를 사용한다. 큰 Pell 계수의 상쇄에서도 작은 부호와 절대 정밀도를 보존했다. [수학·입출력 계약](quantum-boson-fock-contract.md)에 기저·계수·관측·해석 범위를 명시했다.

## 독립 검증

두 Python 생성기는 제품 코드를 가져오지 않는다.

- `generate-boson-fock-references.py`: SymPy 국소 ladder·Kronecker 행렬, H², 직접 partial trace, 실제 교환자와 symmetrized covariance, mpmath250자리 정규화로 **33개 준비**를 검산한다. 모든 cutoff 조합과 top/vacuum·복소 phase·최대 정수·작은 양의 분산을 포함한다.
- `generate-boson-algebraic-references.py`: SymPy radsimp·정확 부호·245자리 수치를 사용한 **52개 field 준비**다. 일반 네 좌표의 합·차·곱·나눗셈과 큰 Pell 계수의 상쇄·영 분자를 포함한다.

원시 상태와 field 검증, 물리 불변성·입력 검사의 **95개 새 테스트**와 기존 spin/fermion **118개 회귀**를 통과했다. 새 검사에서 정확 관측 **204,044개**와 방향성 좌표 **2,387개**를 대조했다. 정확 수에는 field 계수와 영 잔차도 포함한다. 두 기준 JSON을 다시 생성해 SHA-256 바이트 일치를 확인했으며 타입 검사와 lint도 통과했다. 정확 계수·관측의 endpoint 일치, 방향성 좌표 대조 여유 10⁻²⁰⁵·max(1,|x|)와 폭 10⁻¹²⁵·max(1,|x|)를 요구한다. 검사 결과의 수·재생성 hash와 타입·lint·기존 회귀 근거는 `.runtime/learning/boson-fock-foundation/verification.json`에 기록한다.

대표 검증은 다음과 같다.

1. 모든 cutoff에서 [a,a†]=I−(C+1)P_top과 trace=0을 정확 확인한다. C=0은 quadrature와 불확정성 경계가 모두 0이다. 무한 CCR과 같다고 강제하지 않는다.
2. n=1·2에서 √2·√3 ladder 계수, 두 모드 hopping의 √6 향상과 cross-mode 교환자의 영을 확인한다.
3. vacuum의 Var(X)=Var(P)=1/2와 top n=3에서의 교환자 −3·불확정성 경계 9/4를 구별한다. 복소 phase는 P 방향까지 대조한다.
4. drive의 입자 수 비보존과 sharp한 현재 진공, cutoff0에서 operator가 사라지는 경우를 확인한다. Hamiltonian 계수를 바꿔도 입력 상태는 자동 진화하지 않는다.
5. H=−n의 유한 바닥, 상수 H의 전체 축퇴 공간·없는 gap, 비대각 영분산과 미계산 바닥, 작은 양의 분산을 구별한다.
6. 복소 원시 상태 배율·에너지 원점·켤레, 서로 다른 cutoff 두 모드의 교환과 hopping 켤레 변환을 검산한다. boson 모드 교환에 fermion 부호를 넣지 않는다.
7. 52개 소유 필드의 snapshot·동결·+0·inactive 항·cutoff·accessor/prototype 거부와 getter 미실행을 확인한다.

## 다음 순서

먼저 coherent builder의 정확 유리수 α·μ≤16 계약과 αⁿ/√n! 계수, 무한/절단/재정규화 상태 및 버린 Poisson 확률의 인증 구간을 만든다. 이 관문 뒤에 세 번째 수치 가족의 준비 완료를 검토한다. 이어 MPS 수축·gauge와 Schmidt cut·절단, 다섯 상세 한영 본문·Worker·개념/실험/안내/예측/퀴즈/저장·Three.js·접근 가능한 표와 실제 reader를 순서대로 연결한다.

이번 검증은 Node 수치 기반이며 새 reader 등록은 0개다. 기존 홈페이지와 **도서 → Interactive Book** 경로·미리보기 서버를 유지한다. commit·push·공개 배포는 수행하지 않았다. [다섯 단원 설계](quantum-many-body-authoring-plan.md)와 [전체 남은 입문 순서](remaining-introductory-queue.md)를 유지한다.
