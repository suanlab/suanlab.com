# 다체계 세 번째 가족: Coherent 투영과 Poisson 인증

2026-11-09. boson raw-state 기반에 **coherent builder·절단·재정규화·Poisson tail 인증**을 추가했다. 다섯 예정 수치 가족 중 spin·fermion·boson의 세 가족을 준비했다. 아직 새 reader 등록은 0개이며 기존 **566개 한영 단원·1,698개 안내, 원래 입문 308/362개 검증·54개 미완료**를 유지한다. 전체 51영역·509주제와 연구 깊이도 미완료다.

## 구현과 결과

`learning-boson-coherent-input.ts`는 26개 소유 정수와 유리수 α, 모드별 μ≤16을 정확 검사한다. `learning-boson-coherent.ts`는 αⁿ/√n!를 Q(√2,√3)에서 만들고, 무한 급수·비정규화 투영·재정규화 유한 상태를 구별한다. 모드별/공동 retained와 tail, 원래 무한 상태와의 overlap·fidelity 제곱·서로 다른 거리 제곱을 제공한다.

유한 관측은 기존 boson engine의 정확 field-valued 준비 경로로 계산한다. 파생 상태에 가짜 Gaussian-integer 입력을 반환하지 않는다. 무한 coherent의 a 고유상태 조건이 유한 cutoff에서 깨지는 잔차 μp_top과 실제 경계 교환자를 검산한다. 무한 coherent 에너지는 같은 normally ordered 식의 기대값이며 무한 바닥 인증으로 해석하지 않는다. 세부 식·정밀도와 출력 의미는 [수학 계약](quantum-boson-coherent-contract.md)에 있다.

**새 45개 테스트·기존 213개 회귀**와 타입 검사·lint를 통과했다. 새 독립 기준 **32개 준비**에서 정확 관측 **180,379개**, 방향성 값 **5,221개**를 대조했다. 정확 수에는 유리수/field 계수와 영 잔차가 포함한다. 기존 회귀는 raw-state boson95·fermion57·spin61이다. coherent 기준을 다시 생성해 SHA-256 바이트 일치를 확인했다. import 가능한 함수로 정리한 기존 raw-state 생성기도 다시 실행하여 33개 기준 JSON이 바이트 그대로임을 확인했다.

## 독립 기준이 확인한 것

`scripts/learning/generate-boson-coherent-references.py`는 제품 코드를 가져오지 않는다. SymPy coherent 급수·정확 상태, 별도 국소 ladder/Kronecker·H²·partial trace·symmetrized covariance와 mpmath250자리 지수·incomplete gamma를 사용한다. 제품의 지수 Taylor 구현과 독립이다.

1. 모든 20개 cutoff 조합과 일반 복소 α·drive·hopping·상호작용을 대조한다. 모드별 μ=16·공동 μ=32와 최대 정수 계수도 확인한다.
2. α=1,C=3에서 정확 계수 [1,1,√2/2,√6/6], Z=8/3, p_top=1/16, 평균 점유15/16과 a 고유상태 잔차1/16을 확인한다.
3. 진공은 모든 cutoff에서 tail=0이다. α≠0·cutoff0에서는 정규화 상태가 진공이어도 원래 무한 coherent 상태와 다르고 잔차가 남는다.
4. μ=10⁻¹²,C=3의 약4.17×10⁻⁵⁰ 양의 tail과 두 작은 모드의 공동 tail을 인증한다. 부동소수점의 1−CDF가 0처럼 보일 수 있는 경우에도 영으로 바꾸지 않는다.
5. 비정규화 거리 제곱=tail, 재정규화 fidelity 제곱=retained, 거리 제곱=2(1−√retained)를 구별한다. 조건부 유한 확률의 합은 정확히1이다.
6. 두 모드 직사각형 cutoff의 retained 곱, cutoff 증가의 단조성, 유리수 α 표현의 동치와 복소 켤레의 위상·quadrature 방향을 확인한다. 결합 H가 있어도 준비한 곱 상태의 국소 purity는1이다.
7. cutoff1의 coherent 투영과 비례하는 Gaussian-integer raw 상태의 H·에너지·분산·밀도·γ·국소 관측을 교차 대조한다.
8. 정확 μ 경계·비활성 모드·소유 snapshot·getter 미실행과 prepared bridge의 영 상태·차원·잘못된 유리수·비유리수 노름 거부를 확인한다.

근거는 `.runtime/learning/boson-coherent-foundation/verification.json`, 독립 JSON·재생성 hash·로그·소스/근거 manifest에 있다. 기존 raw model과 독립 생성기 두 파일의 의도한 확장, 진행 문서 외에는 이전 snapshot을 보존했다. 정적 출력·홈페이지와 미리보기 서버도 유지한다. 이번 검증은 **Node 수치 기반**이며 Worker·교수 화면·실제 reader·생산 자산을 검증한 것으로 세지 않는다.

## 이어갈 순서

다음은 MPS의 실제 수축·환경·가역 gauge와 영 수축, 이어 Schmidt cut·rank·가중치·절단 오차·축퇴다. 남은 두 수치 가족을 검산한 다음 다섯 단원의 상세 한영 원고, Worker, 고정 개념·자유 실험·예측·안내·해설 퀴즈·저장·Three.js와 접근 가능한 표를 연결한다. 실제 **도서 → Interactive Book** reader와 생산 자산 검증 뒤 등록·원주제 입문 완료를 검토한다.

[전체 다섯 단원 설계](quantum-many-body-authoring-plan.md)와 [남은 입문 순서](remaining-introductory-queue.md)를 유지한다. commit·push·공개 배포는 수행하지 않았다.
