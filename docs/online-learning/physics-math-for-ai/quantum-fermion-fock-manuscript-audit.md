# Fermion Fock·parity 원고 감사

2026-11-09. 원고는 `fermion-fock-parity-operator-algebra-chapter.{ko,en}.md`다. 두 번째 집필 원본이며 `content/`·단원 registry·정적 홈페이지 출력은 아직 바꾸지 않았다.

## 가정과 계산 주장

| 위치 | 독자가 확인할 내용 | 확인한 근거 |
| --- | --- | --- |
| 1…2절 | 직교정규 한 입자 모드·입자 이름 구별, 전체 좌표의 반대칭성, 진공 노름과 영벡터 | symbolic 교환 반대칭, 정규화 진공·두 생성·같은 생성의 영 결과 |
| 3…4절 | CAR와 앞부분 parity, n₀ 최상위, 오른쪽 연산자 먼저 | 독립 JW Kronecker의 모든 열, 실제 두 경로±01·naive 합2·잘못된 CAR residual4 |
| 5…6절 | 복소 계수·역순 adjoint, hopping/Pauli 계수1/2, 중간 점유 부호 | 전체 행렬의 XY/2 대응, h=i에서 gamma01=−i/2·E=1, long hop E=−1과 naive+1 |
| 7…8절 | pair 생성/소멸 순서·even number 중첩·보존과 sharpness | 실수/복소 pair eigenstate의±1/0분산, 실제 N/P commutator·N 분산, 잘못된 adjoint의 Hermiticity 차이 |
| 9…10절 | parity SSR와 number SSR 구별, pure/mixed 조건·formal 입력·twirl | symbolic2p_ep_o, 단일 plus·세 모드 cat의 실제 density와 twirl·odd 평균·purity·even 에너지 |
| 11절 | gamma/F 색인과 생성 pair의 켤레 순서, 낮은 통계의 비유일성 | gamma/F 전체 행렬, 같은 I/2·F=0인 두 pure even 상태의 N²=4/8·분산0/4 |
| 12…13절 | 모드 재배열의 부호·subsystem 정의·full/sector 최소 | 기존57검사의 점유 permutation 변환, one-mode H=−n의 직접 대각·even/odd 최소와 projector 무게 |
| 14…16절 | 노름·에너지 잔차와 물리 가정, 범위와 연구 주장 | 제품 정확 관측과 독립 H²·CAR·commutator, 여덟 해설·한영 수식/번호/링크 일치 |

## 원고에서 유지한 구별

- 직교정규 모드와 ordered creation convention을 먼저 선언한다. 공간 파동함수만 항상 반대칭이라고 하지 않으며 spin을 포함한 전체 한 입자 좌표를 사용한다.
- 입력 pure state의 parity 호환성과 even Hamiltonian의 parity 보존은 별도다. definite parity를 모든 혼합 density에 강제하거나 입자 수 superselection과 동일시하지 않는다.
- formal even–odd 순수 coherence와 twirl 혼합 상태를 함께 비교한다. odd 평균은 형식적 진단이며 선언한 even 물리 관측 대수의 직접 허용 측정으로 제시하지 않는다. 재정규화나 twirl을 입력 자동 수정으로 쓰지 않는다.
- gamma_ij=〈c_i†c_j〉를 고정하여 hopping 에너지를2Re(h_ij gamma_ij)로 계산한다. 반대 색인의 문헌식이나 Tr(h gamma)를 그대로 복사하지 않는다. F_ij=〈c_ic_j〉의 반대칭과〈c_i†c_j†〉=F_ji*도 유지한다.
- 전체 density와 연산자 평균만 제공하며 보통 qubit partial trace를 일반 fermionic mode 얽힘 계산으로 이름 붙이지 않는다. 큰 계·DMRG·유효 pairing 유도·전역 particle conservation의 연구 주장을 대신하지 않는다.

## 독립 유도와 자료 범위

`generate-fermion-fock-authoring-references.py`는 제품 모듈을 가져오지 않는다. SymPy JW Kronecker/H²·외적 density·실제 commutator·twirl과 symbolic pure coherence를 계산한다. sector 최소도 한 모드 대각 행렬에서 구한다. 일반 CAR 입력 범위는 기존45개 독립 준비·57개 수치 검사를 유지하며, 새 원고 검증과 같은 검사를 새로 센 것으로 혼동하지 않는다.

[Google Quantum AI 공식 설명](https://quantumai.google/openfermion/tutorials/jordan_wigner_and_bravyi_kitaev_transforms)의 CAR·Fock 기저·부호·JW 식을 확인했다. 외부 패키지 실행 실습을 집필 근거로 대체하지 않았다.

[Szalay 등의 원문](https://arxiv.org/html/2006.03087v4) §6.1의 parity 공간·projector, §6.2의 even/odd operator 대수·보존 성질, §3.4의 fermionic partial trace 정의와 embedding 관계를 확인했다. 논문 전체의 maps·locality·entanglement 분류를 모두 구현·검증했다고 주장하지 않는다. 본문의 숫자와 부호·반례는 위 독립 행렬로 직접 확인했다.

## 검증과 남은 관문

각 언어16절·17표시 수식·4표·8개 해설이다. 한국어 Hangul5730자, 영어3517단어이며 영어에 한글이 없다. 새11검사가 strict KaTeX/GFM·ket가 들어간 표의 열 수·양쪽 수식/문제 번호·본문 사례를 확인했다. 기존 fermion57·첫 spin 원고10의67개 회귀와 타입·lint도 통과했다. 기준 JSON 재생성은 SHA-256 바이트 그대로다.

구조·분량·언어 검사는 편집 관문이다. 실제 content loader·Worker·시각화·안내·quiz·저장·reader·접근성·키보드/터치·모션·생산 자산 관문은 후속이다. 첫 두 원고의 검증을 다체 원주제·전체 교육과정·연구 깊이 완료로 세지 않는다.
