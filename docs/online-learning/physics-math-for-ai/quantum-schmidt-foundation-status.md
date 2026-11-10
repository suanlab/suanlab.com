# 다체계 다섯 번째 가족: Schmidt cut·최적 절단 기반

2026-11-09. `many-body·tensor network`의 마지막 수치 가족인 **모든 연속 cut의 밀도·가중치·rank·entropy·최적 절단**을 구현·검산했다. 다섯 수치 가족이 준비됐지만 원고·교수 화면·reader는 아직 미등록이다. 기존 **566개 한영 단원·1,698개 안내, 원래 입문308/362개 검증·54개 미완료**, 전체51영역·509주제와 연구 깊이를 유지한다.

## 구현과 기준

`learning-schmidt.ts`는2…4개 qubit·35개 소유 정수 입력에서 모든 연속 cut을 계산한다. 전체 좌우 물리 밀도·exact rank/purity를 유지하고, 작은 쪽(최대4차원)의4096비트 Sturm roots·exact multiplicity와2048비트 spectral projector를 계산한다. 로그는768비트, 출력은 바깥512비트다. [수학 계약](quantum-schmidt-contract.md)에 ket 밀도의 전치·rank budget·nullable 상태와 정밀도를 명시했다.

각 budget에서 버린 가중치·최적 비정규화 오차와 재정규화 fidelity/거리를 구별한다. 양의 축퇴 group을 가르는 cutoff에서는 오차 값은 주되 `chosen=null`로 둔다. 특정 projector가 정해지면 실제 절단 계수·원래 입력에 대한 진폭·재정규화 진폭과 오차/노름/직교성 계정을 계산한다. budget0의 영벡터를 정규화하지 않는다.

`scripts/learning/generate-schmidt-references.py`는 제품 코드를 가져오지 않는다. SymPy exact 밀도·rank·characteristic·root multiplicity와 mpmath250자리 Hermitian eigenvectors의 **전체 group projector**를 사용한다. 제품의 polynomial projector와 달라 위상·색인·선택 오류를 독립 검산할 수 있다.26개 준비를 생성하고 재생성 바이트 일치를 확인한다.

## 주요 검증

1. 2…4 qubit의 모든 연속 cut과 일반 복소 상태를 확인한다.4-qubit의 smallerSide는 left/left/right이고 차원은2/4/2다. 더 큰 쪽의 영 고유값 수를 구별한다.
2. 중간 계수 diag(3,2,1,0)에서 가중치9/14·4/14·1/14·0, rank2 절단 오차1/14, 재정규화 fidelity 제곱13/14와 서로 다른 거리를 확인한다. budget0…4도 대조한다.
3. GHZ rank1과 네 겹 축퇴의 partial cutoff는 최적 오차를 공유하지만 상태를 선택하지 않는다. 두 양의 축퇴 group의 budget1/2/3으로 group 전체 유지와 내부 분할을 구별한다.
4. 오른쪽 복소 밀도와 `C Pᵀ` 반례에서 실제 절단 오차가1/5임을 확인한다. 잘못된 Gram 방향으로 같은 스펙트럼만 맞추는 오류를 잡는다.
5. 정수 determinant가−1인 거의 rank deficient한2×2 계수의 작은 양의 가중치와 exact rank2를 유지한다. 서로 가까운 가중치를 tolerance로 합치지 않는다.
6. 전체 양의 rank를 넘는 budget과 budget0, 최대 정수 복소 곱 상태, 영 가중치의 entropy 기여를 확인한다. nullspace의 임의 벡터는 반환하지 않는다.
7. 복소 상태 배율·국소 Hadamard·qubit 순서 반전에서 스펙트럼/entropy/절단 값이 보존된다. rank는 이전 MPS 수축과 bond 하한으로도 교차 대조한다.
8. 35개 입력의 소유 snapshot·+0·동결·getter 미실행과 비영/차원/cut/budget 검사를 확인한다. 실제 projector·절단 잔차는 Number가 없는 성분도 인증 구간으로 대조한다.

**새39개 검사·기존388개 회귀**, 타입 검사와 lint를 통과했다. 새 검사에서 정확 관측 **9,496개**, 방향성 기준 관측 **8,613개**, 구간 잔차 인증 **10,261개**를 확인했다. 독립26개 기준의 SHA-256 재생성 바이트 일치도 확인했다. 결과와 보존 근거는 `.runtime/learning/schmidt-foundation/verification.json`에 기록한다. 기존 회귀는 spin·fermion·boson raw/coherent·MPS와 공유 root/interval/log helper를 포함한다. 이번 단계는 **Node 수치 기반**이며 Worker·교수 화면·실제 reader·생산 자산 검증으로 세지 않는다.

## 이어갈 순서

이제 다섯 단원의 **상세 한영 원고**를 spin→fermion→boson→MPS→Schmidt 순서로 작성한다. 가정·색인·직접 유도·계산 표·직관·일반 대학원생용 예제·실패 반례·AI 연결·해설 연습을 갖춘다. 다음은 Worker, 고정 개념·자유 실험·예측·세 안내·해설 퀴즈·저장·Three.js와 접근 가능한 표, 실제 **도서 → Interactive Book** reader와 생산 자산 검증이다. 이 단계들이 갖춰진 뒤 새 단원 등록·원주제 입문 완료를 검토한다.

기존 홈페이지와 미리보기 서버를 유지하고 commit·push·공개 배포는 수행하지 않았다. [다섯 단원 설계](quantum-many-body-authoring-plan.md)와 [남은 입문 순서](remaining-introductory-queue.md)를 유지한다. 다섯 수치 가족 준비를 전체 교육과정·영역 감사·연구 깊이 완료로 세지 않는다.
