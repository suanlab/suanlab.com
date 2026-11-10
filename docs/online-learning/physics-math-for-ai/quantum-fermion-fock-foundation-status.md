# 다체계 두 번째 가족: Fermion Fock·parity 수치 기반

2026-11-09. `many-body·tensor network` 다섯 예정 단원 중 spin에 이어 **두 번째 수치 가족**을 구현·검산했다. 새 단원 등록은 아직 0개이며 기존 **566개 한영 단원·1,698개 안내, 원래 입문 308/362개 검증·54개 미완료**를 유지한다. 전체 51영역·509주제와 연구 깊이도 미완료다.

## 구현과 검산

`src/lib/learning-fermion-fock.ts`는 1…4모드·차원 2…16의 정확 복소 유리수 Fock 행렬을 만든다. 복소 hopping·pairing, onsite·밀도 상호작용, 일반 복소 순수 입력과 even/odd 제한을 다룬다. 모든 모드 쌍의 CAR·생성 순서·점유 부호, 에너지·분산·잔차·항별 계정, one-body/anomalous/입자 수 상관을 반환한다. 상세 표기·가정·입출력 범위는 [수학 계약](quantum-fermion-fock-contract.md)에 있다.

`scripts/learning/generate-fermion-fock-references.py`는 점유 부호를 구현한 제품 코드와 달리 **Jordan–Wigner Kronecker 행렬**로 45개 기준 준비를 만든다. H² 모멘트·실제 행렬 교환자·밀도 행렬의 parity twirl·γ와 I−γ의 모든 principal minor를 독립 계산한다. 정규화 좌표는 mpmath250자리로 비교한다. 입력 크기 경계와 1…4모드·세 sector의 일반 복소 준비를 포함한다.

**57개 새 테스트·기존 spin 61개 회귀**를 통과했다. 새 검사에서 정확 유리수 관측 **32,310개**, 방향성 좌표 **695개**를 대조했다. 정확 관측 수에는 영 잔차 인증도 포함한다. 기준 JSON을 두 번 생성하여 SHA-256 바이트 일치를 확인했고 타입 검사와 lint도 통과했다.

대표 검증은 다음과 같다.

1. 모든 모드 쌍·네 생성/소멸 선택의 CAR를 행렬 전체로 확인한다. 같은 모드의 두 생성은 영 결과이며 진공을 영벡터로 취급하지 않는다.
2. 점유한 중간 모드를 가로지르는 hopping은 E=−1인데 parity string을 버린 비교는 +1이다. 단순 spin 비트 교환으로 대체할 수 없음을 실제 에너지로 드러낸다.
3. 복소 pairing의 역순 adjoint와 F의 반대칭·켤레 방향을 확인한다. 입자 수가 sharp인 상태에서도 N̂ 보존은 깨질 수 있다. pairing은 N̂를 깨면서 P를 보존한다.
4. 형식적인 even–odd coherence와 parity twirl을 구별한다. 홀수 일차 평균이 모두 0인 cat도 superselection과 양립하지 않을 수 있다. 비교 twirl은 혼합 상태이며 원래 입력을 바꾸지 않는다.
5. 같은 γ와 F를 가진 고정 입자 수 상태와 even cat의 입자 수 분산은 각각 0과 4다. one-body 정보에서 전체 분포를 추정하지 않는다.
6. full Fock 바닥과 선택 parity 구역의 바닥이 다른 경우, 상수 H의 축퇴 공간·없는 gap, 비대각 영분산과 미계산 바닥 판정, 매우 작은 양의 분산을 확인한다.
7. 복소 배율·에너지 원점·Hamiltonian 부호/배율·복소 켤레·점유 permutation 부호를 포함한 모드 재배열의 불변성을 확인한다. 두 모드의 일반 복소 Hamiltonian은 기존 spin 모델에 독립 Pauli 계수를 조립하여 H·E·Var를 교차 대조한다.
8. 별도 74-field 입력 모듈의 소유권·동결 snapshot·+0·accessor 거부·inactive 항·sector 검사를 확인한다. caller getter를 실행하지 않는다.

근거는 `.runtime/learning/fermion-fock-foundation/verification.json`, 독립 기준 JSON·재생성 hash·검사 로그와 소스/근거 snapshot에 있다. 이번 단계는 **Node 수치 기반** 검증이다. Worker·교수 화면·실제 reader·생산 자산이 준비됐다는 근거로 세지 않는다. 이전 소스/근거와 현재 정적 출력은 별도 hash 비교로 보존한다.

## 이어갈 범위

다음은 boson Fock cutoff의 √n 계수와 경계 CCR, 이후 MPS 수축·gauge, Schmidt cut·절단이다. 이 세 수치 가족과 다섯 단원의 상세 한영 본문·Worker·개념·실험·예측·해설 안내·퀴즈·저장·Three.js/접근 가능한 표를 연결한다. 실제 Interactive Book reader와 생산 자산 검증 뒤에 등록·입문 완료를 검토한다.

[다섯 단원 설계](quantum-many-body-authoring-plan.md)와 [남은 입문 순서](remaining-introductory-queue.md)를 유지한다. 기존 홈페이지·미리보기 서버와 **도서 → Interactive Book** 연결을 유지하며 commit·push·공개 배포는 수행하지 않았다.
