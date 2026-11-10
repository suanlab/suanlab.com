# 다체계 네 번째 가족: MPS 수축·gauge 기반

2026-11-09. 다섯 예정 수치 가족 중 **MPS의 일반 복소 tensor 수축·환경·정규형 진단·가역 gauge**를 구현·검산했다. 새 단원 등록은0개이며 기존 **566개 한영 단원·1,698개 안내, 원래 입문308/362개 검증·54개 미완료**를 유지한다. 전체51영역·509주제와 연구 깊이도 미완료다.

## 모델과 독립 기준

`learning-mps.ts`는1…4개 qubit, 내부 bond1…4의 일반 복소 유리수 열린 MPS를 계산한다. 전체 진폭·prefix, 좌우 환경과 모든 cut의 노름 closure, 국소 밀도·Bloch·purity·모든 쌍의 아홉 Pauli 상관을 반환한다. 한 bond의 exact scale/complex shear와 inverse를 이웃 tensor에 적용하고, 전체 상태 보존과 환경 covariance를 실제 계산한다. [수학 계약](quantum-mps-contract.md)에 색인·공간·nullable 상태와 검증 범위를 명시했다.

`scripts/learning/generate-mps-references.py`는 제품 코드를 가져오지 않는다. 제품의 transfer recurrence 대신 **virtual 색인의 모든 경로를 명시적으로 합산**한다. prefix/suffix 계수의 직접 Gram과 전체 물리 상태의 dense Pauli·partial trace, SymPy rank·일반 matrix inverse를 사용한다. 정규화 좌표는 mpmath250자리다. 일반 복소·서로 다른 bond·극소/최대 크기·영 수축을 포함한 **29개 독립 준비**를 만들고 재생성 바이트 일치를 확인한다.

## 주요 검증

- 가능한 열린 bond 차원 **85개 조합**에서 알려진 곱 상태·노름·rank·물리 관측을 확인한다. 독립 기준의 일반 복소 profile20개는 모든 site/bond 크기와 비균일 차원도 포함한다. 모든85조합에 일반 복소 독립 기준이 있다고 주장하지 않는다.
- GHZ₃의 N=2, 국소 purity1/2·연결 ZZ=1·cut rank2를 확인한다. G=[[2,1],[0,1]] 뒤 tensor 좌표 길이는6→19/2지만 전체 진폭은 정확히 같다.
- 복소 shear에서 E′=G†EG와 R′=conj(G⁻¹)R(G⁻¹)ᵀ의 실제 잔차가0이다. 전체 세 bond와 네 pivot row에서 검사한다. inverse gauge를 다시 적용하면 원래 tensor까지 회복한다.
- 각 tensor는 비영인데 전체 상태가 상쇄되는 경우, 모든 tensor가0인 경우를 확인한다. 원시 대수는 남지만 정규화 물리 출력·Schmidt rank는 null이다.
- 오른쪽 Y=+1 위상과 국소 partial trace의 부호, tensor 복소 켤레와 한 tensor의 배율 효과를 확인한다. GHZ 관측은 기존 dense spin 모델과도 교차 대조한다.
- bond4의 곱 상태 rank1, canonical/isometry 검사와 상태 정규화의 차이, tensor 항 수가 dense 상태보다 큰 예를 확인한다. 압축과 runtime 보장으로 부르지 않는다.
- 분모10⁶을 네 site에 적용한 최소 진폭 규모의 비유리수 정규화, 최대 정수 tensor/gauge를 검산한다. 영·rank 판정에는 tolerance를 쓰지 않는다.
- 269개 소유 필드의 descriptor snapshot·동결·+0·inactive 항·gauge 범위와 getter 미실행을 확인한다. factory도 반복 전에 sites를 검사한다.

**새41개 테스트·기존258개 회귀**, 타입 검사와 lint를 통과했다. 새 검사에서 정확 관측 **26,969개**, 방향성 정규화 좌표 **782개**를 대조했다. 독립29개 기준의 재생성 SHA-256 바이트 일치도 확인했다. 결과와 보존 근거는 `.runtime/learning/mps-foundation/verification.json`에 기록한다. 정확 관측 수에는 영 잔차 인증도 포함한다. 기준 대조 여유10⁻²⁰⁵·max(1,|x|), 구간 폭10⁻¹²⁵·max(1,|x|)를 요구한다. 이번 단계는 **Node 수치 기반**이며 Worker·교수 화면·실제 reader·생산 자산 검증은 후속이다.

## 다음 순서

다음은 다섯 번째 수치 가족인 Schmidt cut·rank·가중치·절단 오차와 축퇴다. 그 뒤 다섯 단원의 상세 한영 원고·Worker·고정 개념/자유 실험/예측/세 안내/해설 퀴즈/저장·Three.js와 접근 가능한 표를 연결한다. 실제 **도서 → Interactive Book** reader·정적 출력·생산 자산을 검증한 뒤 등록·원주제 입문 완료를 검토한다.

기존 홈페이지·미리보기 서버를 유지하며 commit·push·공개 배포는 수행하지 않았다. [다섯 단원 설계](quantum-many-body-authoring-plan.md)와 [남은 입문 순서](remaining-introductory-queue.md)를 유지한다. 수치 가족 네 개의 준비를 전체 교육과정이나 연구 깊이 완료로 세지 않는다.
