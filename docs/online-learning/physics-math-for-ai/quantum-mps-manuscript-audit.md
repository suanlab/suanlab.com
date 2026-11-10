# MPS 수축·gauge 원고의 색인·가정·검산

2026-11-09. `matrix-product-state-contraction-gauge`의 등록 전 한영 원고를 대상으로 한다. 실제 content loader·Worker·reader의 검증으로 확대하지 않는다.

## 설명의 범위

열린 경계 D₀=Dₙ=1, 물리 bit0 최상위, 출력 A[site][physical][left][right]를 고정했다. 가상 색인은 물리 입자가 아니라 합산하는 표현 좌표다. 활성 tensor 항 수는 gauge 중복을 제거한 물리 상태 공간의 차원이 아니다. Bond1 곱 상태와 GHZ의 두 경로를 실제 행렬로 전개한다.

Cut의 계수는 C=LB이며 coefficient rank≤bond dimension이다. N>0일 때만 이를 Schmidt rank로 해석한다. 오른쪽 환경은 Rₐᵦ=Σᵧconj(Bₐᵧ)Bᵦᵧ이므로 노름은 ΣₐᵦEₐᵦRₐᵦ=Tr(ERᵀ)다. 같은 크기의 환경과 물리 reduced density를 서로 바꿔 쓰지 않는다.

G는 가역이며 이웃에 AG와 G⁻¹B를 함께 적용한다. 상태·rank·물리 관측과 환경 closure는 보존되지만 텐서 길이와 가상 환경은 달라질 수 있다. 실수 shear만으로 복소 covariance를 검증했다고 하지 않는다. QR/SVD에 의한 canonicalization과 Gram 진단을 구별하며 rank 부족으로 직사각 인자가 되는 경우를 설명한다.

## 독립 기준과 대표 예제

`scripts/learning/generate-mps-authoring-references.py`는 제품 모듈 없이 모든 가상 경로를 명시적으로 합산한다. 별도의 prefix/suffix Gram·전체 벡터 partial trace·복소 Pauli 행렬로 `scripts/fixtures/mps-authoring-reference.json`을 만든다. 일곱 준비는 실수/복소 GHZ gauge, 실수/복소 영 수축, 위상 곱 상태, 중복 bond4 곱 상태와 한 site 상태다.

- GHZ₃의 원시 진폭은000·111에서1, 나머지0이며 N=2·두 cut rank2다. G=[[2,1],[0,1]]에서 좌표 길이는6→19/2, 첫 환경은 E′=[[4,2],[2,2]], R′=[[1/2,−1/2],[−1/2,1]]로 변한다. Closure는2다.
- L=(1,i), B=(1,i)ᵀ는 amplitude0이지만 E=R=[[1,i],[-i,1]]이다. 올바른 closure0과 잘못된 Tr(ER)=4를 실제 복소 곱으로 대조한다. 영 수축에는 normalized physical output이 없다.
- 오른쪽 |0〉+i|1〉의 density upper-right는−i/2이며 Y=+1이다. 일반 복소 shear도 전체 진폭·국소 물리 정보와 inverse/covariance 계정을 보존한다.
- 세 site에서 bond2 GHZ의 활성 항은16, bond4 중복 곱은48이며 dense amplitudes는8이다. 추가 가상 용량이 실제 rank·얽힘·압축의 증거가 되지 않는다.

생성기는 GHZ와 같은 비율 비간섭 혼합의 모든 pair Pauli 평균이 같고 XXX가1/0임도 별도 dense 행렬로 증명한다. 혼합과 three-body XXX는 이 원고의 **해석적 비교**이며 현재 pure-MPS/pair 관측 입력의 새 기능이 아니다. √2를 흡수하는 정규화와 QR 식도 설명이며 자동 canonicalization 구현을 주장하지 않는다.

## 출처와 검증 한계

[Schollwöck 리뷰](https://arxiv.org/html/1008.3477v2)의 식57, §4.2.1, §4.4.1을 확인해 gauge·수축·정규형의 배경을 대조했다. 본문 색인 convention과 예제는 직접 유도했다. DMRG·MPO·최적화·시간 진화나 리뷰 전체를 구현했다고 세지 않는다.

새13검사에서 정확 관측2296개와 한영 strict KaTeX/GFM 렌더링을 대조했다. 기존73회귀(MPS41·이전 원고32), 타입·lint와 기준 JSON 재생성 바이트 일치를 확인한다. 각 언어16절·14수식·4표·8해설이며 영어3224단어·한국어 Hangul5853자다. 구조·분량 검사는 학습 효과의 증거가 아니다. 이 단계는 실제 loader·Worker·학습 화면·생산 자산 검증을 포함하지 않는다.
