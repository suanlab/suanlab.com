# Schmidt 원고의 정규화·방향·축퇴 검산

2026-11-09. 대상은 `schmidt-cuts-bond-rank-truncation`의 등록 전 상세 한영 원고다. 실제 content loader·Worker·reader의 검증을 대신하지 않는다.

## 가정과 직접 유도

직교정규 곱 기저의 비영 순수 상태를 사용하고 site0 최상위·C의 행이 left ket·열이 right ket임을 고정했다. 양쪽 partial trace를 성분으로 유도하여 ρL=CC†/N, ρR=Cᵀconj(C)/N을 구별한다. SVD의 오른쪽 ket은 V 열의 켤레다. 동일 스펙트럼만으로 물리 행렬의 방향을 증명하지 않는다.

Rank는 양의 가중치 multiplicity의 합이며 bond 용량과 다르다. Entropy는 순수 입력의 특정 양분할에 대한 값이다. At-most-rank 근사의 열공간 projector를 사용해 비정규화 Frobenius 최적 오차를 유도하고, Cauchy–Schwarz로 정규화 overlap 최적성도 연결한다. 정규화 거리에는 양의 overlap 위상 선택을 명시했다.

원래 정규화 상태·비정규화 유지 성분·재정규화 상태를 구별한다. Retained S는 fidelity **제곱**, 비정규화 오차 제곱은1−S, 정규화 거리 제곱은2(1−√S)다. 양의 축퇴 경계를 가르면 최적값은 정의되지만 상태는 비유일하다. Budget0의 영벡터는 정규화하지 않으며 support 이상의 budget에 가짜 양의 모드를 덧붙이지 않는다.

## 독립 기준과 핵심 예제

`scripts/learning/generate-schmidt-authoring-references.py`는 제품 코드를 가져오지 않고 SymPy exact eigenvectors·Gram–Schmidt 전체 고유공간·성분 partial trace·실제 절단 행렬을 계산한다. 제품의 polynomial projector와 다른 경로다. 정확한 식을250자리로 평가한 기준은 `scripts/fixtures/schmidt-authoring-reference.json`이다.

13준비는 unequal2/1의 budget0·1·4, Bell 축퇴 경계, diag(3,2,2,1)의 budget0…4, 복소 오른쪽 projector, 작은 양의 weight, 인접/crossing Bell pairs다. 모든 연속 cut을 대조한다.

- Unequal 예제의 weights4/5·1/5, purity17/25, 오류1/5와 정규화 거리2−4/√5를 구별한다.
- Diag(3,2,2,1)은 purity19/54이며 budget2에서만 반복2/9 group을 갈라 상태 선택이 불가능하다. Budget3은 전체 group을 유지해 선택이 가능하다.
- 복소 오른쪽 예제에서 ρR₀₁=−3i/10이다. CPᵀ는(2,2i) 행을 유지해 오차1/5, 잘못된 CP는 작은 행을 유지해 오차4/5다. 실제 절단 진폭을 대조한다.
- 정수 M=10⁶의 near-rank-deficient 행렬은 determinant−1·rank2다. 약6.25×10⁻²⁶의 weight를0으로 자르지 않는다.
- 인접 Bell pairs는 cut별 rank(2,1,2), crossing pairs는(2,4,2)다. 순서가 바뀌면 연속 cut이 가리키는 물리 부분계가 달라짐을 설명한다.

별도 해석 검산은 Bell의 서로 다른 최적 rank1 벡터, 잘못된 오른쪽 projection 오차4/5, bounded observable에 쓰는 pure-state trace distance²=ε를 포함한다. 한 cut 최적성을 순차·다중 cut MPS 압축의 전역 최적성으로 확대하지 않는다. 자동 압축·DMRG·새 혼합 입력 모드를 구현했다고 세지 않는다.

## 출처와 검사 범위

[Schollwöck §4.1.1](https://arxiv.org/html/1008.3477v2)의 식18…27과 재정규화 설명을 대조했다. 원문의 Gram 표기를 ket density로 그대로 복사하지 않고 본문의 성분 convention으로 다시 유도했다. 리뷰 전체나 연구 알고리즘 전체를 구현·검증했다고 주장하지 않는다.

새20검사에서 정확2561·방향성3589관측과4374개 구간 잔차를 대조한다. 기존84회귀(Schmidt39·이전 네 원고45), 타입·lint·기준 재생성 바이트 일치를 확인한다. Strict KaTeX/GFM에서 각 언어16절·16수식·4표·8해설을 렌더링하고 수식·문제 번호·참고 링크의 한영 일치와 영어 한글 부재를 확인했다. 한국어 Hangul5674자·영어3185단어다. 구조·분량은 학습 효과나 실제 browser 검증의 증거가 아니다.
