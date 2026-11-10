## 연산을 줄이는 것과 동시에 실행하는 것

희소성은 0과 곱하는 작업을 없애고 병렬성은 독립 작업을 동시에 실행합니다. 둘은 같은 최적화가 아닙니다. 이 단원은 대각 2, 선택한 off-diagonal band -1인 행렬에 대해 실제 dense 곱, CSR 곱, balanced reduction task 그래프를 모두 계산합니다. 파랑·주황은 각각 CSR와 dense 출력이고 표에는 실제 스케줄로 얻은 출력도 남깁니다.

CSR는 row pointer, column index, value로 nonzero를 저장합니다. 각 row의 범위는 rowptr[i]부터 rowptr[i+1] 직전까지입니다. 교육 모형은 비교를 위해 dense 행렬과 trace도 생성하므로 화면 전체 준비 비용을 O(nnz)라고 부르지 않습니다. 아래 core multiplication의 비용과 setup을 분리합니다.

## 실제 작업 DAG 만들기

각 nonzero product를 한 unit task로 만들고 같은 row의 항들을 이진 트리로 더합니다. 두 child가 끝나야 parent addition이 준비됩니다. 홀수 개 항에서는 마지막 항을 다음 layer로 넘깁니다. 모든 row에는 diagonal nonzero가 있어 빈 reduction은 없습니다.

$$
W=\sum_i\{r_i+(r_i-1)\}=2\,\mathrm{nnz}-N,\qquad
S=\max_i\{1+\lceil\log_2r_i\rceil\}.
$$

rᵢ는 row의 nonzero 수입니다. W는 product와 addition의 전체 task 수, S는 product부터 output까지 가장 긴 의존 경로입니다. Task마다 같은 시간, 무료 데이터 접근, 독립 processor라는 이상화가 들어갑니다. Sparse index 접근이나 실제 GPU kernel 비용까지 포함한 cycle 예측은 아닙니다.

## 유한 Processor에서 스케줄하기

$$
T_P\ge\max\{\lceil W/P\rceil,S\}.
$$

아무리 잘 스케줄해도 전체 W개 작업을 P개보다 많이 동시에 실행할 수 없고 한 의존 경로를 건너뛸 수도 없습니다. 현재 구현은 매 tick의 시작에 모든 dependency가 끝난 task 중 index 순으로 최대 P개를 골라 실제 실행합니다. 이 greedy schedule이 항상 최적이라는 주장은 하지 않습니다. 각 task를 한 번만 실행했는지, child가 이전 tick에 끝났는지, 결과가 같은지를 독립 검사합니다.

## 손계산과 직접 풀기

N=8, band 폭 1이면 양 끝 row는 각각 2개, 내부 여섯 row는 각각 3개 nonzero입니다. nnz=22, W=44-8=36, S=3입니다. P=1이면 36 tick이고 P=4의 하한은 9 tick입니다. 현재 greedy 실행은 10 tick으로 하한보다 한 tick 큽니다. 하한과 달라도 유효한 스케줄일 수 있습니다.

직접 풀기: band 폭 0, N=P=8이면 W,S,T를 구하세요. 풀이: row마다 diagonal product 하나뿐이므로 W=8, S=1이며 8개 task를 같은 tick에 실행해 T=1입니다. Processor를 16개로 늘려도 span 1보다 작아지지 않습니다.

## AI의 희소·병렬 계산 평가

Sparse attention, graph aggregation과 sparse linear layer는 nonzero 수만으로 실제 가속을 보장하지 않습니다. Index 저장, 불규칙 접근, synchronization과 occupancy가 영향을 줍니다. 또한 균형 reduction은 덧셈 순서를 바꾸므로 유한 정밀도에서 serial 합과 bitwise 동일할 필요가 없습니다. 정확도 tolerance와 데이터 이동 비용도 함께 기록하세요. 현재 모형은 작업·의존성의 검증 가능한 기준이며 실제 장치 timing은 별도입니다.

참고: [MIT 성능 공학 강의](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
