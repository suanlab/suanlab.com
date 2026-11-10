## 같은 원소 수인데 왜 메모리 비용이 달라질까

행렬 전치는 B[j,i]=A[i,j]로 같은 N²개 원소를 읽고 씁니다. 하지만 row-major A를 순서대로 읽으면 B에는 열 방향으로 쓰게 되어 cache line 재사용이 나빠질 수 있습니다. Blocked transpose는 작은 tile 안에서 양쪽 line을 반복 사용합니다. 연산 횟수가 같아도 이동하는 line 수는 달라질 수 있습니다.

이 실험은 실제 주소 trace와 전치 결과를 만들고 명시한 한 수준 cache를 시뮬레이션합니다. 실제 CPU counter를 읽거나 DRAM traffic을 실측한 것이 아닙니다. 파랑은 선택한 순서의 누적 miss, 주황은 같은 설정의 naive 순서입니다. 둘의 scalar 접근 수는 항상 2N²입니다.

긴 주소 trace 표는 첫 32개와 마지막 8개 접근만 보여 줍니다. 합계와 cache 시뮬레이션은 전체 trace로 계산합니다.

## 주소·Line·LRU 모형

Double 한 원소는 8 byte이며 A의 scalar 주소는 iN+j, B의 주소는 N²+jN+i입니다. 두 배열은 겹치지 않습니다. 한 line에 L개 double을 담고 C개 line을 저장하는 fully associative cache를 가정합니다.

$$
\mathrm{line}(a)=\lfloor a/L\rfloor,\qquad
\mathrm{LRU\ miss}\iff\mathrm{first\ access}\ \text{or}\ RD\ge C.
$$

RD는 같은 line의 직전 접근 이후 만난 서로 다른 line의 수입니다. Miss 때 가장 오래 쓰지 않은 line을 내보내고, hit 때는 최신 위치로 옮깁니다. Set 충돌, prefetch, 여러 cache 수준, TLB와 병렬 coherent traffic은 포함하지 않습니다. 실제 메모리 계층에서는 L1 miss가 곧 DRAM 읽기라고 단정할 수도 없습니다.

## 쓰기 정책과 마지막 Flush

Write allocate는 store miss에도 line을 읽어 들입니다. 수정한 line은 dirty이고 축출할 때 writeback이 필요합니다. 프로그램 끝의 dirty resident line도 flush해 traffic에 포함합니다.

$$
\mathrm{transfers}=\mathrm{misses}+\mathrm{dirty\ evictions}+\mathrm{final\ flushes},\qquad
\mathrm{bytes}=8L\,\mathrm{transfers}.
$$

Miss만 세면 실제로 수정한 출력의 쓰기를 빠뜨립니다. 반대로 write-through나 non-temporal store 같은 다른 정책에는 같은 식을 그대로 적용할 수 없습니다. 현재 정책을 먼저 고정한 뒤 tile 크기와 용량을 바꾸세요.

## 손계산과 직접 풀기

N=8, L=4, C=8에서 naive의 miss는 80개, 4×4 blocked는 32개입니다. 전체 A,B가 차지하는 고유 line은 32개라 후자는 compulsory line 수에 도달합니다. 쓰기까지 포함한 모델 byte traffic은 각각 4608과 1536입니다. 이는 이 구체적인 trace와 정책의 값이지 모든 실제 장치에서의 속도비가 아닙니다.

직접 풀기: N=4, L=1, C=1에서 읽기와 쓰기가 번갈아 나오면 traffic은 얼마인가요? 풀이: 32개 scalar 접근이 모두 miss입니다. 16개 출력 store의 dirty line은 축출 또는 마지막 flush로 각각 한 번씩 기록되어 총 transfer는 48개, byte는 384입니다. Last flush를 빼면 한 line을 누락합니다.

## AI Kernel의 데이터 이동

Attention, tensor transpose와 fused kernel은 연산 수뿐 아니라 재사용·layout·중간 배열에 민감합니다. Arithmetic intensity는 대략 연산 수를 이동 byte로 나눈 값이며 memory bandwidth와 함께 성능을 제한할 수 있습니다. 그러나 현재 주소 모형의 절약량을 곧바로 실측 실행 시간이라고 부르지 마세요. 실제 benchmark에서는 장치·dtype·크기·warmup·반복과 정확성을 함께 기록해야 합니다.

참고: [MIT 캐시와 성능 공학](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
