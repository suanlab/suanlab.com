## 그래프의 비용 단위와 시계의 밀리초

이 단원은 같은 배열과 target에 대한 선형·이진 탐색을 두 관점에서 비교합니다. 위 그래프는 실제 key probe 수에 사용자가 정한 단위 비용과 고정 overhead를 적용한 모형입니다. 아래 버튼은 현재 브라우저에서 실제 코드를 실행해 시간을 측정합니다. 둘을 같은 데이터라고 표시하지 않습니다.

모형에서 linear 단위 비용 α, binary 단위 비용 γ, binary 고정 비용 β를 고르면 다음을 계산합니다.

$$
C_L(N)=\alpha q_L(N),\qquad
C_B(N)=\beta+\gamma q_B(N).
$$

q는 실제 탐색이 수행한 probe 수입니다. Cost unit은 nanosecond나 millisecond로 보정하지 않은 추상 단위입니다. α,γ,β를 바꾸면 곡선이 달라지지만 실제 측정 코드를 빨라지게 바꾸는 것은 아닙니다. 실측 repetitions 설정만 batch의 실행 횟수를 바꿉니다.

## 같은 작업과 정당한 비교

Present target은 rank로 정하고 두 방법 모두 같은 sorted array를 받습니다. 배열 생성은 측정 밖에서 합니다. Linear와 binary가 같은 index를 반환하는지 검증하고, 구축 비용을 포함하는지 제외하는지 보고해야 합니다. 이미 정렬된 배열을 주는 현재 비교와 미정렬 데이터의 처음부터 끝까지 비용은 다른 과제입니다.

Asymptotic bound는 N이 충분히 클 때의 입력 가족을 설명합니다. 유한 크기의 crossover, cache와 branch, interpreter·JIT·allocator의 비용은 별도입니다. 첫 원소만 묻는 입력에서는 linear probe가 1이므로 단지 알고리즘 이름만 보고 항상 더 느리다고 말할 수도 없습니다.

## 실제 Benchmark 절차

버튼을 누르면 두 방법을 각각 20회 warmup하고, 선택한 repetition 수의 batch를 방법별 7개 측정합니다. Trial마다 먼저 실행하는 방법을 번갈아 바꾸며 trial 사이에는 브라우저에 제어를 돌려줍니다. Timed 구간은 반복된 실제 search 함수와 결과 checksum 누적을 포함하지만 배열 setup은 포함하지 않습니다. 함수의 비교 계수·반환 object 생성도 실행의 일부입니다.

$$
\widehat t=\operatorname{median}(t_1,\ldots,t_7)\quad\text{milliseconds per batch}.
$$

Checksum은 결과를 실제 사용하도록 돕습니다. 같은 입력·크기·반복에서 다시 실행해 변동을 관측하세요. 타이머 해상도보다 짧은 구간은 0으로 보일 수 있습니다. 이를 임의의 양수로 바꾸거나 항상 한 방법이 빠르도록 보정하지 않습니다. 다른 tab·부하·브라우저·JIT 상태도 측정에 영향을 줍니다.

## 손계산과 직접 풀기

N=1024의 마지막 key에서 linear는 1024 probe, binary는 11 probe입니다. α=γ=1, β=0이면 모형 비용은 1024와 11입니다. γ=500, β=1000이면 binary 비용은 6500으로 순위가 뒤집힙니다. 이 변화는 선언한 비용 모형의 결과이며 실측이 아닙니다.

직접 풀기: N=4의 마지막 key에 α=1, γ=5, β=50을 적용하세요. 풀이: q_L=4, q_B=3이므로 C_L=4, C_B=65입니다. 단위 비용 설정을 바꾼 뒤 실측 버튼도 눌러 보세요. 가중치 변경 자체가 같은 실제 search 함수의 timing을 인과적으로 바꾸었다고 해석하면 안 됩니다.

## AI 논문의 성능 주장

장치·runtime·버전·dtype·데이터 크기·warmup·반복·측정 구간·정확도를 기록해야 재현 가능한 속도 비교가 됩니다. GPU 같은 비동기 장치에는 적절한 동기화도 필요합니다. 현재 버튼은 동기 JavaScript CPU 실행이며 GPU timing 절차를 대신하지 않습니다. 일반 복잡도 증명, 선언한 비용 모형, 유한 환경의 실측 결과를 함께 제시하되 서로 바꾸어 부르지 마세요.

참고: [MIT 성능 공학 강의](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
