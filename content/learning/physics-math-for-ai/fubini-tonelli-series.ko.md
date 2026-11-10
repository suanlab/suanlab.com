## 이중 적분은 교환 조건이 필요하다

곱 측도 공간에서 비음수 가측함수에는 Tonelli를 적용해 반복 적분의 순서를 바꿀 수 있으며 값이 무한이어도 됩니다. 부호 함수에는 절댓값 적분이 유한하다는 Fubini 조건이 순서 교환을 보장합니다. 여기서는 두 공간 모두 양의 정수 집합과 세기 측도이므로 σ-유한 조건을 만족합니다. 적분은 이중합이 됩니다. 유한 표와 무한 배열을 구분하는 것이 핵심입니다.

$$
f\ge0\implies\iint f=\int\!\left(\int f\,d\nu\right)d\mu=\int\!\left(\int f\,d\mu\right)d\nu.
$$

## 같은 유한 직사각형에서는 항상 같다

부호 모드는 대각 a_ii=1, 바로 오른쪽 a_i,i+1=−1이고 나머지는 0입니다. N=M=4인 표에서는 마지막 대각 1이 남아 합이 1입니다. 열을 하나 더 넣으면 모든 행의 +1과 −1이 상쇄되어 합은 0입니다. 어느 직사각형에서도 유한 행합들의 합과 유한 열합들의 합은 같습니다. 유한 덧셈 자체가 실패하는 예제가 아닙니다.

$$
\sum_{i=1}^N\sum_{j=1}^Ma_{ij}=\sum_{j=1}^M\sum_{i=1}^Na_{ij}.
$$

## 무한 반복합은 서로 다른 극한이다

각 무한 행은 +1과 −1이 있어 합이 0이므로 행들의 합은 0입니다. 첫 열의 합은 1이고 이후 열은 한 행의 −1과 다음 행의 +1이 상쇄되므로 열들의 합은 1입니다. 양의 부분과 음의 부분의 전체 합이 모두 무한이라 부호 있는 곱 적분은 정의되지 않습니다. 두 반복합이 유한하다는 것만으로 Fubini의 절대적분가능 조건을 대신할 수 없습니다.

$$
\sum_i\left(\sum_j a_{ij}\right)=0,\quad\sum_j\left(\sum_i a_{ij}\right)=1,\quad\sum_{i,j}|a_{ij}|=\infty.
$$

## 비음수로 바꾸면 무엇이 달라지는가

비음수 모드는 대각과 바로 오른쪽을 모두 +1로 둡니다. 두 무한 반복합과 곱 적분은 모두 ∞입니다. 무한이라는 이유로 Tonelli가 실패하는 것이 아닙니다. 4×4 유한 표의 합 7은 그 무한값을 유한하다고 주장하는 값이 아니라 절단 계산입니다. AI에서 배치·잠재변수·시간 축의 기대를 교환할 때 비음수 손실인지, 부호 경사의 절댓값이 적분가능한지 구별해 확인해야 합니다.

$$
\iint|f|\,d(\mu\times\nu)<\infty\implies\text{the signed Fubini interchange is valid}.
$$

## 확인 문제와 해설

1. 부호 모드의 4×5 유한 합은? **0입니다.**
2. 무한 행·열 반복합이 다른 이유는? **서로 다른 순서로 극한을 취하고 절대합이 무한이기 때문입니다.**
3. 비음수 무한합은 ∞여도 되나요? **예. Tonelli는 이를 허용합니다.**

추가 읽기: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
