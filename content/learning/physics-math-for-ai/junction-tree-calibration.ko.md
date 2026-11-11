## 순환을 클러스터의 나무로 바꾸기

네 변수의 cycle A−B−C−D−A에 대각선 AC를 넣으면 클러스터 ABC와 ACD로 나눌 수 있습니다. 두 클러스터가 공유하는 separator는 AC입니다. 원래 그래프가 순환이어도 클러스터들을 연결한 구조는 나무입니다.

Junction tree에는 각 인자의 scope를 담는 클러스터가 있어야 하고, 같은 변수를 담는 클러스터들은 연결된 부분나무를 이루어야 합니다. 두 번째 조건이 running intersection입니다. 단순히 임의의 클러스터들을 나무로 연결하는 것만으로는 충분하지 않습니다.

## 인자를 정확히 한 번 배정하기

φ₁(A,B,C)=φAψABψBCe(C), φ₂(A,C,D)=ψCDψDA로 둡니다. 공통 변수 A,C가 있다고 unary나 관측 인자를 두 번 곱하면 다른 모형이 됩니다. 이 실습은 원래 인자를 한 클러스터에만 배정합니다.

## 두 방향 메시지

메시지는 공유하지 않는 변수를 합한 함수입니다. μ₁₂(A,C)=ΣBφ₁, μ₂₁(A,C)=ΣDφ₂입니다. 두 메시지의 곱을 합하면 원래 모형과 증거의 normalizer Z를 얻습니다.

$$
b_{ABC}=\phi_1\mu_{21}/Z,\qquad b_{ACD}=\phi_2\mu_{12}/Z,
\quad b_{AC}=\mu_{12}\mu_{21}/Z.
$$

클러스터 belief를 AC로 각각 주변화하면 같은 bAC가 나와야 합니다. 파랑·주황은 두 계산을 비교합니다. 수치창의 작은 잔차는 부동소수점 연산의 불일치이며 논리적으로 다른 확률이 필요하다는 뜻은 아닙니다.

## 증거와 표의 해석

C=1을 관측하면 C=0인 separator 행은 0입니다. 메시지 원시값은 합이 1일 필요가 없고, 표의 마지막 두 열만 정규화한 주변입니다. 모든 field와 결합이 0이면 각 할당 가중치 1, Z=16이고 separator의 네 질량은 .25입니다.

## 정확성과 계산량

나무 위에서 유효한 클러스터 인자화와 충분한 메시지 교환을 사용하면 정확한 추론을 할 수 있습니다. 하지만 클러스터 크기가 커지면 표가 지수적으로 늘어납니다. 현재 클러스터는 8칸, separator는 4칸입니다. 일반 모델의 treewidth가 크면 junction tree로 옮겼다는 사실만으로 계산이 작아지지 않습니다.

## Running intersection이 필요한 증명상의 위치

클러스터 나무의 한 간선을 자르면 두 부분나무가 됩니다. Running intersection이 성립하면 두 쪽에서 동시에 쓰는 변수는 반드시 그 간선의 separator에 포함됩니다. 어떤 변수가 양쪽에 있는데 separator에는 없다면, 그 변수를 가진 두 클러스터 사이의 경로가 중간에 끊겨 조건을 어기기 때문입니다.

따라서 separator 값을 고정하면 양쪽의 나머지 합을 독립적으로 수행할 수 있습니다. 부분나무의 인자 곱을 내부 변수에 대해 합한 함수를 메시지라고 정의하면, 잎에서 시작한 귀납으로 일반 메시지 식이 얻어집니다.

$$
m_{i\to j}(x_{S_{ij}})=
\sum_{x_{C_i\setminus S_{ij}}}\Phi_i(x_{C_i})
\prod_{k\in N(i)\setminus\{j\}}m_{k\to i}(x_{S_{ki}}).
$$

각 Φᵢ에는 원래 인자를 한 번만 배정합니다. 메시지의 변수별 의존을 잃지 않고 충분히 교환하면 bᵢ∝Φᵢ∏m이 정확한 클러스터 주변입니다. Separator가 양쪽 부분문제에 필요한 모든 공유 정보를 보관한다는 것이 핵심입니다.

## 작은 separator 잔차만으로는 구조를 인증할 수 없다

별도 세 변수 예제로 w(a,b,c)=1{a=b}1{a=c}를 보세요. 가능한 할당은 000,111이라 Z=2입니다. 클러스터를 AB−BC−AC로 연결하고 가운데 BC의 인자는 1로 두면, A가 양 끝에 있지만 가운데 없어서 running intersection이 깨집니다. 잘못된 나무 계산은 A를 왼쪽·오른쪽의 별도 변수로 취급합니다.

$$
Z_{\mathrm{bad}}=\sum_{a_L,b,c,a_R}1\{a_L=b\}1\{a_R=c\}=4\ne2.
$$

양 끝 메시지는 각각 상수 1입니다. B와 C separator 주변은 모두 (1/2,1/2)로 calibration이 맞아도 원래 모형의 Z와 다릅니다. 가운데 잘못된 BC belief는 균등하지만 원래 B,C는 항상 같습니다. 그림은 A를 가운데에도 담는 AB−ABC−AC로 고치는 이유를 보여 줍니다. 원래 인자는 여전히 한 번씩만 배정합니다.

![잘못된 AB–BC–AC 경로에서는 A의 클러스터가 끊기지만 가운데를 ABC로 확장하면 연결되며, 두 separator가 일치해도 잘못된 모형의 정규화 상수가 4가 될 수 있음을 비교한 그림.](/assets/learning/graphical-running-intersection-ko.svg)

| 구성 | 같은 변수의 연결 | Z |
|---|---|---|
| 원래 세 변수 결합 | A를 하나로 공유 | 2 |
| AB−BC−AC | A가 중간에서 끊김 | 4 |
| AB−ABC−AC | A가 경로 전체에 존재 | 2 |

분수 코드는 위 반례와 separator calibration을 확인합니다. Zero 인자가 허용된 작은 예제이며 화면의 양의 Ising 네-cycle과는 별도 모형입니다. 양의 separator에서는 보정된 두-clique 결합을 bABC·bACD/bAC로 복원할 수 있습니다. bAC=0인 곳의 0/0은 그대로 계산하지 말고 그 지지 밖 결합 질량을 0으로 취급해야 합니다.

**직접 풀기:** 더 많은 메시지 반복으로 빠진 A를 복원할 수 있나요? **해설:** 아닙니다. 빠진 공유 변수를 담도록 클러스터 구조를 고쳐야 합니다.

```python
from fractions import Fraction as F
from itertools import product

def equal(x, y):
    return F(x == y)

# Intended three-variable law; its normalizer is two.
z = sum(equal(a,b)*equal(a,c)
        for a,b,c in product([0,1], repeat=3))
# Invalid AB--BC--AC path forgets A between the endpoint clusters.
# Its tree sum has separate copies a_left and a_right.
z_bad = sum(equal(al,b)*equal(ar,c)
            for al,b,c,ar in product([0,1], repeat=4))
assert z == 2 and z_bad == 4
# Messages from either leaf are the constant function one.
m_left = [sum(equal(a,b) for a in [0,1]) for b in [0,1]]
m_right = [sum(equal(a,c) for a in [0,1]) for c in [0,1]]
assert m_left == m_right == [1,1]
# Both separator marginals calibrate to (1/2,1/2) despite wrong Z.
separator_b = [sum(m_left[b]*m_right[c]
                   for c in [0,1])/z_bad for b in [0,1]]
separator_c = [sum(m_left[b]*m_right[c]
                   for b in [0,1])/z_bad for c in [0,1]]
assert separator_b == separator_c == [F(1,2),F(1,2)]
print('intended Z, invalid cluster-tree Z:', z, z_bad)
print('both calibrated separators:', separator_b)
```


## 확인 문제와 해설

1. AC가 두 클러스터에 있다고 인자를 두 번 곱하나요? **아니요. 원래 인자는 한 번만 배정합니다.**
2. 메시지가 반드시 확률분포인가요? **아니요. separator 함수입니다.**
3. 두 separator 주변은 어떤 관계여야 하나요? **수치 정밀도 안에서 일치해야 합니다.**

추가 읽기: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/jt/).
