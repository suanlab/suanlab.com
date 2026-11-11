## 벡터공간에서 모듈로

벡터공간은 체의 스칼라가 가법군에 작용하는 구조입니다. 스칼라를 일반적인 단위원 있는 환으로 바꾸면 모듈을 얻습니다. 덧셈과 스칼라 분배법칙은 유지되지만 비영 스칼라의 역원이 보장되지 않습니다. 실습은 R=Z/nZ 위의 자유 모듈 R²를 사용합니다. 원소는 두 잉여류의 순서쌍이며 n²개입니다. 모든 모듈이 자유 모듈이라는 뜻은 아닙니다.

$$
r(u+v)=ru+rv,\quad(r+s)v=rv+sv,\quad(rs)v=r(sv),\quad1v=v.
$$

## 환 위 선형사상의 실제 계산

T(x,y)=(ax,by)는 R-선형사상입니다. 표에서 n²개 입력과 출력을 모두 확인할 수 있습니다. 기준 n=6,a=2,b=3에서는 2x=0의 해가 x=0,3이고 3y=0의 해가 y=0,2,4입니다. 따라서 영으로 가는 순서쌍은 2·3=6개입니다. 계수가 0이 아니라는 이유만으로 소거하거나 나누지 마세요.

$$
|\ker T|=\gcd(n,a)\gcd(n,b),\qquad |\operatorname{im}T|=\frac{n^2}{|\ker T|}.
$$

## 차원과 원소 수를 구분하기

계수체가 F5이면 같은 대각 계수 2와 3이 모두 단원이라 가역이고 핵은 영벡터 하나입니다. 이때 핵의 차원은 0입니다. 영 사상은 핵이 전체 F5²이므로 원소는 25개, 차원은 2입니다. 합성수 법의 경우에도 핵과 상의 개수는 셀 수 있지만 체 위 벡터공간 차원이라고 부를 수는 없습니다. 해당 차원 항목은 정의하지 않습니다.

$$
\dim_{\mathbb F_p}\ker T=\#\{\text{zero diagonal coefficients}\}\quad\text{in this diagonal example}.
$$

## 어떤 선형대수 직관이 남는가

모듈 준동형의 핵과 상은 부분모듈이며 몫 구조를 생각할 수 있습니다. 그러나 일반 환에서는 모든 부분모듈에 벡터공간처럼 보완공간이나 기저가 존재하지 않습니다. 자유 모듈 R²의 표를 일반 모듈의 완전한 분류로 읽지 마세요. 대수적 부호나 이산 구조의 학습을 다룰 때 스칼라 영역을 명시해야 같은 행렬 기호가 다른 계산을 뜻하는 혼동을 피할 수 있습니다.

정수환 Z 위의 모듈 Z/2Z는 자유가 아닌 구체적인 예입니다. 비영 원소 u에 2를 곱하면 0입니다. 반면 정수 위 자유 모듈의 비영 벡터는 2배 해도 0이 되지 않습니다. 따라서 이 모듈에는 자유 기저를 줄 수 없습니다.

## 군 표현을 군대수 위 모듈로 읽기

군대수 F[G]의 원소는 체 F의 계수를 붙인 군 원소의 형식적 합입니다. 곱셈은 군의 곱과 분배법칙으로 정합니다. V 위 군 표현은 이 합을 선형으로 확장하여 F[G]의 작용을 만듭니다. 군 작용에 불변인 부분공간은 이 작용의 부분모듈과 같습니다. 따라서 모듈의 스칼라환 관점이 대칭 제약을 갖는 층과 연결됩니다. 아래는 앞의 잉여류 대각 사상 조절기와 별도로 다루는 예제입니다.

$$
\left(\sum_{g\in G}a_g g\right)v=\sum_{g\in G}a_g\rho(g)v.
$$

## 평균으로 불변 보완공간을 만드는 조건과 증명

G는 유한군, V는 체 F 위 유한 차원 공간이고 F의 표수가 |G|를 나누지 않는다고 가정합니다. 표수 0도 포함됩니다. 불변 부분공간 W의 기저를 V의 기저로 확장하면 W 위로 가는 선형 사영 Q를 고를 수 있습니다. Q 자체는 군과 교환하지 않아도 됩니다. 켤레 작용을 평균하여 이 결함을 고칩니다.

$$
P=\frac1{|G|}\sum_{g\in G}\rho(g)^{-1}Q\rho(g).
$$

W는 g와 그 역원의 작용에 모두 불변이므로 각 항은 V를 W 안으로 보내고 W에서는 항등사상입니다. 따라서 P의 상은 W 안에 있고 W에서는 항등이며 P²=P입니다. P를 군 원소로 켤레 변환하면 합의 항 순서만 바뀌므로 모든 작용과 교환합니다. 그러면 ker P도 불변이고 V는 W와 ker P의 직합입니다. 실제로 v=Pv+(v−Pv)로 쓰고 P를 적용하면 둘째 항이 핵에 속함과 두 부분공간의 교집합이 0임을 확인할 수 있습니다. 이것이 Maschke 정리의 분해 논증입니다. 내적은 필요하지 않으며, 임의 좌표에서 이 사영이 직교 사영이라는 뜻도 아닙니다.

## 표수 3에서 완전히 열거할 수 있는 반례

이번에는 F3 위에서 C3의 생성원을 J로 표현하고 W를 (1,0)이 생성하는 직선으로 둡니다. J=I+N, N²=0이므로 세제곱은 이 체에서 I+3N=I입니다. 같은 정수 행렬을 실수에서 쓰면 C3 표현이 아닙니다. W 위로 사영하고 W에서 항등인 모든 사영은 다음 형태여야 합니다.

$$
J=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad
P_b=\begin{pmatrix}1&b\\0&0\end{pmatrix},\quad
P_bJ-JP_b=\begin{pmatrix}0&1\\0&0\end{pmatrix}\ne0.
$$

| F3의 b | 사영의 두 행 | 교환자의 두 행 |
|---|---|---|
| 0 | (1,0), (0,0) | (0,1), (0,0) |
| 1 | (1,1), (0,0) | (0,1), (0,0) |
| 2 | (1,2), (0,0) | (0,1), (0,0) |

어느 사영도 J와 교환하지 않습니다. 불변 보완공간이 존재한다면 그 공간을 따라 W로 가는 사영은 군과 교환해야 하므로 모순입니다. 따라서 이 불변 직선에는 불변 보완공간이 없습니다. 여기서 I+J+J²는 영 행렬입니다. 이를 3으로 나눌 수 없고, 나누기 전 합도 W에서 항등이 아닙니다. 표수 가정이 실패하면 이런 반례가 허용된다는 뜻이지 표수 3의 모든 표현이 분해되지 않는다는 뜻은 아닙니다.

## 유한 반례를 독립 코드로 검산하기

다음 표준 라이브러리 코드는 81개 행렬을 모두 열거하여 W 위의 사영을 고른 뒤 교환자를 검사합니다. 이 완전 열거는 F3의 2차원 예제에 대한 것이며 일반 분해 정리는 앞에서 별도로 증명했습니다.

```python
from itertools import product

p = 3
I, J = (1, 0, 0, 1), (1, 1, 0, 1)

def mul(a, b):
    return tuple(sum(a[2*i+k] * b[2*k+j] for k in range(2)) % p
                 for i in range(2) for j in range(2))

def sub(a, b):
    return tuple((x-y) % p for x, y in zip(a, b))

powers = [I, J, mul(J, J)]
assert mul(powers[-1], J) == I
assert len(set(powers)) == 3
assert tuple(sum(g[i] for g in powers) % p for i in range(4)) == (0,)*4
projections = [a for a in product(range(p), repeat=4)
               if a[0] == 1 and a[2:] == (0, 0) and mul(a, a) == a]
assert projections == [(1, b, 0, 0) for b in range(p)]
for a in projections:
    commutator = sub(mul(a, J), mul(J, a))
    assert commutator == (0, 1, 0, 0)
    print(a, commutator)
fixed = [(x, y) for x, y in product(range(p), repeat=2)
         if ((x+y) % p, y) == (x, y)]
assert fixed == [(0, 0), (1, 0), (2, 0)]
print("all matrices:", p**4, "projections:", len(projections), "fixed:", fixed)
```

## 추가 문제와 연구 연결

1. 불변 보완공간이 있으면 왜 사영이 군과 교환하나요? **두 직합 성분이 모두 보존되므로 작용 전후 어느 때 성분을 골라도 같습니다.**
2. 평균이 영이면 불변 보완공간을 얻은 것인가요? **교환하는 연산자는 얻었지만 W에서 항등이 아니므로 W를 분리하는 사영이 아닙니다.**
3. 실수 대칭층에도 평균 논증을 쓸 수 있나요? **유한군이고 행렬들이 실제 표현을 이룬다면 가능합니다. 수치 학습의 정확도와 과제의 대칭 가정은 별도로 평가해야 합니다.**

평균 증명과 표수 가정은 [Etingof 외, §3.1, 정리 3.1](https://math.mit.edu/~etingof/replect.pdf)을 참고하세요. 불변 비선형 사상은 [대수적 표현](../algebraic-representations-scope/)에서 이어집니다.

## 확인 문제와 해설

1. Z6에서 2·3은? **0이므로 비영 입력이 사라집니다.**
2. 기준 상의 크기는? **36/6=6입니다.**
3. F5² 영 사상의 핵 차원은? **2이며 원소 수는 25입니다.**

추가 읽기: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).

[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf).
