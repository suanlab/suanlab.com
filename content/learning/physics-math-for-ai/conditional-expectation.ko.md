## 조건 사건 대신 정보 전체에 조건 걸기

조건부기대 E[X|G]는 관측 정보 G로 표현할 수 있는 확률변수입니다. 하나의 숫자가 아니라 표본점마다 값을 갖습니다. 적분가능한 X와 부분 σ-대수 G를 가정합니다. 정의는 G-가측성과 모든 G 사건에서의 적분 보존입니다. 관측하지 못하는 원자 내부를 구분하는 값을 내면 정보 G만 사용하는 예측이 아닙니다.

$$
Y=E[X\mid\mathcal G]:\quad Y\text{ is }\mathcal G\text{-measurable},\quad\int_A Y\,dP=\int_A X\,dP\ \forall A\in\mathcal G.
$$

## 유한 원자에서 직접 구성하기

기준 X=(0,2,1,3), P=(0.1,0.2,0.3,0.4)이고 정보 원자는 {0,1}, {2,3}입니다. 각 원자 안에서는 상수여야 하므로 원자의 가중평균을 택합니다. 첫 값은 (0·0.1+2·0.2)/0.3=4/3이고 둘째는 (1·0.3+3·0.4)/0.7=15/7입니다. 각 원자에서 적분이 보존되므로 그 합집합인 모든 G 사건에서도 보존됩니다.

$$
Y(\omega)=\frac{\sum_{i\in A}p_iX_i}{P(A)}\quad(\omega\in A,\ P(A)>0).
$$

## 기대의 tower와 정보 해상도

전체 공간을 사건으로 넣으면 E[E[X|G]]=E[X]=1.9입니다. 모든 정보를 주면 조건부기대는 원래 X가 되고, 자명한 정보만 주면 전체 평균이라는 상수가 됩니다. 더 작은 정보 H가 G 안에 포함되면 G로 평균낸 뒤 H로 다시 평균낸 결과는 처음부터 H로 평균낸 것과 같습니다. 일반 공간의 존재 정리는 Radon–Nikodym 정리로 연결되며 유일성은 거의 확실한 의미입니다.

$$
\mathcal H\subseteq\mathcal G\implies E[E[X\mid\mathcal G]\mid\mathcal H]=E[X\mid\mathcal H].
$$

## 영 원자와 최소제곱 해석

원자의 확률이 0이면 비율은 정의되지 않지만 그 원자 위 어떤 상수 버전을 택해도 적분 보존에는 영향이 없습니다. 실습은 명시적으로 0을 택합니다. 이것이 영 사건의 초등 조건부확률을 0으로 정의한다는 뜻은 아닙니다. X가 L2이면 조건부기대는 G로 예측할 수 있는 변수들 중 제곱오차를 최소화합니다. 가중 잔차가 모든 원자에서 적분 0이 되는 직교성을 확인하면 이 해석을 볼 수 있습니다.

$$
E[(X-Z)^2]=E[(X-Y)^2]+E[(Y-Z)^2]\quad(Y=E[X\mid\mathcal G],\ Z\in L^2(\mathcal G)).
$$

## Radon–Nikodym으로 조건부 기대 구성하기

확률공간에서 X가 적분가능하고 G가 부분 σ-대수라고 합시다. G 위에 두 유한 측도 ν±(A)=∫A X± dP를 정의합니다. 둘 다 G로 제한한 P에 절대연속입니다. Radon–Nikodym 정리로 비음수 G-가측 밀도 Y±를 얻습니다. 각각의 적분이 유한한 E[X±]이므로 Y=Y+−Y−는 적분가능하고 정의에서 요구한 사건별 적분을 보존합니다. 이는 RN 정리를 사용한 조건부 기대의 구성이며 RN 정리 자체의 증명은 아닙니다.

유일성을 위해 Y,Y′가 같은 정의를 만족한다고 합시다. G 사건 A_k={Y−Y′>1/k}에서 차이의 적분은 0이지만 적어도 P(A_k)/k입니다. 따라서 P(A_k)=0이고 가산 합집합을 취하면 P(Y>Y′)=0입니다. 둘을 뒤집어 같은 논증을 적용하면 거의 확실히 같습니다. 영 정보 원자 위 값은 여전히 버전의 선택입니다.

## 정보 포함 조건으로 tower 증명하기

H가 G에 포함되면 모든 H 사건은 G 사건이기도 합니다. Y=E[X|G]를 H로 조건화한 변수는 각 H 사건 A에서 ∫A Y=∫A X를 보존하고 H-가측입니다. 따라서 유일성으로 E[X|H]와 같습니다. 중간 적분을 바꿀 수 있게 하는 조건이 바로 정보의 포함 관계이며 임의의 두 분할에는 성립하지 않을 수 있습니다.

## 결론을 가정하지 않고 L2 사영을 정당화하기

X가 제곱 적분가능하다고 가정합니다. 먼저 Y=E[X|G]도 L2임을 보여야 합니다. Y를 [−K,K]로 자른 Y_K를 사용합니다. 단순함수 근사를 통해 적분 보존을 지시함수에서 유계 G-가측 함수로 확장할 수 있습니다. YY_K≥Y_K²이므로 Cauchy–Schwarz로 다음을 얻습니다.

$$
\|Y_K\|_2^2\le E[YY_K]=E[XY_K]\le\|X\|_2\|Y_K\|_2.
$$

따라서 ||Y_K||2≤||X||2이고 Y_K²가 Y²로 단조 증가하므로 단조수렴이 L2 성질을 줍니다. 잔차 X−Y는 모든 유계 G-가측 함수와 내적 0입니다. 임의의 L2 G-가측 함수를 절단한 함수들로 근사한 뒤 다시 Cauchy–Schwarz를 쓰면 직교성이 확장됩니다. 이제 X−Z=(X−Y)+(Y−Z)를 전개하면 앞의 제곱 오차 항등식을 얻습니다. 예측기가 왜 허용 공간에 속하는지와 교차항이 왜 사라지는지를 함께 정당화한 논증입니다.

## 포함되지 않는 두 정보 집합에서는 다른 답이 나올 수 있다

네 결과가 동일 확률이고 X=(0,0,0,4)라고 합시다. G의 원자는 {0,1},{2,3}, H의 원자는 {0,1,2},{3}입니다. 어느 σ-대수도 다른 쪽을 포함하지 않습니다. 조건부 기대 연산자를 P_G,P_H로 씁니다. X는 이미 H-가측이므로 P_H X=X이지만 표에서 P_H P_G X≠P_H X입니다.

| $\omega$ | X | $P_GX$ | $P_HP_GX$ |
|---|---|---|---|
| 0 | 0 | 0 | $2/3$ |
| 1 | 0 | 0 | $2/3$ |
| 2 | 0 | 2 | $2/3$ |
| 3 | 4 | 2 | 2 |

또한 P_G P_H X=P_G X는 P_H P_G X와 다르므로 두 연산자는 교환하지 않습니다. 두 정보의 교집합은 자명한 정보라 예측은 평균 1이라는 상수여야 하지만 표의 이중 사영은 그 상수도 아닙니다. 관련 없는 정보를 차례로 평균하는 것은 교집합 정보에 조건을 거는 일반적인 대체 방법이 아닙니다.

허용되는 G-가측 비교 변수 Z=(3,3,−1,−1)의 전체 제곱 오차는 11이며 잔차 오차 2와 예측기 차이 오차 9의 합입니다. 정보가 실제로 포함 관계에 따라 늘어난다면 이전 예측기도 여전히 허용되므로 최적 제곱 오차는 증가할 수 없습니다. 예측 위험을 비교하기 전에 사용할 수 있는 정보를 명시해야 합니다.

## 사건 항등식과 반례를 정확하게 계산하기

다음 표준 라이브러리 코드는 네 G 사건을 모두 열거하여 위험 항등식과 비포함 사영을 계산합니다. 기존 실습의 가중 기준값과 영 원자에서 선택한 0 버전도 재현합니다. 비포함 분할 H는 기존 조절기의 새 선택지가 아니라 별도로 실행하는 예제입니다.

```python
from fractions import Fraction as Q

def expect(x, p):
    return sum(a*b for a, b in zip(x, p))

def condition(x, p, atoms):
    out = [Q(0)] * len(x)
    for atom in atoms:
        mass = sum(p[i] for i in atom)
        value = sum(p[i]*x[i] for i in atom)/mass if mass else Q(0)
        for i in atom:
            out[i] = value
    return tuple(out)

def squared_error(x, y, p):
    return expect(tuple((a-b)**2 for a, b in zip(x, y)), p)

G, H = [[0, 1], [2, 3]], [[0, 1, 2], [3]]
p, x = (Q(1, 4),)*4, (Q(0), Q(0), Q(0), Q(4))
y = condition(x, p, G)
hx, hy = condition(x, p, H), condition(y, p, H)
assert y == (0, 0, 2, 2) and hx == x
assert hy == (Q(2, 3), Q(2, 3), Q(2, 3), Q(2))
assert hy != hx and condition(hx, p, G) != hy
for mask in range(4):
    event = [i for j, atom in enumerate(G) if (mask >> j) & 1 for i in atom]
    assert sum(p[i]*(x[i]-y[i]) for i in event) == 0
z = (Q(3), Q(3), Q(-1), Q(-1))
assert squared_error(x, z, p) == squared_error(x, y, p)+squared_error(y, z, p)
assert squared_error(x, z, p) == 11 and squared_error(x, y, p) == 2
assert condition(y, p, [[0, 1, 2, 3]]) == (Q(1),)*4
reference_p = tuple(Q(i, 10) for i in [1, 2, 3, 4])
reference_x = tuple(Q(i) for i in [0, 2, 1, 3])
assert condition(reference_x, reference_p, G) == (Q(4, 3), Q(4, 3), Q(15, 7), Q(15, 7))
assert condition(reference_x, (Q(1, 2), Q(1, 2), Q(0), Q(0)), G) == (1, 1, 0, 0)
for i in range(4):
    print("point, X, P_G X, P_H P_G X:", i, x[i], y[i], hy[i])
print("squared errors: total, residual, predictor gap:", 11, 2, 9)
```

## 추가 문제와 연구 연결

1. 최적 예측기가 왜 거의 확실한 의미에서만 유일한가요? **노름과 사건 적분이 영집합 수정을 구분하지 못하기 때문입니다.**
2. 임의의 두 정보 집합에 tower를 적용해도 되나요? **표가 보여주듯 포함 조건이 필요합니다.**
3. 예측 시점에 없는 특징을 G에 넣어도 되나요? **정보 문제가 바뀌며 누출이 생길 수 있으므로 사용할 정보를 구분해야 합니다.**

존재·tower·제곱 오차 예측은 [Durrett, §4.1, 정리 4.1.13·4.1.15 및 RN 정리 A.4.8](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf)을 참고하세요. 네 결과의 비교는 직접 계산했습니다. [Gaussian 예측](../multivariate-gaussians/)은 조건부 평균 모형의 구체적인 예입니다.

## 확인 문제와 해설

1. 전체 평균은 보존되나요? **예. G에 Ω가 포함됩니다.**
2. 영 원자의 선택값은 유일한가요? **아니요. 조건부기대의 버전은 영집합에서 달라도 됩니다.**
3. 정보가 늘면 최적 제곱오차가 증가하나요? **아니요. 이전 예측도 새 정보에서 허용됩니다.**

추가 읽기: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
