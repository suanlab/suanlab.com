## 익숙한 소수 뒤의 유한한 격자

컴퓨터는 모든 실수를 저장하지 못하고 유한한 이진 유리수 중 하나를 저장합니다. 정규화 구간 [2^e,2^(e+1))에서 p가 선행 1을 포함하는 유효 bit 수이면 이웃 값의 간격은 2^(e-p+1)입니다. 유효 자릿수가 같아도 지수가 커지면 절대 간격이 커집니다. 따라서 작은 값에서 적절한 절대 허용오차를 큰 값에 그대로 적용하면 잘못 판단할 수 있습니다.

실험은 지수 범위를 제한하지 않은 toy format입니다. 가장 가까운 값으로 반올림하고 정확히 중간이면 유효 정수의 마지막 bit가 짝수인 쪽을 택합니다. IEEE 형식의 exponent 한계·subnormal·예외까지 모사하지는 않습니다. 지수 범위의 실패는 다음 단원에서 따로 다룹니다.

$$
\operatorname{fl}_p(x)=\underset{y\in\mathcal F_p}{\operatorname{nearest}}x,
\qquad u=2^{-p},\qquad
|\operatorname{fl}_p(x)-x|\le u|x|.
$$

이 상대 오차 상한은 정규화된 값의 한 번 반올림에 대한 것입니다. 긴 알고리즘 전체의 오차가 항상 u 이하라는 뜻은 아닙니다.

## 입력 표현과 누적 계산을 구별하기

먼저 입력을 s0=fl(x)로 저장하고 이진 증분 d=2^-k를 여러 번 더합니다. n번째 기준값은 s0+nd입니다. 선택 범위에서 이 dyadic 기준값은 binary64로 정확히 표현되므로 그림은 toy 연산의 오차를 분리해서 보여줍니다. 입력 표현 오차 |s0-x|는 별도 수치로 읽습니다.

p=4일 때 1 위쪽의 간격은 1/8입니다. 여기에 1/32를 더하면 다시 1로 반올림됩니다. 열여섯 번 순차적으로 더해도 결과는 1이지만 기준값은 1.5입니다. 실수의 덧셈 법칙이 틀린 것이 아니라, 중간마다 반올림하는 연산을 반복한 결과입니다.

## 보상 변수가 잃은 부분을 전달한다

보상 합은 다음처럼 낮은 부분을 추정해 다음 덧셈에 전달합니다.

$$
y=\operatorname{fl}(d-c),\quad t=\operatorname{fl}(s+y),\quad
c=\operatorname{fl}(\operatorname{fl}(t-s)-y),\quad s=t.
$$

실험은 이 식의 각 연산에도 선택한 bit 수의 반올림을 적용합니다. 위 예에서는 1.5를 복원하지만 모든 입력을 정확하게 더한다는 보장은 없습니다. 처음부터 표현하지 못한 입력 정보도 복원할 수 없습니다. 주황 오차가 0이면 저장된 입력에서 출발한 기준값과 맞는다는 뜻입니다.

## 한 번의 반올림에서 합의 조건수까지

범위 실패가 없고 정규화된 반올림 fl(a+b)=(a+b)(1+δ), |δ|≤u를 가정하자. 순차 덧셈을 전개하면 각 저장 입력에 이런 인자의 곱이 붙는다. ku<1이면 그 곱과 1의 차이는 γₖ=ku/(1−ku) 이하다. 상한은 (1+u)ᵏ≤1/(1−ku)의 귀납으로, 하한도 같은 방식으로 얻는다. 저장된 n개 항에 대해

$$
|\widehat S-S|\le\gamma_{n-1}\sum_i|x_i|,\qquad
\frac{|\widehat S-S|}{|S|}\le\gamma_{n-1}
\frac{\sum_i|x_i|}{|\sum_i x_i|}\quad(S\ne0).
$$

마지막 비율은 입력 성분의 상대 섭동에 대한 합의 조건수다. 각 연산이 상대 상계를 만족해도 상쇄로 합의 조건수가 커질 수 있다. 저장 입력에 대한 정리이므로 소수 표현 오차를 없애거나 보상 합의 정확성을 무조건 보장하지 않는다.

p=4에서 (1,d,−1)을 순차 합산하면 absorption이 드러난다.

| d | 합의 조건수 | 절대 오차 |
|---|---:|---:|
| 1/8 | 17 | 0 |
| 1/32 | 65 | 1/32 |
| 1/128 | 257 | 1/128 |

```python
from fractions import Fraction as F
def rounding(x, p):
    if not x: return x
    sign, a = (1 if x > 0 else -1), abs(x)
    e = a.numerator.bit_length()-a.denominator.bit_length()
    if a < F(2)**e: e -= 1
    step = F(2)**(e-p+1)
    scaled = a/step
    k = scaled.numerator//scaled.denominator
    remainder = scaled-k
    if remainder > F(1, 2) or (remainder == F(1, 2) and k % 2): k += 1
    return sign*k*step
for d in (F(1, 8), F(1, 32), F(1, 128)):
    terms = (F(1), d, F(-1))
    value = terms[0]
    for x in terms[1:]: value = rounding(value+x, 4)
    condition = sum(map(abs, terms))/abs(sum(terms))
    bound = F(1, 7)*sum(map(abs, terms))  # gamma_2, u=1/16.
    error = abs(value-d)
    assert error <= bound
    print(d, condition, error)
```

독립 유리수 반올림은 부동소수점 로그로 binade를 판정하지 않는다. 사이트 helper도 binary64 경계에서 반올림된 로그를 보정한다. Helper의 53-bit 지원이 실습의 3–24-bit 조절기를 늘린 것은 아니다.

## 직접 풀기: 중간 값과 경계

p=4에서 1.0625와 1.1875를 반올림하세요. 간격은 0.125입니다. 첫 값은 유효 정수 8이 짝수여서 1로, 둘째는 10이 짝수여서 1.25로 갑니다. 1.9375는 다음 binade의 경계인 2로 갑니다.

이제 입력 0.1, 덧셈 횟수 0으로 설정하세요. 두 누적 오차는 0이어도 입력 표현 오차는 남습니다. 두 오차가 서로 다른 기준을 사용하기 때문입니다. AI 실험에서 reduction 순서·재현성·gradient 허용오차를 해석할 때 이 구분부터 확인하세요.

더 읽기: [Python 부동소수점 설명](https://docs.python.org/3/tutorial/floatingpoint.html). 이 단원의 toy 연산과 손계산 예는 별도로 구성했습니다.
