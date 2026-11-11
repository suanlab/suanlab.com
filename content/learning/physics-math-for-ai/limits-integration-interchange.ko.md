## 사라지는 점값과 남는 면적

[0,1]에서 n≥2인 연속 삼각형 봉우리를 정의합니다.

$$
f_n(x)=n\max(0,1-|nx-1|).
$$

꼭짓점은 x=1/n, 높이는 n이고 지지는 [0,2/n]입니다. 삼각형 면적은 밑변×높이/2=1입니다. 반면 고정 x>0은 충분히 큰 n에서 지지 바깥에 있고 x=0의 값도 항상 0이므로 모든 점에서 fₙ→0입니다.

$$
\lim_n\int_0^1 f_n(x)\,dx=1\ne0=\int_0^1\lim_n f_n(x)\,dx.
$$

유한 적분이 모두 존재하고 점별 극한도 존재해도 교환은 실패할 수 있습니다. 봉우리가 좁다는 이유로 수치 격자가 놓친 면적을 0이라고 해석하면 이 핵심 반례를 잘못 읽게 됩니다.

## 균등 수렴이면 가능한 교환

유한 길이 구간에서 적분 가능한 fₙ이 f로 균등 수렴하면

$$
\left|\int_a^b(f_n-f)\right|\le(b-a)\|f_n-f\|_\infty\to0.
$$

봉우리 모형은 supremum이 n이므로 이 조건을 충족하지 않습니다. 균등 수렴이 적분 교환의 충분조건이라는 사실과 필요한 유일 조건이라는 주장은 다릅니다. 더 약한 측도론적 조건도 있습니다.

## 지배수렴 정리의 가정

측도공간에서 가측 함수 fₙ이 거의 모든 곳에서 f로 수렴하고, 모든 n에 대해 |fₙ|≤g인 적분 가능한 공통 g가 존재하면 적분과 극한을 교환할 수 있습니다. 가측성·거의 모든 곳·절댓값·공통 지배·적분 가능성을 함께 읽어야 합니다.

실습의 비교 모형 x/n은 |x/n|≤1이고 [0,1]에서 1이 적분 가능하므로 적분이 1/(2n)→0입니다. 이 간단한 예제는 균등 수렴도 만족합니다. 지배수렴의 일반 증명은 측도·확률 단원에서 확장합니다.

## 봉우리에는 왜 공통 지배가 없나

공통 적분 가능 g가 봉우리를 모두 지배한다면 ∫₀²⁄ⁿg≥1이어야 합니다. 그러나 적분 가능한 함수의 적분은 길이가 0으로 가는 집합에서 0으로 갑니다. 이는 모순입니다. 단순히 상수 1이 지배하지 못한다는 사실보다 강한 논증입니다.

봉우리의 mass가 점별 극한에서 사라지는 현상은 함수의 약한 수렴이나 측도의 극한과도 연결됩니다. 여기서는 분포 극한을 도입하지 않고 보통 함수값과 적분의 교환 실패만 확인합니다.

## AI 수식의 교환 기호를 점검하기

기대값 안의 극한, 무한 합의 항별 적분, 손실의 기대 안 미분은 조건을 확인해야 하는 단계입니다. 지배수렴이 극한과 적분을 정당화하더라도 미분 교환에는 차분 몫 또는 도함수의 적절한 지배 등 추가 검토가 필요합니다. “정규화된다”라는 말만으로 모든 교환이 허용되지는 않습니다.

## 같은 면적을 무작위 입력에서 추정하기

$U_1,\ldots,U_M$을 $[0,1]$의 독립 균등 입력, $M\ge1$을 배치 크기라고 합시다. 표본평균 $\widehat I_{n,M}=M^{-1}\sum_i f_n(U_i)$로 같은 삼각형 봉우리의 적분을 추정합니다. 정수 $n\ge2$에서 양쪽 선형 구간을 적분하면

$$
\mathbb E[f_n(U)]=1,\qquad
\mathbb E[f_n(U)^2]=\frac{2n}{3},\qquad
\operatorname{Var}(\widehat I_{n,M})=\frac{2n/3-1}{M}
$$

입니다. 이차 모멘트는 상승 구간에서 $v=nx$로 치환한 뒤 두 배 하여 $2n\int_0^1v^2\,dv=2n/3$으로 구합니다. 표본들이 독립이므로 평균의 분산은 한 표본의 분산을 M으로 나눈 값입니다. 고정한 n과 M에서 추정기는 불편이지만, 배치 크기를 고정하면 n과 함께 분산이 증가합니다.

함수값이 양수인 영역은 $(0,2/n)$입니다. 모든 표본이 봉우리를 놓칠 확률은 $(1-2/n)^M$입니다. 배치 크기가 8이면 다음과 같습니다.

| 인덱스 n | 배치 전체가 0일 확률 | 표본평균의 분산 |
|---|---|---|
| 4 | 0.003906 | 5/24 |
| 16 | 0.343609 | 29/24 |
| 64 | 0.775700 | 125/24 |

전부 0인 배치에서는 경험적 분산도 0이지만 모집단 분산은 양수입니다. 작은 오차막대나 면적이 보이지 않는 그림을 해석할 때 입력을 어떻게 추출했는지 기록하세요.

## 두 극한에서 무엇을 고정하는지 명시하기

M을 고정하고 같은 유한 개의 균등 표본을 사용하면서 n을 늘려 봅시다. 확률 1로 표본의 최솟값은 양수입니다. 충분히 큰 n에서는 $2/n$이 그 최솟값보다 작아져 모든 표본값이 0이므로 $\widehat I_{n,M}\to0$입니다. 그래도 기대값은 계속 1입니다. 드물지만 커지는 값들이 극한과 기대값의 교환을 막습니다.

n을 고정하고 M을 늘리면 분산이 감소합니다. 둘을 함께 늘리되 $M_n/n\to\infty$이면 위 분산이 0으로 갑니다. 추정기가 불편이므로 1로 평균제곱 수렴합니다. 서로 다른 극한 절차입니다. 더 큰 시뮬레이션이나 배치가 근사를 해결한다고 주장할 때 어떤 매개변수를 고정하고 무엇을 늘리는지 명시하세요.

## 제안 분포를 바꾸면서 목표 적분 보존하기

$X_i$를 $(0,2/n)$에서 서로 독립으로 균등하게 추출하면 그 구간의 밀도는 $q_n=n/2$입니다. 원래 적분을 추정하려면 $f_n(X_i)/q_n(X_i)$를 평균합니다. 일반적인 항등식을 쓰려면 제안 분포가 적분에 비영으로 기여하는 영역을 모두 덮어야 합니다. [Owen, Importance sampling, §9.1·정리 9.1](https://artowen.su.domains/mc/Ch-var-is.pdf)

이 예제에서는 $V=nX_i-1$이 $(-1,1)$의 균등 변수이고 $H=1-|V|$가 $[0,1]$의 균등 변수입니다. $0\le h\le1$에서 양쪽 바깥 구간의 확률을 더하면 $\Pr(H\le h)=h$이기 때문입니다. 따라서

$$
\frac{f_n(X_i)}{q_n(X_i)}=2H,\qquad
\mathbb E[2H]=1,\qquad
\operatorname{Var}\!\left(\frac1M\sum_i2H_i\right)=\frac1{3M}
$$

입니다. 가중 추정기의 분산은 n에 의존하지 않습니다. 같은 제안 분포에서 가중치 없이 봉우리 값을 평균하면 기대값은 $n/2$로 바뀝니다. 제안 구간을 $(0,1/n)$으로 더 줄이면 면적 절반을 놓쳐 가중 후에도 기대값이 $1/2$입니다. 올바른 가중치와 충분한 지지 영역이 함께 필요합니다. 이 구성에서는 지지를 알고 있지만 실제 연구에서는 위치를 추정하거나 찾아야 할 수 있습니다.

## 유한한 정확 계산으로 모멘트 검산하기

$[0,1]$의 균등 추출에서 양수 관측의 개수를 K라고 합시다. 성공 확률이 $2/n$인 이항분포입니다. $K=k$가 주어지면 k개의 양수 높이는 서로 독립인 $nH_j$이며 $H_j$는 $[0,1]$의 균등 변수입니다. 그 평균과 분산이 $1/2$, $1/12$이므로

$$
\mathbb E[\widehat I\mid K=k]=\frac{nk}{2M},\qquad
\mathbb E[\widehat I^2\mid K=k]
=\frac{n^2}{M^2}\left(\frac{k}{12}+\frac{k^2}{4}\right)
$$

입니다. 가능한 개수 M+1개에 대해 조건부 모멘트를 합하면 됩니다. Python 예제는 정확한 유리수 산술을 사용하므로 항등식 검산이 난수 시드나 유리한 유한 시뮬레이션에 의존하지 않습니다. 알려진 경계 0, 1/n, 2/n에서 각 선형 구간과 그 제곱을 Simpson 공식으로 적분하여 모멘트도 독립 검산합니다. 삼차 이하 다항식에서는 구간 길이에 (왼쪽 값 + 중간 값의 네 배 + 오른쪽 값)/6을 곱하면 정확한 적분이며 네 단항식으로 확인할 수 있습니다.

```python
from fractions import Fraction
from math import comb


def spike(index, x):
    return index * max(0, 1 - abs(index * x - 1))


batch_size = 8
for index in [4, 16, 64]:
    # Simpson's rule is exact on each linear side and its quadratic square.
    area = integral_squared = Fraction(0)
    for left, right in [(Fraction(0), Fraction(1, index)),
                        (Fraction(1, index), Fraction(2, index))]:
        values = [spike(index, x) for x in [left, (left + right) / 2, right]]
        piece_area = (right - left) * (values[0] + 4*values[1] + values[2]) / 6
        piece_squared = (right - left) * (
            values[0]**2 + 4*values[1]**2 + values[2]**2
        ) / 6
        if left == 0:
            half_area = piece_area
        area += piece_area
        integral_squared += piece_squared
    assert area == 1 and half_area == Fraction(1, 2)
    assert integral_squared == Fraction(2 * index, 3)

    hit_probability = Fraction(2, index)
    mean = estimate_second = Fraction(0)
    for hits in range(batch_size + 1):
        probability = (
            comb(batch_size, hits)
            * hit_probability**hits
            * (1 - hit_probability)**(batch_size - hits)
        )
        conditional_mean = Fraction(index * hits, 2 * batch_size)
        conditional_second = index**2 * (
            Fraction(hits, 12 * batch_size**2)
            + Fraction(hits**2, 4 * batch_size**2)
        )
        mean += probability * conditional_mean
        estimate_second += probability * conditional_second
    variance = estimate_second - mean**2
    missed = (1 - hit_probability)**batch_size
    assert mean == area
    assert variance == (integral_squared - area**2) / batch_size

    proposal_density = Fraction(index, 2)
    importance_variance = (
        integral_squared / proposal_density - area**2
    ) / batch_size
    assert importance_variance == Fraction(1, 3 * batch_size)
    assert variance / importance_variance == 2 * index - 3
    assert proposal_density * area == Fraction(index, 2)
    print(index, round(float(missed), 6), variance, importance_variance)
```

각 행은 n, 배치가 봉우리를 놓칠 확률, 원래 분산, 지지 제안 분포의 분산을 출력합니다. M=8에서 마지막 분산은 항상 `1/24`입니다. 확률 표시만 반올림하며 모멘트 검산에는 정확한 분수를 씁니다. 두 가지 지지·가중 실패는 한 배치가 그럴듯하게 보이는지와 별도로 달라진 기대값으로 확인합니다.

## 기대 손실의 경사와 연결하기

스칼라 손실 $\ell_n(\theta,x)=\theta^2/2-\theta f_n(x)$를 생각합시다. 모집단 목적과 경사는

$$
L_n(\theta)=\frac{\theta^2}{2}-\theta,\qquad
L_n'(\theta)=\theta-1
$$

입니다. 고정한 각 n에서 표본 경사는 $\theta-\widehat I_{n,M}$로 불편이고 분산은 $(2n/3-1)/M$입니다. $\theta=0$에서 배치가 모두 0이면 표본 경사는 0이지만 모집단 경사는 −1입니다. 표의 확률이 그 사건을 수치화합니다. 최적화의 수렴 주장에는 갱신 규칙·보폭·표본 추출 일정이 추가로 필요합니다. 이 손실에서는 이차항을 직접 계산하고 봉우리 적분에 중요도 가중치를 적용합니다.

n을 고정하고 국소 매개변수 구간을 $|\theta|\le C$로 두면 도함수가 $|\theta-f_n(x)|\le C+n$을 만족합니다. 이 유한 상수는 적분 가능한 국소 지배 함수여서 여기의 기대 손실 미분과 일치합니다. 상한이 n에 의존하므로 앞서의 극한 교환 반례를 그대로 검토해야 합니다.

후속 연구에서는 n과 M을 바꾸고, 전부 0인 배치도 보존하며, 같은 평가 예산에서 두 제안 분포를 비교할 수 있습니다. 제안 구간을 의도적으로 줄이거나 위치를 옮기는 대조도 추가하세요. 목표 측도·가중 규칙·편향·분산·비용을 기록합니다. 극한과 기대값을 교환하는 단계와 고정한 n에서 미분하는 단계를 구분하세요. 이 삼각형과 유한 모멘트 검산은 교재에서 직접 구성한 예제입니다.

## 확인 문제와 답

1. n=32의 봉우리 높이와 면적은? **32와 1**입니다.
2. 봉우리의 점별 극한 적분은? **0**입니다.
3. fₙ=x/n의 n=8 적분은? **1/16**입니다.
4. 유한 격자가 봉우리를 놓치면 정리가 증명되나요? **아니요. 표본 격자의 실패입니다.**

5. 불편 추정기는 모든 n에서 분산도 작나요? **아니요. 여기서는 $(2n/3-1)/M$으로 M을 고정하면 커집니다.**
6. 지지 절반만 덮는 제안은 왜 가중 후에도 편향되나요? **기대값이 그 절반의 적분만 계산하며 면적은 $1/2$이기 때문입니다.**

추가 읽기: [MIT 공식 강좌 안내](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). 강좌 범위를 확인해 보충 자료로 연결하며, 아래의 작은 예제와 논증은 직접 구성했습니다.
