## 유리식만으로 변환을 정할 수 없습니다

변환은 신호 전체의 기여를 적분하거나 합합니다. 간단한 유리식이 나와도 원래 적분·급수가 실제 수렴하는지는 별개입니다. 수렴 영역 ROC도 답의 일부입니다. 여기서는 bilateral 변환을 사용하며, 초기값 문제의 unilateral 관례와 구별합니다.

a>0에서 causal 지수함수와 음수 부호를 붙인 anti-causal 지수함수를 비교합니다. 두 번째 함수의 음수 부호는 의도한 것입니다.

$$
\begin{aligned}
\mathcal L\{e^{-at}u(t)\}(s)&=\frac1{s+a},&&\Re s>-a,\\
\mathcal L\{-e^{-at}u(-t)\}(s)&=\frac1{s+a},&&\Re s<-a.
\end{aligned}
$$

첫 적분의 크기는 양의 시간에서 exp(−(Re s+a)t)처럼 줄어들어야 합니다. 두 번째는 음의 무한대 방향으로 적분하므로 부등호가 반대입니다. t=0 한 점의 값은 적분에 영향을 주지 않습니다.

## 식이 같아도 신호와 존재 영역은 다릅니다

a=1, s=0이면 causal 적분은 1입니다. anti-causal 적분은 발산하지만 1/(s+1)에 대입하면 여전히 1이 나옵니다. 수치 표가 rational continuation과 실제 수렴 변환을 분리하는 이유입니다. s=−1은 극점이므로 어느 쪽에도 유한한 값을 주지 않습니다. s=−2에서는 anti-causal 변환이 존재하고 −1입니다.

파란 그림은 선택한 신호의 유한 시간 구간만 보여줍니다. 왼쪽 꼬리가 지수적으로 커지는 신호를 그림에서 잘랐다고 전체 적분이 수렴하지는 않습니다. 빈 점과 채운 점은 끝점의 값 관례를 구별합니다.

## 이산 시간에서는 원형 ROC가 나옵니다

bilateral Z 변환은 모든 정수에 대해 x[n]z⁻ⁿ을 더합니다.

$$
\begin{aligned}
\mathcal Z\{a^nu[n]\}(z)&=\frac z{z-a},&&|z|>a,\\
\mathcal Z\{-a^nu[-n-1]\}(z)&=\frac z{z-a},&&|z|<a.
\end{aligned}
$$

기하급수의 공비 조건 때문에 수렴 영역이 수직 반평면 대신 원 안팎이 됩니다. a=0.5, z=0.25이면 anti-causal 합은 −1입니다. 음의 시간 급수를 −Σₘ≥₁(z/a)ᵐ으로 쓰면 z=0에서 값 0도 정의됩니다. causal 급수는 그곳에서 수렴하지 않습니다.

## 고리 모양 ROC에서 양방향 안정 응답 유도하기

같은 두 극점의 유리식에도 세 가지 sidedness가 가능합니다.

$$
H(z)=\frac z{z-1/2}+\frac z{z-2}.
$$

두 기하 항이 모두 오른쪽 신호이면 ROC는 |z|>2이고 오른쪽으로 자랍니다. 둘 다 음수 부호의 왼쪽 신호이면 ROC는 |z|<1/2이며 한 꼬리가 왼쪽으로 자랍니다. 가운데 1/2<|z|<2에서는 첫 항이 오른쪽, 두 번째 항이 음수 왼쪽 신호입니다.

$$
h[n]=(1/2)^n u[n]-2^n u[-n-1],
\qquad \sum_{n\in\mathbb Z}|h[n]|=3.
$$

단위원이 가운데 ROC에 들어가므로 보통의 연속 주파수 응답이 존재합니다. 단위원 밖 극점이 하나 있어도 이 시스템은 안정이고 비인과적입니다. “모든 극점이 단위원 안”이라는 익숙한 규칙은 인과적 유리 시스템을 전제하며 그 가정을 지우면 결론이 달라집니다.

| 양쪽 꼬리 절단 N | 부분 DC 합 | 부분 절댓값 합 |
|---|---|---|
| 1 | 1/2 | 3/2 |
| 2 | 3/4 | 9/4 |
| 4 | 15/16 | 45/16 |
| 8 | 255/256 | 765/256 |

코드는 오른쪽 n=0,…,N−1과 왼쪽 n=−1,…,−N을 합합니다. DC gain은 1, 절댓값 합은 3으로 가며 같은 양이 아닙니다. 이 두 극점 예제는 실습의 한 극점 제어와 별도입니다.

## Bounded-input bounded-output 안정성의 의미 증명하기

이산 impulse response가 절대 합 가능하면 |(h*x)[n]|≤‖h‖₁‖x‖∞로 BIBO 안정입니다. 반대로 유한 지지의 유계 시험 입력을 늘려 가며 x[−k]=sign(h[k])를 고르면 0에서의 출력은 해당 |h[k]|의 합입니다. 이 합들이 유계가 아니면 공통 BIBO gain이 없습니다. 복소 h에는 sign 대신 켤레 위상을 씁니다.

연속 시간의 보통 impulse-response 함수에는 ∫|h|를 쓰는 대응 논증이 있습니다. 초기조건·unilateral 변환·분포 impulse에는 각각의 정식화가 필요합니다. 잘라 그린 꼬리나 ROC 밖의 유한 유리식 값으로 적분 가능성을 판단하지 마세요.

**확인 문제:** H에 z=1을 대입하면 세 ROC 모두의 안정성이 증명되나요? **해설:** 아닙니다. 단위원은 가운데 고리에만 들어갑니다. 해석적 연장은 원래의 수렴 합과 다릅니다.

```python
from fractions import Fraction as F

# h[n]=(1/2)^n for n>=0, and -2^n for n<=-1.
# Its Z transform has ROC 1/2<|z|<2 and includes z=1.
for N in [1,2,4,8]:
    causal=sum((F(1,2)**n for n in range(N)),F(0))
    anticausal=-sum((F(1,2)**m for m in range(1,N+1)),F(0))
    dc=causal+anticausal
    absolute_sum=causal-anticausal
    assert dc==1-F(1,2)**N
    assert absolute_sum==3*(1-F(1,2)**N)
    print(N,str(dc),str(absolute_sum))
print('limits: DC gain 1; absolute impulse-response sum 3')
# Changing the ROC changes sidedness even though z/(z-1/2)+z/(z-2) stays the same.
```

## 직접 풀기와 시스템 해석

a=1.5, z=1에서 어느 쪽 변환이 존재할까요? 1<1.5이므로 anti-causal 쪽만 존재하고 값은 −2입니다. 두 모드의 readout을 확인한 뒤 z=1.5의 극점으로 옮겨 보세요.

주파수 응답을 읽을 때 Laplace의 허수축과 Z의 단위원이 중요합니다. 이를 보통 Fourier 응답으로 해석하려면 해당 집합이 ROC에 들어가는지 확인해야 합니다. 인과성·안정성과 극점 위치의 관계는 신호 또는 시스템과 ROC를 함께 명시해야 하며 유리식 하나만으로 판단할 수 없습니다.

추가 읽기: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
