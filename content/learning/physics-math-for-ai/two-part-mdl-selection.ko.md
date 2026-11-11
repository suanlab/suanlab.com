## 적합도와 모형을 함께 전송하기

어떤 모형이 데이터를 잘 설명하는지는 가능도로 비교할 수 있습니다. 하지만 모형마다 공짜 매개변수를 무제한으로 허용하면 데이터를 외운 모형이 유리해집니다. 두 부분 MDL은 “어느 모형인가”를 보내는 비용과 그 모형으로 데이터를 보내는 비용을 더합니다.

$$
L(M,x)=L(M)+L(x\mid M).
$$

이 실습은 순서 있는 N개의 Bernoulli 관측을 사용합니다. K개가 1이고 나머지가 0입니다. 네 모형은 p=.5, .25, .5, .75이고 각각 prefix 헤더 0, 10, 110, 111을 가집니다. 같은 p=.5를 두 번 둔 것은 헤더 비용만 바뀌면 어떤 일이 생기는지 보이기 위해서입니다. 이 헤더들은 데이터 비교 전에 고정합니다.

## 로그 비용은 무엇을 세는가

독립 Bernoulli 모형의 순서 있는 한 비트열에 대한 가능도는 pᴷ(1−p)ᴺ⁻ᴷ입니다. 음의 로그는 모형 아래 이상적인 자료 부호 길이입니다. 자료가 개수 K만인 다른 실험이라면 이항 계수가 필요하므로 무엇을 전송하는지 명확히 해야 합니다.

$$
L(x\mid p)=-K\log_2p-(N-K)\log_2(1-p).
$$

표의 값은 분수 비트일 수 있습니다. 실제 단일 비트열 프로토콜을 구현한 것이 아니라 이상 로그 비용을 비교합니다. 구체적인 산술 부호기 등을 사용하면 유한 길이 오버헤드와 끝 표시 규칙을 추가로 분석해야 합니다.

## 수치 예제: N=16, K=4

p=.5는 자료 16비트와 헤더 1비트로 합계 17입니다. p=.25는 자료 약 12.98045비트와 헤더 2비트로 약 14.98045입니다. p=.5의 다른 모형은 자료가 같지만 헤더 3비트라 19로 불리합니다. 파란 막대가 전체, 주황 막대가 자료만의 비용입니다.

$$
L(x\mid .25)=-4\log_2(.25)-12\log_2(.75).
$$

K=8이면 균형 모형의 합계 17이 선택됩니다. K=12, N=16에서는 p=.75의 더 긴 헤더도 더 좋은 적합으로 상쇄됩니다. 표본 반복 배수는 K/N을 유지하면서 로그 가능도 기여를 배수만큼 늘리고, 헤더 비용은 한 번만 냅니다.

## MLE와 일반화에 관한 주의

MLE p̂=K/N의 자료 비용도 표시하지만 이 값을 다른 모형의 전체 비용과 바로 비교하면 매개변수가 무료라는 잘못된 계산이 됩니다. 학습한 실수 p̂를 어느 정밀도로 표현할지, 어떤 모형 집합인지 비용 규칙이 필요합니다. 연속 모형의 refined MDL이나 NML은 이 단순 네 후보 실험보다 더 정교한 정의를 사용합니다.

짧은 설명 선택이 특정 분포 변화에서의 정확도나 인과 해석을 자동 보장하지도 않습니다. AI 모형 비교에서는 데이터 분할, 코드·가중치·사전 공유 비용과 실제 예측 평가를 함께 기록합니다.

## 두 부분 코드와 mixture는 다른 합을 쓴다

현재 네 헤더의 길이 ℓ=(1,2,3,3)는 Σ2⁻ˡ=1을 만족합니다. 각 모형 아래의 자료를 ceil(−log₂Pᵢ(x)) 길이의 prefix 부호로 만들 수 있습니다. 이 길이들의 Kraft 합은 1 이하이기 때문입니다. 헤더 뒤에 그 모형의 자료 부호를 붙이면 전체도 prefix 부호이고, 실제 정수 길이는 표시한 이상 비용보다 1비트 미만 더 듭니다. 단, 외부 N과 공유 모형·부호 배정을 알고 있다는 조건입니다. 브라우저가 그 부호기를 구현했다는 뜻은 아닙니다.

헤더 길이를 prior 질량으로 읽고 전체 후보를 합하면 정규화한 mixture를 얻습니다. 이와 달리 현재 두 부분 비용은 가장 큰 기여 하나를 고릅니다.

$$
m(x)=\sum_{i=1}^4 2^{-\ell_i}P_i(x),\qquad
L_2(x)=-\log_2\max_i[2^{-\ell_i}P_i(x)],
$$
$$
L_2(x)-\log_2 4\le-\log_2 m(x)\le L_2(x).
$$

N=16,K=4에서 두 부분 비용은 14.980450, mixture 비용은 14.592679 bits입니다. 중복 .5 모형은 더 긴 헤더 때문에 최소 선택에서 이기지 못해도 mixture에서는 그 질량을 더합니다. 후보의 중복이나 prior 규칙을 바꾸면 비교 대상도 달라집니다.

## Bernoulli NML의 정확한 정규화와 regret

모든 길이 N의 순서 있는 문자열을 대상으로, 각 문자열에 맞춘 MLE 확률을 합합시다. K=k인 문자열이 binom(N,k)개임을 사용하면

$$
C_N=\sum_{k=0}^N{N\choose k}(k/N)^k(1-k/N)^{N-k},\qquad
q_{\mathrm{NML}}(x)=\frac{P_{\widehat\theta(x)}(x)}{C_N}.
$$

끝점의 0⁰은 1로 둡니다. 이제 q_NML의 확률 합은 1입니다. 정규화하지 않은 MLE를 이미 하나의 공통 자료 분포라고 쓰는 실수를 피합니다. 임의의 정규화 q에 대해 P_MLE(x)/q(x)≤r가 모든 x에서 성립한다면 양변의 확률을 합해 C_N≤r를 얻습니다. 따라서 최악 로그 regret은 적어도 log₂C_N입니다. NML은 모든 x에서 그 비율을 정확히 C_N으로 만들어 하한을 달성합니다.

| N | C_N | 상수 regret, bits |
|---|---|---|
| 4 | 3.218750 | 1.686501 |
| 8 | 4.245018 | 2.085771 |
| 16 | 5.704258 | 2.512039 |

Regret은 q의 로그 손실과 그 자료에 최적으로 맞춘 Bernoulli 모형의 로그 손실 차이입니다. 아래 분수 코드는 정규화·모든 count의 동일 regret 비율을 확인합니다. N은 고정·공유하고, 표는 이상 로그 길이입니다. 연속 모형에서 정규화 적분이 무한하면 같은 정의를 그대로 적용할 수 없습니다.

**직접 풀기:** 각 x에 가장 잘 맞는 MLE 확률들의 합이 왜 1이어야 하지 않나요? **해설:** x마다 서로 다른 모형을 고르기 때문입니다. 하나의 고정 모형의 확률표가 아니며 C_N으로 정규화해야 합니다.

```python
from fractions import Fraction as F
from math import comb,log2

models=[(1,F(1,2)),(2,F(1,4)),(3,F(1,2)),(3,F(3,4))]
assert sum(F(1,2**length) for length,p in models)==1
N,K=16,4
parts=[F(1,2**length)*p**K*(1-p)**(N-K) for length,p in models]
mixture=sum(parts)
two_part=-log2(float(max(parts)))
mix_length=-log2(float(mixture))
assert two_part-2<=mix_length<=two_part
print('two-part ideal length, mixture ideal length:',round(two_part,6),round(mix_length,6))
for n in [4,8,16]:
    # Every ordered string with k ones has this maximized likelihood.
    maxima=[F(k,n)**k*(1-F(k,n))**(n-k) for k in range(n+1)]
    complexity=sum(comb(n,k)*maxima[k] for k in range(n+1))
    q=[v/complexity for v in maxima]
    assert sum(comb(n,k)*q[k] for k in range(n+1))==1
    assert all(maxima[k]/q[k]==complexity for k in range(n+1))
    print('N, NML normalizer, constant regret bits:',
          n,round(float(complexity),6),round(log2(float(complexity)),6))
```


## 확인 문제와 해설

1. p가 같고 헤더만 2비트 길어지면? **전체 비용도 정확히 2비트 늘어납니다.** 적합은 그대로입니다.
2. 반복 배수를 늘리면 헤더도 반복 전송하나요? **현재 실험에서는 아닙니다.** 하나의 모형으로 더 긴 데이터열을 전송합니다.
3. N=16, K=4의 .25 모형은 실제로 정확히 14.98045비트 파일인가요? **아니요.** 이상 길이이며 실제 코덱 비용과 구분합니다.

추가 읽기: [Grünwald and Roos, Minimum Description Length Revisited](https://arxiv.org/abs/1908.08484).
