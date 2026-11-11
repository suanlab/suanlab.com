## 역전파에는 저장 비용이 있다

Local 미분을 계산하려면 forward 값이 필요합니다. 모든 층의 상태를 보관하면 간단하지만 깊이와 batch 크기에 따라 activation 메모리가 커집니다. Checkpointing은 일부 forward 계산을 반복하는 대신 중간 상태 저장을 줄입니다. 핵심은 checkpoint 이름의 개수가 아니라 어느 값들이 동시에 존재하는지입니다.

실험은 폭 W인 실제 dense chain h_l=tanh(A h_(l-1))을 실행합니다. 모든 행렬 원소는 alpha/W, 초기 성분은 x(i+1)/W이고 출력은 마지막 평균입니다. Full tape와 block 재계산 reverse를 각각 실행하여 입력 벡터와 공유 alpha의 gradient를 비교합니다.

## 경계만 남기고 한 block씩 복원한다

L개 층, block 길이 B에서 0·B·2B·마지막 층을 저장합니다. 마지막 block은 짧을 수 있고 경계 벡터는 ceil(L/B)+1개입니다. 역전파할 때 한 block의 내부만 복원하고 이미 저장한 두 경계를 이용한 뒤 내부를 버립니다.

$$
R=L-\lceil L/B\rceil,\qquad
V_{\mathrm{checkpoint}}=
\left(\lceil L/B\rceil+\min(B,L)+2\right)W.
$$

R은 재계산 forward 층 수입니다. 각 block 끝은 저장되어 있으므로 다시 계산하지 않습니다. V는 경계·가장 큰 block의 내부·adjoint 벡터 두 개를 합친 값 수입니다. 같은 기준의 full tape는 (L+3)W입니다. 복원한 모든 block을 한꺼번에 남기면 이 저장 추정은 성립하지 않습니다.

## 명시한 연산만 세고 실행 시간과 구별한다

Dense matvec 하나는 W²개의 scalar 곱셈을 합니다. 두 방식의 reverse matvec은 L번이고, checkpoint forward에는 R번이 추가됩니다.

$$
M_{\mathrm{full}}=2LW^2,\qquad
M_{\mathrm{checkpoint}}=(3L-\lceil L/B\rceil)W^2.
$$

이 수는 activation·local derivative·parameter gradient의 연산을 제외합니다. Byte 추정은 값 수에 4 또는 8을 곱하며 parameter·optimizer·kernel 임시 저장·이 시연의 독립 비교용 full tape는 제외합니다. 브라우저의 실제 산술은 JavaScript number입니다. Byte 선택은 가상 저장 dtype의 비용만 바꾸며 실제 계산 dtype을 바꾸지 않습니다.

실제 시간은 대역폭·병렬성·cache·kernel launch 등의 영향도 받습니다. 연산 수가 같아도 속도는 같지 않습니다. 이 실험은 비용 관례를 명시한 비교이며 GPU 학습 성능 측정은 아닙니다.

## Gradient까지 확인하기

L=16, W=8, B=4, 값당 4 byte에서 full tape는 608 byte, checkpoint는 320 byte입니다. Dense 곱셈은 2048에서 2816으로 늘고 forward 재계산은 12회입니다. 실제 두 역전파의 gradient는 일치합니다. 파랑·주황은 block별 저장 비율과 곱셈 수 비율입니다.

## 미분할 함수를 바꾸지 않고 일정 최적화하기

균일 chain의 1≤B≤L에서 저장 벡터 수는 ceil(L/B)+B+2다. Ceiling을 무시하면 AM–GM으로 L/B+B≥2√L이다. √L 근처 정수 B를 택하면 저장량은 full tape의 O(WL) 대신 O(W√L)이다. 유한 최적값은 ceiling에 영향받으므로 실제 일정은 허용한 B를 열거하자. 재계산은 L−ceil(L/B)로 남아 최소 메모리와 최소 작업은 다르다.

| L | 한 최소 B | 저장 벡터 |
|---|---:|---:|
| 4 | 2 | 6 |
| 16 | 4 | 10 |
| 32 | 4 | 14 |

```python
from math import ceil
from fractions import Fraction as F
for L in (4, 16, 32):
    vectors, B = min((ceil(L/b)+b+2, b) for b in range(1, L+1))
    assert vectors <= L+3
    print(L, B, vectors, "recomputed:", L-ceil(L/B))
theta, x = F(2), F(3)
original_mask, changed_mask = F(1), F(0)
correct = original_mask**2*theta*x*x
wrong = changed_mask**2*theta*x*x
assert correct == 18 and wrong == 0
print("same forward mask:", correct, "changed mask:", wrong)
```

재계산은 같은 수학적 함수를 재현해야 한다. 무작위 mask·변경 가능한 상태·외부 효과를 일관되게 재생하거나 저장해야 한다. F=(mθx)²/2에서 forward mask m=1,θ=2,x=3이면 gradient는 18이다. Mask 0으로 재계산하면 잘못된 0이 나온다. Mask를 저장하거나 같은 난수 상태를 재생해야 원래 프로그램을 미분한다.

실제 chain은 결정론적이고 두 reverse 일정을 직접 비교한다. Byte 선택은 용량 추정이며 JavaScript 산술을 바꾸지 않는다. 제곱근 일정도 균일 chain 모형이지 임의의 분기 신경망 전체에 보편적으로 최적인 메모리 계획은 아니다.

![결정론적 균일 chain의 block checkpoint 저장량은 깊이의 제곱근 규모로, full tape는 깊이에 비례하여 증가합니다.](/assets/learning/numerical-checkpoint-depth-ko.svg)

## 직접 풀기: 양 끝과 중간 선택

B=1이면 모든 층을 저장해 재계산이 0이고 full tape와 같은 저장량입니다. 한 block이 전체 chain을 포함해도 내부를 한꺼번에 복원하므로 peak 저장은 full tape와 같습니다. 중간 block 크기가 유용할 수 있는 이유를 식으로 설명하세요. 8 byte를 선택하면 메모리 추정만 두 배이고 gradient·연산 수는 같습니다. 빠르다고 주장하려면 별도 timing 실험에 무엇을 통제해야 할지도 적으세요.

더 읽기: [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html).
