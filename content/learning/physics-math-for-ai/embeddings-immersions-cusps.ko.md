## 국소 Rank와 전역 형상은 다른 질문입니다

매끄러운 사상이 immersion이려면 모든 원공간 점에서 미분이 단사여야 합니다. 1차원 원공간에서는 도함수 벡터가 어디서도 0이 아니라는 뜻입니다. embedding은 여기에 원공간과 상 사이의 위상동형 조건도 요구합니다. 국소 미분과 전역 위상을 함께 봐야 합니다.

앞의 두 예제는 θ=π(u+1), u∈[−1,1]의 양 끝을 붙인 원 S¹을 원공간으로 씁니다. 마지막 예제는 실수 전체를 원공간으로 하되 [−1,1]만 그립니다. 원의 매개화 끝점이 반복되는 것을 S¹ 사상의 비단사성으로 잘못 읽지 않도록 원공간을 명시합니다.

## 원은 Embedding입니다

f(θ)=(cosθ,sinθ)의 θ에 대한 속력은 1이고 u에 대한 속력은 π입니다. 미분 rank는 항상 1입니다. 원공간 S¹에서 일대일 연속이며, compact 원공간과 Hausdorff 목표를 이용하면 상에 대한 위상동형임을 알 수 있습니다.

$$
f'(\theta)=(-\sin\theta,\cos\theta),\qquad
\|df/du\|=\pi.
$$

파랑은 상, 주황은 질의점과 정규화한 비영 접방향입니다. 접선분의 고정 길이는 미분 속력을 나타내지 않으므로 속력은 수치로 읽습니다.

## Immersion도 자기 교차할 수 있습니다

figure-eight는 f(θ)=(sinθ,sin2θ)입니다. θ=0과 π가 같은 원점으로 가므로 단사가 아닙니다. 하지만 다음 도함수는 어디서도 0이 아닙니다.

$$
f'(\theta)=(\cos\theta,2\cos2\theta)
$$

cosθ=0이면 cos2θ=−1이기 때문입니다. 따라서 모든 점에서 immersion이지만 embedding은 아닙니다. 교차하는 두 가지는 서로 다른 원공간 점과 접방향입니다. 상의 교차점에 하나의 접공간을 억지로 주면 이 원공간 정보를 잃습니다.

## 단사여도 Immersion이 아닐 수 있습니다

cusp c(u)=(u²,u³)는 두 번째 성분의 연속 세제곱근으로 u를 유일하게 복구하므로 실수 전체에서도 상에 대한 위상동형입니다. 그러나 다음과 같습니다.

$$
c'(u)=(2u,3u^2),\qquad c'(0)=(0,0).
$$

원점에서 rank가 떨어지므로 이 매개화는 위상적 embedding이지만 smooth embedding은 아닙니다. 실험은 영도함수를 정규화해 가짜 방향을 만들지 않고 그 점에서 접선분을 생략합니다.

## 전역 가정을 Rank로 대체할 수 없는 이유

Compact 다양체에서 Hausdorff 다양체로 가는 단사 immersion은 embedding입니다. 필요한 연결 고리는 위상입니다. compact 원공간의 닫힌 부분은 compact이고 그 상도 compact이므로 목표에서 닫힙니다. 따라서 상으로 가는 연속 전단사의 역함수도 연속입니다. 더 일반적으로 보통의 다양체에서 proper 단사 immersion도 embedding입니다. properness는 원공간에서 무한히 멀어지는 부분이 상에서는 국소적으로 축적되는 현상을 막습니다.

Properness가 없는 반례는 실수선을 두 원의 곱에 무리수 비율로 감는 사상입니다.

$$
f(t)=(\cos t,\sin t,\cos(\sqrt2t),\sin(\sqrt2t)).
$$

같은 상을 가지려면 각변위가 2π의 정수배인 동시에 무리수 배율에서도 맞아야 하므로 단사입니다. 속력은 √3이므로 immersion입니다. √2의 유리 근사 p/q를 쓰면 t=2πq는 무한히 커지지만 f(t)는 f(0)에 접근합니다. 상 위의 역함수는 f(0)에서 연속일 수 없습니다. 따라서 단사성과 full rank만으로 전역 embedding을 보장하지 못합니다.

아래 Pell 근사는 이 재방문을 정량적으로 보여줍니다. 수학적 공식에서 첫 원은 정확히 돌아오므로 그 오차를 0으로 두고, 두 번째 원은 작은 위상 q√2−p로 평가합니다. 이 확장 예제는 실습에 새 선택 곡선을 추가한 것이 아닙니다.

| p/q | q | f(0)까지의 대략적 거리 |
|---|---:|---:|
| 3/2 | 2 | 1.027 |
| 17/12 | 12 | 0.185 |
| 99/70 | 70 | 0.032 |

## Python으로 계산 재현하기

```python
from decimal import Decimal, localcontext
from math import pi, sin
with localcontext() as ctx:
    ctx.prec = 60
    root2 = Decimal(2).sqrt()
    p, q = 3, 2
    for _ in range(6):
        error = Decimal(q)*root2-Decimal(p)
        distance = 2*abs(sin(pi*float(error)))
        assert p*p-2*q*q == 1
        print(p, q, round(distance, 9))
        p, q = 3*p+4*q, 2*p+3*q
```

## 직접 풀기와 풀이

cusp에서 u=0과 0.5를 비교하세요. 국소 rank는 0에서 1이 되지만 전역 smooth-embedding 표시는 계속 거짓입니다. 모든 점의 조건은 원점 하나에서 반박됩니다. figure-eight의 θ=0,π에 해당하는 u=−1,0도 비교하세요. 위치는 같고 도함수는 다릅니다.

## 표현 학습과의 연결

학습한 latent map은 rank를 잃거나, 다른 입력을 같은 점에 보내거나, 이웃 위상을 왜곡할 수 있습니다. 서로 다른 실패 모드입니다. 표본 재구성만으로 모집단 다양체 전체의 smooth embedding을 인증하지 못합니다. 저차원 그림을 보존된 기하로 읽기 전에 미분 조건·원공간의 동일시·전역 단사성 가정을 확인하세요.

추가 읽기: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
