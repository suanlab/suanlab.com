## 통신이 기울기를 평가하는 점을 바꾼다

두 worker의 목적을 F₁(w)=(w+1)²/2, F₂(w)=κ(w−1)²/2로 두자. 전역 목적은 같은 가중치의 평균이다. 전역 최적점은 (κ−1)/(κ+1)이며, 곡률이 다르면 지역 최소점 두 개의 단순 평균과 다르다.

각 round는 두 worker를 같은 broadcast 점에서 시작시킨다. 각각 정확한 지역 기울기로 E번 갱신한 뒤 끝점을 평균한다. 비교하는 중앙 알고리즘은 round당 전체 GD를 E번 수행한다. 성분 기울기 평가 수는 같지만 중간 평가점이 다르다. 비동기 지연·압축·확률 batch·불균등 가중치는 모형화하지 않는 동기식 결정론 예제다.

## 한 번만 갱신하면 같은 이유

E=1이면 w−η∇Fᵢ(w)의 평균이 w−η∇F(w)다. 같은 초기점에서는 중앙과 지역 궤적이 정확히 일치한다. 모든 기울기를 같은 점에서 평가하고 목적함수 가중치도 맞춘다는 조건이 중요하다.

E>1이면 worker가 통신 사이에 서로 다른 목적을 따라간다. 끝점을 한 번 평균하는 것과 매 중간 단계에서 기울기를 평균하는 것은 일반적으로 다르다. 이를 local drift라고 부르며 평균 연산의 수치 오류가 원인은 아니다.

## Round map과 고정점 유도

지역 곡률 hᵢ, 최소점 aᵢ인 이차함수에서

$$
w_i^{(E)}=a_i+(1-\eta h_i)^E(w-a_i).
$$

Aᵢ=(1−ηhᵢ)ᴱ, mᵢ=1−Aᵢ로 쓰고 a₁=−1,a₂=1을 넣으면

$$
w^+=Aw+B,\qquad A=1-(m_1+m_2)/2,
\quad B=(m_2-m_1)/2,
\quad w_{fixed}=\frac{m_2-m_1}{m_1+m_2}.
$$

마지막 식은 분모가 0이 아닐 때만 유효하다. η=0이면 항등 map이라 모든 점이 고정점이고 유일한 고정점은 없다. 대수적 고정점이 존재해도 거기로 수렴하려면 |A|<1이어야 한다. 불안정 고정점을 궤적의 예측값으로 읽지 말자.

## 수렴해 보여도 남는 편향

κ=4,η=0.1에서 E=1의 고정점은 전역 최적점 0.6이다. E=4이면 m₁=1−0.9⁴=0.3439, m₂=1−0.6⁴=0.8704이므로 고정점은 약 0.433583이다. A≈0.39285라 round map은 수축하지만 극한의 전역 기울기는 0이 아니다.

통신 round 수열의 수렴이 의도한 전역 목적의 최적화를 뜻하지 않는 실제 반례다. 같은 곡률에서는 이 고정점 편향이 사라진다. 따라서 지역 계산을 늘리는 것이 언제나 좋거나 언제나 나쁘다고 일반화할 수 없다.

## Broadcast Gradient 보정으로 이 고정점 편향을 없앨 수 있습니다

Broadcast 점 w̄에서 지역 gradient를 gᵢ(w)−gᵢ(w̄)+G(w̄)로 바꾸세요. 모든 worker가 처음에는 같은 전체 gradient를 봅니다. 이차 곡률 hᵢ의 지역 점화식을 풀면 다음과 같습니다.

$$
w_i^{(E)}=\bar w-\frac{1-(1-\eta h_i)^E}{h_i}G(\bar w),\qquad
w^+=\bar w-sG(\bar w),\quad
s=\frac12\sum_{i=1}^2\frac{1-(1-\eta h_i)^E}{h_i}.
$$

따라서 보정 round map이 수축하면 고정점은 전역 gradient 방정식을 풉니다. κ=4,η=0.1,E=4에서 s=0.28075, 오차 배율은 0.298125입니다. 보정하지 않은 map은 약 0.433583으로 갑니다. 보정은 새로운 anchor 기울기를 사용하므로 추가 계산·정보 비용이 있습니다.

| 이 예제의 방법 | 수축할 때의 고정점 |
|---|---:|
| 보정 없는 지역 평균 | 약 0.433583 |
| 정확한 Broadcast Gradient 보정 | 0.6 |
| 중앙 전체 Gradient | 0.6 |

투명한 결정론적 control-variate 예제이며 SCAFFOLD의 저장 확률 제어를 모두 구현한 것은 아닙니다. 실제 알고리즘·가정은 [Karimireddy 등](https://proceedings.mlr.press/v119/karimireddy20a.html), local SGD 이론은 [Stich](https://arxiv.org/abs/1805.09767)를 참고하세요. 기존 제어는 보정하지 않은 평균을 유지합니다. 엄격 수축 표시는 round 보정량을 1에서 빼기 전에 검사하여 아주 작은 η를 경계로 오인하지 않습니다.

## Python으로 계산 재현하기

```python
from fractions import Fraction as Q
h=[Q(1),Q(4)]; a=[Q(-1),Q(1)]; eta=Q(1,10); E=4
local=corrected=Q(0); optimum=Q(3,5)
for round_index in range(8):
    raw=[]; fixed=[]; G=(5*corrected-3)/2
    for hi,ai in zip(h,a):
        u=local; v=corrected
        for _ in range(E):
            u-=eta*hi*(u-ai)
            v-=eta*(hi*(v-corrected)+G)
        raw.append(u); fixed.append(v)
    local=sum(raw)/2; corrected=sum(fixed)/2
    print(round_index+1,float(local),float(corrected),float(corrected-optimum))
s=sum((1-(1-eta*hi)**E)/hi for hi in h)/2
assert s==Q(1123,4000) and 1-s*Q(5,2)==Q(477,1600)
```

## 직접 풀기와 풀이

w₀=0,κ=4,η=0.1,E=4로 여덟 round를 수행하자. 지역 평균은 약 0.433337, 중앙 GD는 약 0.599940에 이른다. 지역 평균에서 전역 기울기는 약 −0.416657이다. 뒤쪽 round의 점 변화가 작다는 종료 기준만으로 왜 충분하지 않은지 설명하자.

E=1과 E=4를 같은 총 기울기 계산량에서 비교하도록 round 수를 조절하고 통신 예산과 계산 예산을 함께 기록하자. 분산 학습 연구에서는 이질성·동기화·집계 가중치·어느 목적의 정상성을 측정하는지를 명시해야 한다.

더 읽기: [Bottou, Curtis와 Nocedal](https://arxiv.org/abs/1606.04838).
