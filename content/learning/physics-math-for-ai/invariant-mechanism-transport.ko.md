## 기제가 같아도 평균 효과는 달라진다

한 모집단에서 검증한 모형을 구성이 다른 모집단에 적용할 수 있을까? 집단별 처치 효과가 다르면 층별 기제가 같아도 모집단 평균 효과는 달라진다. 이전 가능성은 명시한 공통점과 차이 아래 어떤 source 정보와 target 측정으로 목표 효과를 복원할 수 있는지 묻는다.

실습은 측정한 이진 집단 Z, 처치 X, 결과 Y를 사용한다. 환경 E가 Z 분포를 바꾸고, 선택적으로 E→Y를 추가하면 결과 기제도 달라진다. E는 환경 표시이지 선택된 collider S=1을 조건으로 삼는 변수가 아니다.

## 두 세계의 기제 명시

Source와 target의 Z=1 확률을 s,t라 하고 source 처치 확률을 e0,e1이라 하자.

$$
q(z,x)=0.1+0.4z+(0.1+0.2z)x.
$$

Target은 q*(z,x)=q(z,x)+Delta x를 사용한다. Delta=0이면 조건부 결과 기제가 불변이다. 현재 SCM은 Z가 주어졌을 때 처치·결과 innovation이 독립이어서 층 안의 잔여 교란이 없다고 가정한다. 두 환경의 Z 측정 의미도 같아야 한다.

집단별 source 효과는 0.1,0.3이다. Source ATE=0.1+0.2s, target oracle ATE=0.1+0.2t+Delta다. Oracle은 선언한 전체 모형의 값이며 source 관측으로 식별한 값과 다르다.

## 실제 source 셀에서 이전하기

목표 질량이 양수인 층에서 source의 두 처치 평균을 계산해 차이를 구한다. 이를 target 질량으로 평균한다.

$$
\tau_{transport}=\sum_z
\{E_s[Y\mid Z=z,X=1]-E_s[Y\mid Z=z,X=0]\}P_t(Z=z).
$$

P_t(Z=z)>0인 층에는 source 집단 Support와 두 처치의 Support가 필요하다. 조건부 교환가능성·일관성·결과 불변성도 가정이다. 실습은 source 결합확률에서 실제 조건부 평균을 구한다. 비어 있는 셀을 oracle q로 채우거나 없는 집단에 회귀 외삽을 하지 않는다.

## Target의 Label 없는 자료는 Outcome 불변성을 검증하지 못한다

Source 관측과 target P(Z)는 효과 불변성 가정 아래 재가중 functional을 정한다. 그러나 일반적으로 그 가정을 검정하지는 못한다. Target이 q_source(z,x)를 쓰는 세계와 q_source(z,x)+Delta_z*x를 쓰는 세계는 같은 source 자료·target Z 분포를 가지면서 target ATE가 다르다. 구분하려면 추가 target 결과·실험·정당화된 제한이 필요하다.

정확한 불변성 대신 실질적 민감도 모형으로 |Delta_z|≤d_z를 가정하자. 이 예에서는 전체 범위에서 결과 확률도 유효하다.

$$
\tau_t=\tau_{transport}+\sum_zP_t(z)\Delta_z,\quad
\tau_t\in\left[\tau_{transport}-\sum_zP_t(z)d_z,\ \tau_{transport}+\sum_zP_t(z)d_z\right].
$$

독립 group shift 구간에서 모두 같은 방향의 끝값을 택하면 양 끝을 실현하므로 sharp하다. Source에서 자동 학습한 confidence interval은 아니다. 확률 제약이나 shift 결합 제약이 그 corner를 제외하면 sharpness도 다시 분석해야 한다.

D0=1/20,d1=1/10, source 층 효과 (1/10,3/10), target group-one 비중 t이면 다음과 같다.

| Target t | 이전 효과 | 민감도 구간 |
|---|---:|---|
| 0 | 1/10 | [1/20,3/20] |
| 1/2 | 1/5 | [1/8,11/40] |
| 1 | 3/10 | [1/5,2/5] |

```python
from fractions import Fraction as F
from itertools import product
effect=[F(1,10),F(3,10)];bounds=[F(1,20),F(1,10)]
for t in (F(0),F(1,2),F(1)):
    Q=[1-t,t];center=sum(p*v for p,v in zip(Q,effect));radius=sum(p*v for p,v in zip(Q,bounds))
    candidates=[]
    for signs in product((-1,1),repeat=2):
        shifts=[s*d for s,d in zip(signs,bounds)]
        for z in (0,1):
            q1=F(1,10)+F(2,5)*z+effect[z]+shifts[z]
            assert 0<=q1<=1
        candidates.append(center+sum(p*d for p,d in zip(Q,shifts)))
    assert min(candidates)==center-radius and max(candidates)==center+radius
    print(t,center,min(candidates),max(candidates))
```

화면의 shift 조절기는 공통 Delta 하나를 유지한다. 독립 민감도 모형은 층별 bounded shift를 허용한다. Source 조건부율은 log cell로 정규화해 joint 곱이 표현 범위 아래여도 양의 Support를 유지한다. 실제로 없는 source 층·처치는 여전히 미정이며 outcome oracle로 채우지 않는다. 기제 유사성·Support·민감도 제한·표본 불확실성을 나누어 보고해야 한다.

## 확인 문제와 풀이

s=0.2,t=0.8, e0=e1=0.5, Delta=0일 때 세 효과를 구하라.

**풀이.** Source ATE는 0.14다. 이전값은 0.2×0.1+0.8×0.3=0.26으로 target oracle과 같다. 조건부 기제가 같아도 source 주변 평균을 그대로 옮길 수는 없다. 층별 효과의 이질성과 혼합 비율 변화가 차이를 만든다.

Delta=0.1로 바꾸면 source 관측과 이전값은 그대로 0.26이지만 target oracle은 0.36이다. 집단 재가중만으로 바뀐 처치 반응 기제를 고칠 수 없다. 추가한 E→Y가 이 실패를 표시한다.

## 없는 Support는 별도 문제

s=0,t=1이면 source에 Z=1 사례가 없다. 처치가 무작위여도 이 target 집단의 결과 평균을 비모수 셀 조정으로 복원할 수 없어 이전값은 —다. 선언한 oracle은 여전히 정의된다. 반대로 t=0이면 source 집단 0만 필요하다. 목표 질량이 0인 집단의 부재까지 실패로 처리하지 않는다.

곡선은 target 구성을 바꾼다. 주황은 조정 가능한 곳에만 표시하며 한 끝점만 가능하면 그 점만 그린다. 생략된 부분을 효과 0으로 해석해서는 안 된다.

## 연구에서의 적용 범위

이는 특정 선택 도식과 source 관측 셀을 사용한 측정 공변량 이전 예제다. 일반 transportability는 source 실험과 target 관측을 더 복잡한 그래프에서 결합할 수 있다. 이 재가중 식이 모든 문제를 푸는 알고리즘은 아니다. 예측 점수가 비슷하다는 사실만으로 불변성이 입증되지 않는다. 변할 수 있는 기제, 공유 측정, 목표 모집단, source Support를 먼저 기록하자.

## 원전

Pearl·Bareinboim, [Transportability across studies: A formal approach](https://ftp.cs.ucla.edu/pub/stat_ser/r372.pdf), R-372, 2018 개정판. 이질적 효과의 이진 예제와 source 셀 계산은 직접 구성했다.
