## 에너지는 확률의 음의 로그 형태

세 부호 sᵢ∈{−1,1}가 삼각형으로 연결된 작은 모형을 생각합니다. 결합 J와 공통 field h로 에너지를 정의하고 온도 T>0에서 exp(−E/T)를 가중치로 사용합니다. 물리 단위를 생략한 이 예제는 kB=1인 무차원 관례입니다.

$$
E(s)=-J(s_As_B+s_Bs_C+s_Cs_A)-h(s_A+s_B+s_C),\quad
p(s)=e^{-E(s)/T}/Z.
$$

## 모든 상태를 합한 Partition Function

세 이진 부호의 8개 상태를 열거해 Z=Σs exp(−E(s)/T)를 구합니다. J=h=0이면 모든 에너지가 0, Z=8, 각 확률은 1/8입니다. Z는 확률이 아니라 가중치 합이므로 1보다 커도 됩니다.

수치 계산은 로그 가중치의 최댓값 M을 빼서 상대 가중치를 만든 뒤 logZ=M+logΣexp(logw−M)로 배율을 복구합니다. 표의 relative weight를 그대로 더한 값은 원래 Z가 아닙니다.

## 낮은 온도와 대칭

J>0이면 이웃이 같은 방향인 상태를 선호합니다. h=0이면 모든 부호를 뒤집어도 에너지가 같아 평균 전체 부호는 정확히 0입니다. 낮은 온도에서는 +++와 --- 같은 최저 에너지 상태에 질량이 몰립니다. 평균 부호가 0이라고 모든 상태가 균등하다는 뜻은 아닙니다.

## 기대 에너지와 엔트로피

정규화한 확률로 E[E], E[Σs]를 구합니다. 로그 확률 −E/T−logZ를 평균하면 entropy=E[E]/T+logZ입니다. 단위는 nats입니다. 온도에 대한 미분이나 다른 모수에 대한 미분을 읽을 때 이 1/T 인자를 빼먹으면 안 됩니다.

$$
H(p)=-\sum_s p(s)\log p(s)=E[E]/T+\log Z,\quad
\frac{\partial\log Z}{\partial h}=E[\sum_i s_i]/T.
$$

## 큰 에너지 모형의 학습으로 연결

정확한 Z는 상태 수에 지수적으로 많은 합일 수 있습니다. 에너지 기반 학습은 관측 상태의 점수뿐 아니라 이 정규화의 변화도 고려해야 합니다. 작은 8상태 모형에서 가능한 직접 열거를 대규모 신경 에너지 모형에서도 무료로 할 수 있다고 일반화하지 마세요. 근사 표집·변분법 등은 별도의 정확성과 혼합 조건을 갖습니다.

## 정규화 상수의 곡률은 변동이다

유한 상태에서 J,h,T가 유한하고 T>0이면 모든 가중치가 양수이고 미분과 합을 교환할 수 있습니다. M=Σsᵢ, β=1/T로 두고 에너지 E가 β 자체에는 의존하지 않는다고 합시다. log Z를 미분하면 정규화된 가중치가 나타나고, 한 번 더 미분하면 평균의 곱을 빼는 공분산이 나옵니다.

$$
\partial_h\log Z=\frac{E[M]}T,\qquad
\partial_h^2\log Z=\frac{\operatorname{Var}(M)}{T^2}\ge0,
\qquad
\partial_\beta\log Z=-E[E],\quad
\partial_\beta^2\log Z=\operatorname{Var}(E).
$$

고정 J,h에서 ∂T E[E]=Var(E)/T²도 얻습니다. 이 무차원 모형의 열용량입니다. 서로 다른 모수의 혼합 미분은 대응 충분통계량의 공분산에 온도 인자를 곱한 값입니다. 단순히 log Z가 크다는 것보다 곡률이 어떤 상태 변동을 측정하는지가 더 유용합니다.

## 작은 triangle에서 식을 직접 검산하기

J=1,h=0이면 정렬된 두 상태의 에너지는 −3, 나머지 여섯은 1입니다. 따라서 $Z=2e^{3/T}+6e^{-1/T}$이고 정렬 상태의 총 질량 $g=1/(1+3e^{-4/T})$입니다. 두 에너지 준위의 차가 4라 Var(E)=16g(1−g)입니다. 부호 반전 대칭 때문에 E[M]=0이어도 Var(M)는 클 수 있습니다.

| T | 정렬 질량 g | 열용량 Var(E)/T² | field 곡률 Var(M)/T² |
|---|---|---|---|
| .5 | .998995 | .064279 | 35.967828 |
| 1 | .947915 | .789955 | 8.583320 |
| 2 | .711235 | .821520 | 1.672469 |

코드는 여덟 상태의 직접 합·두 에너지 준위의 닫힌식·중심 유한차분을 비교합니다. 유한차분은 허용 오차를 둔 수치 검산이며 미분 항등식의 증명 자체는 아닙니다.

## Energy-based learning의 두 평균

T,J를 고정하고 h만 학습할 때 자료의 평균 부호를 M̄라고 하면, 표본당 로그우도의 h-gradient는 (M̄−E_model[M])/T입니다. 관측 상태의 field 점수만 올리면 안 되고 log Z가 만드는 모형 평균을 빼야 합니다. 음의 로그우도 곡률은 Var_model(M)/T²≥0이므로 이 한 모수 문제는 볼록합니다. 일반 신경망 에너지의 비선형 모수화까지 같은 볼록성을 보장하지 않습니다.

유한한 여덟 상태와 T>0에서는 log Z가 매끄럽습니다. 표의 급격한 변화만으로 열역학적 극한의 비해석적인 상전이를 주장해서는 안 됩니다.

**직접 풀기:** h=0의 평균 부호가 0이면 field 곡률도 0인가요? **해설:** 아닙니다. ±3에 질량이 몰리면 평균은 0이어도 분산과 곡률은 큽니다.

```python
from math import exp, log, cosh
from itertools import product

def moments(T,h=0):
    states = list(product([-1,1],repeat=3))
    energy = [-sum(s[i]*s[(i+1)%3] for i in range(3))-h*sum(s) for s in states]
    weights = [exp(-e/T) for e in energy]
    Z = sum(weights)
    p = [w/Z for w in weights]
    mean_e = sum(q*e for q,e in zip(p,energy))
    var_e = sum(q*(e-mean_e)**2 for q,e in zip(p,energy))
    mean_m = sum(q*sum(s) for q,s in zip(p,states))
    var_m = sum(q*(sum(s)-mean_m)**2 for q,s in zip(p,states))
    return log(Z),mean_e,var_e,mean_m,var_m

for T in [.5,1,2]:
    logZ,mean_e,var_e,mean_m,var_m = moments(T)
    aligned = 1/(1+3*exp(-4/T))
    assert abs(logZ-log(2*exp(3/T)+6*exp(-1/T))) < 1e-12
    assert abs(var_e-16*aligned*(1-aligned)) < 1e-12
    eps = 1e-4
    plus,minus = moments(T,eps),moments(T,-eps)
    curvature = (plus[0]-2*logZ+minus[0])/eps**2
    assert abs(curvature-var_m/T**2) < 1e-5
    temperature_slope = (moments(T+eps)[1]-moments(T-eps)[1])/(2*eps)
    assert abs(temperature_slope-var_e/T**2) < 1e-6
    print('T, aligned mass, heat capacity, field curvature:',
          *(round(v,6) for v in [T,aligned,var_e/T**2,var_m/T**2]))
```


## 확인 문제와 해설

1. logZ를 복구할 때 뺀 최댓값은? **다시 더합니다.**
2. h=0의 평균 부호는? **반전 대칭으로 0입니다.**
3. 낮은 온도가 모든 모형에서 단일 상태만 남긴다는 뜻인가요? **아니요. 최저 에너지 상태가 여러 개일 수 있습니다.**

추가 읽기: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
