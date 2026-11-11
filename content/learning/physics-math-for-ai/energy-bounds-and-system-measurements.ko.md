## 1. 하한이 세는 연산 정하기

앞 단원의 가정에서 균등 비트 하나의 완전한 초기화는 평균 열 하한 $k_BT\ln 2$를 가집니다. 같은 온도에서 독립인 균등 비트 $N$개를 초기화하는 작업이면 해당 초기화들의 하한은 $Nk_BT\ln 2$입니다. 300 K에서 10억 개에 해당하는 하한은 약 $2.87\times10^{-12}$ J입니다. 실제 추론의 FLOP 수나 논리 상태 전이 수로는 $N$을 바로 알 수 없습니다.

## 2. 단위와 경계, 작업을 맞추기

전력 계측기는 보드 전력을 읽고, 추론 벤치마크는 콘센트에서 에너지를 잴 수 있습니다. 비트 초기화 하한은 정해진 물리 메모리 연산의 열을 다룹니다. 추론당 실측 에너지는 실행 중 선택한 시스템 경계 안에서 쓴 에너지를 포함합니다. 두 값을 비교하려면 연산 수, 작업량, 측정 경계의 관계를 먼저 정해야 합니다. [Tschand 외의 MLPerf Power 벤치마킹 방법](https://arxiv.org/abs/2410.12032)

## 3. 쌍을 이루는 측정값 얻기

각 측정 블록에서 주위 온도, 부하, 계측기 설정을 비슷하게 유지하며 두 구현을 모두 실행합니다. 실행 순서를 번갈아 바꾸거나 무작위로 배정합니다. 시간 분해능의 영향을 줄이려면 블록마다 여러 추론을 충분히 길게 실행할 수 있습니다. 완료 추론당 에너지와 함께 지연 및 결과 품질을 기록합니다. $E_{j,i}=\mu_j+c_i+\eta_{j,i}$ 모형에서는 쌍별 차이 $d_i=E_{A,i}-E_{B,i}$를 구하면 블록 공통의 가산 오차 $c_i$가 상쇄됩니다.

## 4. 반복 오차와 교정 오차 구분하기

측정 블록이 독립이면 쌍별 차이의 평균 표준오차는 $s_d/\sqrt n$으로 추정하며, $s_d$는 표본 표준편차입니다. 하나의 계측기가 두 실행에 같은 불확실한 배율을 적용하면 블록을 늘려도 그 배율 오차는 남습니다. 반복 측정의 변동과 계측기 교정의 불확실성을 따로 기록합니다. 블록 수가 적거나 온도 드리프트가 의심되면 시간 순서대로 차이를 확인한 뒤 신뢰구간을 해석합니다.

## 5. 가상 쌍별 데이터로 계산하기

다음 값은 같은 측정 경계에서의 추론당 밀리줄을 가정한 예시입니다. 차이가 양수이면 해당 측정 블록에서 B의 에너지가 더 작습니다.

```python
from math import isclose, log, sqrt
from statistics import mean, stdev

a = [60, 62, 59, 61, 63, 60]
b = [55, 58, 56, 56, 59, 55]
differences = [x - y for x, y in zip(a, b)]
delta = mean(differences)
standard_error = stdev(differences) / sqrt(len(differences))
scale_uncertainty = 0.01 * abs(delta)
bound_joule = 1e9 * 1.380649e-23 * 300 * log(2)
assert isclose(delta, 26 / 6)
assert isclose(mean([x + 4 for x in a]) - mean([y + 4 for y in b]), delta)
print(f"{delta:.3f} mJ difference, repeatability SE {standard_error:.3f} mJ")
print(f"1% common meter scale: {scale_uncertainty:.3f} mJ")
print(f"illustrative reset bound: {bound_joule:.3e} J")
```

## 6. 주장을 반증할 수 있게 기록하기

이론적 하한의 연산 모형을 명시합니다. 시스템 실측에서는 실행 스크립트, 전력 시각 표본, 품질 및 처리량 조건, 원시 표본에서 완료 작업당 에너지를 구한 계산을 공개합니다. 호스트 전처리를 포함하거나 처리량을 동일하게 맞추는 등 어떤 조건을 바꾸면 결과가 달라질 수 있는지 적습니다.
