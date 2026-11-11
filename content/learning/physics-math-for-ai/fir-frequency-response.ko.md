## 필터는 주파수마다 다르게 반응합니다

두 tap의 finite impulse response h=[1−r,r]를 생각합니다. r은 0부터 1 사이이며 현재 입력과 한 표본 지연된 입력을 섞습니다. 모든 정수 시각에서 정의한 정상 입력에 대해서는 다음과 같습니다.

$$
y[n]=(1-r)x[n]+rx[n-1],\qquad
H(e^{i\omega})=1-r+re^{-i\omega}.
$$

x[n]=exp(iωn)을 대입하면 공통 지수함수가 빠지고 H가 남습니다. 이동 불변 선형 시스템에서 복소 지수함수는 고유함수이며, 실수 cosine은 이에 따라 크기가 바뀌고 위상이 이동합니다.

## 크기와 위상을 숫자로 해석하기

r=0.5, ω=π/2이면 H=(1−i)/2입니다. 크기는 1/√2≈0.707, 위상은 −π/4입니다. 입력 cosine이 이 비율로 줄고 위상이 이동합니다. 응답 보기에서 tone 보기로 바꾸면 파랑 입력과 실제 두 tap 합으로 계산한 주황 출력을 비교할 수 있습니다.

입력은 n=−1에도 정의되며 양방향으로 계속됩니다. 따라서 정상 응답 계산입니다. n<0에서 0이고 n=0에 켜지는 신호라면 시작 과도 응답과 첫 출력이 다릅니다. 현재 그림에서 그 과도 응답을 추론하면 안 됩니다.

## 영점에는 위상이 없습니다

평균 필터 r=0.5에서 다음 인수분해가 성립합니다.

$$
H(e^{i\omega})=e^{-i\omega/2}\cos(\omega/2),\qquad
H(e^{i\pi})=0.
$$

Nyquist의 교대 입력 [1,−1,1,−1,…]은 정확히 상쇄됩니다. 출력은 0이고 위상은 정의되지 않습니다. 이는 측정된 위상 지연 0과 다릅니다. 화면은 이 정확한 영점에서 위상을 비워 둡니다. 영점 가까이에서도 크기가 작으면 작은 계산·측정 변화에 위상 추정이 민감해질 수 있습니다.

## 세 가지 경우를 비교하기

r=0은 항등 필터입니다. r=1은 한 표본 순수 지연이며 모든 주파수에서 크기가 1이므로 크기 그림만으로는 항등과 구별되지 않습니다. r=0.5는 평균을 내 고주파를 억제합니다. 어느 r에서도 tap 합이 1이므로 DC gain은 1입니다.

## 안정 FIR에도 안정적인 인과적 역이 없을 수 있다

순방향 tap은 비음수이고 합이 1이므로 허용한 FIR의 ℓ¹ 노름은 1이며 BIBO gain도 1 이하입니다. 역의 안정성은 다른 질문입니다. r≠1에서

$$
H(z)=(1-r)+rz^{-1},\qquad
G(z)=\frac1{1-r}\frac1{1+\frac r{1-r}z^{-1}}.
$$

인과적 기하 전개의 공비는 −r/(1−r)입니다. r<1/2에서만 절대 합 가능하며 역의 ℓ¹ 노름은 1/(1−2r)입니다. 1/2<r<1에서는 음의 시간 전개를 쓰면 비인과적이지만 안정이고 노름은 1/(2r−1)입니다. r=1의 역은 한 표본 advance입니다. r=1/2에서는 단위원 위 H가 0이므로 모든 ω에서 G(e^iω)H(e^iω)=1인 ℓ¹ convolution 역이 없습니다.

| r | 안정적인 역 | 역의 ℓ¹ 노름 |
|---|---|---|
| 0 | 항등 | 1 |
| 1/4 | 인과적 기하 꼬리 | 2 |
| 1/2 | ℓ¹ 역 없음 | ∞ |
| 3/4 | 음의 시간 꼬리 | 2 |
| 1 | 한 표본 advance | 1 |

코드는 r=1/4의 인과적 역을 자르고 내부 convolution 계수를 정확히 확인합니다. 끝의 잔차는 (1/3)^N이며 부분 노름은 2로 갑니다. 무한 시간의 역에 대한 명제이며 하나의 유한 주기 기록을 역산하는 것과 구별합니다.

## 선형 위상과 작은 응답을 구분해서 읽기

r=1/2의 영점을 제외한 0≤ω<π에서는 H=e^(−iω/2)cos(ω/2)이므로 group delay −d arg H/dω=1/2 표본입니다. 영점에서는 이 미분이 정의되지 않습니다. 일반 r의 group delay는 주파수에 따라 달라지며 크기 하나로 정할 수 없습니다.

π 근처에서 1−r+r cosω를 직접 계산하면 양의 실수 성분을 잃을 수 있습니다. 구현은 그곳에서 (1−2r)+2r cos²(ω/2), 허수부 −r sinω를 사용하고 지정한 끝점 r=1/2,ω=π의 정확한 0은 보존합니다. ω=π−2⁻⁵¹의 실수부는 약 8.02×10⁻³²이며 크기는 비영입니다. 작은 실수부·작은 gain·정확한 영점은 다른 주장입니다.

**확인 문제:** 비인과적인 역은 모두 불안정한가요? **해설:** 아닙니다. r=3/4가 안정적인 반례입니다. 다만 미래 표본이 필요하므로 지연 없는 온라인 인과 필터로 쓸 수는 없습니다.

```python
from fractions import Fraction as F

r=F(1,4)
for count in [1,2,4,8]:
    g=[(-r/(1-r))**n/(1-r) for n in range(count)]
    convolution=[F(0)]*(count+1)
    for n,v in enumerate(g):
        convolution[n]+=(1-r)*v
        convolution[n+1]+=r*v
    assert convolution[0]==1
    assert all(v==0 for v in convolution[1:count])
    assert abs(convolution[-1])==F(1,3)**count
    norm=sum(map(abs,g))
    assert norm==2*(1-F(1,3)**count)
    print('inverse taps, partial l1 norm, final residual:',count,str(norm),str(abs(convolution[-1])))
# At r=3/4 the stable inverse has negative-time support and the same l1 norm 2.
# At r=1/2 the unit-circle zero rules out any l1 convolution inverse.
```

## 직접 풀기와 AI 연결

상수 입력의 정상 출력은 같은 상수입니다. DC 수치를 확인한 뒤 r=0과 r=1의 tone 보기를 비교해 같은 gain에 숨은 위상 차이를 찾으세요. ω=π, r=0.25이면 H=0.5이므로 교대 출력은 위상 변화 없이 진폭 절반을 유지합니다.

학습한 시간·공간 kernel도 선형 이동 불변 연산자로 볼 수 있을 때 주파수 응답을 가집니다. 비선형 활성화·위치별 다른 가중치·경계 처리는 이 단순한 전역 해석을 깨뜨릴 수 있습니다. 필터 하나의 그래프로 신경망 전체를 설명하기 전에 가정을 명시하세요.

추가 읽기: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
