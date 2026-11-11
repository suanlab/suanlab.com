## 표본 전체를 사용해 미분하기

주기 함수는 정현파들의 합으로 표현할 수 있고 각 모드의 미분은 파수를 곱하는 연산입니다. Spectral 방법은 가까운 두 점의 차이만 보는 대신 전체 주기 표본으로 계수를 구해 미분합니다. 매끄럽고 충분히 해상된 함수에서는 매우 정확하지만 표본에 없는 정보를 새로 만들어내지는 못합니다.

실험은 [0,2π)의 N개 점 xⱼ=2πj/N에서 f=A sin(kx)+0.2 cos(2x)를 관측합니다. 끝점 2π는 0과 같은 위치이므로 중복하지 않습니다. 파랑은 실제 DFT 미분, 주황은 원래 연속 함수의 정확한 미분이며 표에는 중심 차분도 함께 기록합니다.

## 정규화·부호 있는 파수·Nyquist

$$
\widehat f_m=\frac1N\sum_{j=0}^{N-1}f_j e^{-imx_j},\qquad
(Df)_j=\sum_m i\kappa_m\widehat f_m e^{i\kappa_mx_j}.
$$

DFT의 뒤쪽 index는 음의 주파수로 읽습니다. N=16이면 15번은 -1번 모드입니다. 짝수 N의 Nyquist index N/2는 실수 표본에서 양·음 방향을 유일하게 구분할 수 없습니다. 이 페이지는 실수 interpolant의 표준적인 미분 convention에 따라 그 multiplier를 0으로 둡니다. 계수 자체의 에너지는 지우지 않습니다.

전방 변환에 1/N이 있으므로 표본 평균 제곱과 계수 절댓값 제곱의 합이 같아야 합니다. 이 Parseval 검사는 정규화 오류를 찾지만 미해상 연속 신호를 식별해 주지는 않습니다.

## Aliasing이 미분에 미치는 영향

격자 위에서는 sin((k+N)xⱼ)=sin(kxⱼ)입니다. 표본은 같아도 연속 미분은 각각 (k+N)cos((k+N)x), k cos(kx)이므로 격자점에서조차 미분 크기가 다릅니다. N=16에서 k=17은 k=1로 보이며, 정확한 derivative의 첫 성분 17과 수치의 1 사이에 큰 차이가 생깁니다. 매끄러운 파랑 선이 맞는 연속 미분이라는 증거는 아닙니다.

입력 주파수가 Nyquist보다 작다는 표시는 현재 단일 sine 성분에 대한 진단입니다. 일반 데이터에서 진짜 미해상 주파수를 표본만 보고 항상 알아낼 수 있다는 뜻은 아닙니다. 비주기 경계나 불연속에서는 Gibbs와 경계 처리도 별도로 고려해야 합니다.

## Skew 미분에 이산 Chain Rule이 자동으로 생기지는 않는다

지정한 실수 Nyquist 관례의 주기 Fourier 미분은 격자 내적에서 Dᵀ=−D다. 정규화 뒤 unitary 변환과 순허수 부호 multiplier로 증명한다. 따라서 uᵀDu=0으로 선형 수송의 반이산 에너지가 보존된다. 그러나 표본을 곱하는 비선형 항에는 aliasing이 생기며 D(u²)=2uDu는 일반적으로 틀리다.

매끄러운 Burgers에서 split 근사 uₜ=−[D(u²)+uDu]/3를 생각하자. uᵀD(u²)=−(Du)ᵀu²이므로 격자 에너지 미분이 0이다. 잘못된 이산 곱 규칙을 가정하지 않는 대수 상쇄다. 미해상 mode를 복구하거나 dealiased 정확도를 증명하거나 shock 이후 entropy 소산을 제공하지는 않는다.

4절점 실수 spectral 행렬,u=(1,2,3,5)에서 비가중 에너지 E=Σuᵢ²/2를 쓰면

| 반이산 우변 | E′ |
|---|---:|
| −uDu | 9 |
| −D(u²)/2 | −9/2 |
| −[D(u²)+uDu]/3 | 0 |

```python
from fractions import Fraction as F
u=list(map(F,(1,2,3,5)))
def D(v):return [(v[(j+1)%4]-v[(j-1)%4])/2 for j in range(4)]
du,du2=D(u),D([x*x for x in u])
naive=[-x*y for x,y in zip(u,du)]
conservative=[-v/2 for v in du2]
split=[-(a+x*b)/3 for a,x,b in zip(du2,u,du)]
rates=[sum(x*y for x,y in zip(u,rhs)) for rhs in (naive,conservative,split)]
assert rates==[9,F(-9,2),0]
assert sum(u[i]*du[i] for i in range(4))==0
print(rates)
```

아래 4절점 행렬은 정확 유리수 보조 예제이며 live DFT는 8절점부터다. Nyquist 미분은 0이다. 에너지 안정 거동과 해상도는 별개이므로 학습 spectral 연산자에서도 둘을 검사해야 한다. 이차량 보존만으로 비선형 PDE 해를 인증할 수 없다.

## 손계산과 직접 풀기

N=16, k=3이면 sine derivative multiplier는 3입니다. 같은 모드의 중심 차분은 sin(3h)/h를 곱하므로 약 2.35264로 작습니다. spectral은 현재 삼각다항식을 해상하는 한 반올림 수준의 오차를 보입니다.

직접 풀기: k=N/2인 sine의 표본과 정확한 미분을 비교하세요. 풀이: sin(πj)=0이므로 관측 sine는 0입니다. 그러나 연속 미분은 (N/2)cos(πj)로 0이 아닙니다. 현재 Nyquist 미분 convention으로는 이를 복원할 수 없습니다. 입력 크기 A=0이면 그 성분의 미분도 0이므로 주파수 flag만으로 실제 오차를 판단하지 마세요.

## AI의 주파수 연산으로 연결하기

Fourier neural operator나 spectral PDE solver는 주파수 절단, 해상도 이동, 정규화와 경계 가정에 영향을 받습니다. 재구성 오차가 작아도 미분·flux·고주파 에너지는 잘못될 수 있습니다. 해상도를 바꾸며 진짜 연속 목표와 비교하고, FFT 구현을 쓰더라도 파수 convention을 기록하세요. 여기서는 계산 과정을 투명하게 보이기 위해 작은 직접 DFT를 사용합니다.

참고: [MIT 수치 PDE와 spectral 강의 자료](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
