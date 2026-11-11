## 좌표를 바꾸면 배율도 확인해야 합니다

여기서 에너지는 표본 크기 제곱의 합입니다. 수학적 노름이며 자동으로 joule 단위의 물리 에너지가 되는 것은 아닙니다. Parseval 관계는 정규화를 맞추면 Fourier 좌표에서도 같은 노름을 읽을 수 있다는 뜻입니다.

정변환에 배율이 없고 역변환에서 N으로 나누는 관례에서는 다음과 같습니다.

$$
\sum_{n=0}^{N-1}|x[n]|^2=\frac1N\sum_{k=0}^{N-1}|X[k]|^2.
$$

오른쪽을 전개하면 교차항에 Σₖexp(2πik(n−m)/N)이 등장합니다. n=m이면 N이고 그 외에는 0이므로 1/N이 정확히 상쇄됩니다. 확률적 근사나 긴 관측 시간의 극한이 아니라 유한 합의 정확한 항등식입니다.

## 여덟 표본 cosine으로 검산하기

N=8, 한 주기, DC offset 0이면 시간 에너지는 4입니다. raw DFT의 비영 계수 두 개는 각각 4이므로 raw spectrum 에너지는 16+16=32입니다. 이를 8로 나누어야 4가 됩니다. 배율 없이 합만 비교하면 에너지가 여덟 배 증가했다고 잘못 판단합니다.

unitary 모드에서는 U=X/√N으로 둡니다. 두 계수 크기는 √2이고 제곱합은 바로 4입니다.

$$
U[k]=X[k]/\sqrt N,\qquad \sum_n|x[n]|^2=\sum_k|U[k]|^2.
$$

정규화가 바뀌어도 입력 에너지는 바뀌지 않습니다. 수치 표는 표시된 spectrum의 raw 에너지, 필요한 제수, 올바르게 정규화한 합을 따로 보여줍니다.

## 같은 index의 점끼리 같다는 뜻이 아닙니다

파란 stem은 시간 index n의 x[n]²이고 주황 stem은 주파수 bin k의 정규화 에너지 기여입니다. 같은 index 범위에 그리는 것은 표시 편의입니다. 정리는 두 합계를 같게 만들며 점별 값을 같게 만들지 않습니다. 시간 표본 1과 주파수 bin 1은 서로 다른 좌표입니다.

입력은 cos(2πcn/N)+d입니다. 한 완전 주기에서는 cosine의 합이 0이므로 d=1을 더하면 에너지는 N만큼 늘어납니다. N=8이면 4+8=12입니다. 주기 수가 일반적인 분수일 때는 교차항이 0이 아닐 수 있으므로 실제 표본 제곱합으로 계산해야 합니다.

## 모든 Fourier 적분의 존재를 주장하지 않고 항등식 확장하기

Schwartz 함수 f,g에서는 g의 Fourier 역변환과 Fubini로 교차 내적 항등식을 얻습니다.

$$
\int_{\mathbb R}f\overline g
=\frac1{2\pi}\int_{\mathbb R}F\overline G,
\qquad \|f\|_2^2=\frac1{2\pi}\|F\|_2^2.
$$

빠른 감소가 순서 교환을 정당화합니다. 역변환은 Gaussian 주파수 cutoff를 넣어 근사항등원과의 convolution으로 바꾼 뒤 극한을 보내 얻을 수 있습니다. 따라서 f↦F/√(2π)는 조밀한 부분공간의 등거리 사상입니다. L² 완비화로 확장하고 역도 같이 확장하면 L²의 Plancherel을 얻습니다. 임의의 L² 대표에서 절대 수렴 적분으로 변환을 계산할 수 있다는 뜻은 아닙니다.

원 위에서도 삼각다항식의 완전성이 Bessel 부등식을 Parseval로 강화합니다. Fejér 근사는 연속 주기 함수에서 조밀성을 증명하고, 그 함수들은 원의 L²에 조밀합니다. 유한 직교 DFT 항등식과 무한 차원의 완전성 정리를 구별해야 합니다.

## Real spectrum의 생략된 켤레 bin에 가중치 주기

실수 벡터와 짝수 N에서는 X[N−k]=conj(X[k])입니다. 0부터 N/2까지만 저장하면

$$
\sum_n|x[n]|^2=\frac1N\left(|X[0]|^2+|X[N/2]|^2
+2\sum_{k=1}^{N/2-1}|X[k]|^2\right).
$$

DC와 Nyquist는 하나씩만 있고 내부 bin에만 2를 곱합니다. x[n]=1+cos(2πn/N)에서 저장 원소를 그대로 합하면 cosine 기여의 절반을 놓칩니다.

| N | 시간 에너지 | 비가중 저장 spectrum / N | 올바른 가중 에너지 |
|---|---|---|---|
| 4 | 6 | 5 | 6 |
| 8 | 12 | 10 | 12 |
| 16 | 24 | 20 | 24 |

코드는 직접 복소 합으로 가중치를 확인합니다. 실수 잔차 벡터의 시간 MSE에는 가중 raw spectrum 합을 N이 아니라 N²로 나눕니다. 이 켤레 계산 외에 다른 주파수 가중치를 주면 좌표만 바꾼 것이 아니라 손실 자체를 바꾼 것입니다.

**확인 문제:** 저장한 real spectrum 전체에 2를 곱해도 되나요? **해설:** 아닙니다. 짝 없는 DC와 짝수 N의 Nyquist를 중복 계산합니다. 홀수 N에는 별도 끝점 규칙이 필요합니다.

```python
from cmath import exp
from math import cos, pi

for N in [4,8,16]:
    x=[1+cos(2*pi*n/N) for n in range(N)]
    X=[sum(v*exp(-2j*pi*k*n/N) for n,v in enumerate(x)) for k in range(N)]
    time=sum(v*v for v in x)
    wrong=sum(abs(v)**2 for v in X[:N//2+1])/N
    corrected=(abs(X[0])**2+abs(X[N//2])**2+
               2*sum(abs(X[k])**2 for k in range(1,N//2)))/N
    assert abs(time-1.5*N)<1e-12 and abs(corrected-time)<1e-12
    print('N, time energy, unweighted rFFT energy, corrected:',N,
          round(time,6),round(wrong,6),round(corrected,6))
# For time MSE rather than total energy, divide the corrected expression once more by N.
```

## 직접 풀기와 연구에서의 사용

c=0, d=−1이면 입력 전체가 0이므로 양쪽 에너지도 0입니다. 이어 c=1.25에서 leakage가 생겨도 Parseval이 유지되는지 확인하세요. 봉우리가 정수 bin에 놓여야만 성립하는 정리가 아닙니다.

AI의 spectral loss는 N의 배율이나 주파수 가중치에 따라 달라집니다. real FFT는 생략한 켤레 bin의 기여도 처리해야 합니다. 영상 공간 손실과 Fourier 공간 손실을 비교하기 전에 변환 관례·평균 또는 합의 reduction·가중치를 모두 적으세요. 비슷한 API라도 최적화 배율이 같다고 보장하지 않습니다.

추가 읽기: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
