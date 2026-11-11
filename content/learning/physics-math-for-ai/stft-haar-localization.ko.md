## 주파수가 언제 등장했는지 묻습니다

전체 신호의 스펙트럼 하나만 보면 국소 변화의 위치가 숨겨집니다. 여기서는 32개 표본의 앞 절반에 cosine bin 2, 뒤 절반에 bin 6을 넣습니다. 창을 옮기면서 선택한 위치 근처의 진동을 읽습니다. 창이 기록 범위 밖으로 나가면 명시적으로 0을 채웁니다.

창 길이를 L, 시작을 center−L/2로 두고 대칭 Hann 창을 곱합니다.

$$
w[j]=\tfrac12-\tfrac12\cos\frac{2\pi j}{L-1},\qquad
S_c[k]=\sum_{j=0}^{L-1}x[\mathrm{start}+j]w[j]
 e^{-2\pi i k(\mathrm{start}+j)/32}.
$$

지수에 전체 index를 넣어 위상 관례를 고정했습니다. 창 내부 index를 쓰면 크기는 같고 위상은 달라집니다. 그림은 고정 32-bin 격자의 |S|/Σw이며, 일반적인 불편 sine 진폭 추정량이나 무정규화 spectrum이 아닙니다.

## 두 위치의 창을 비교하기

center 8, L=16에서 원신호는 bin 2이지만 표시된 최대 bin은 1입니다. 창에 한 주기만 들어가 양·음 주파수 Hann lobe가 겹치기 때문입니다. bin 0,1,2의 크기는 약 0.5365, 0.5371, 0.5119입니다. 최대 bin을 무조건 실제 tone 주파수로 읽으면 안 됩니다. center를 24로 옮기면 최대 bin은 6이 됩니다. center 16 부근에서는 양쪽이 섞입니다. 짧은 창은 변화의 위치를 세밀하게 보지만 주파수 봉우리는 넓힙니다. 32점 padding은 창을 곱한 스펙트럼을 촘촘히 읽을 뿐 실제 32개 비영 관측의 분해능을 만들지 않습니다.

표는 복소 계수를 보여줍니다. 수치 표의 Σ|S|²/32는 창을 곱한 시간 신호 에너지와 같습니다. 이 Parseval 검사는 전체 spectrum을 사용하며, 최대 bin 검색만 0부터 16까지로 제한합니다.

## Haar는 국소 평균과 차이를 반복합니다

Haar 모드에서는 Hann을 곱하지 않은 원래 블록에 다음 정규직교 변환을 재귀 적용합니다.

$$
a=(x_0+x_1)/\sqrt2,\qquad d=(x_0-x_1)/\sqrt2,
\qquad a^2+d^2=x_0^2+x_1^2.
$$

평균들을 다시 변환해 가장 거친 계수 하나가 남을 때까지 반복합니다. 출력은 가장 거친 scaling 계수 뒤에 거친 detail부터 세밀한 detail까지 이어집니다. 이 index는 STFT의 주파수 bin이 아닙니다. 보존하는 에너지도 Hann 적용 신호가 아닌 원래 블록의 에너지입니다. 선택 index가 L 이상이면 대응 계수가 없으므로 정의되지 않은 값으로 표시합니다.

## 역변환의 창 조건을 가정하지 말고 유도하기

분석 frame에 x[n]w[n−mR]를 넣고 완전한 역 DFT 후 합성 창 v[n−mR]를 곱한다고 합시다. 더하면

$$
y[n]=x[n]D[n],\qquad D[n]=\sum_m w[n-mR]v[n-mR].
$$

Spectrum을 수정하지 않았을 때 정확한 복원에는 D[n]=1 또는 알려진 비영 D[n]으로의 나눗셈이 필요합니다. 같은 실수 창을 분석·합성에 썼는데 어떤 좌표의 D[n]=0이면 덮는 창이 모두 그곳에서 0이어서 정보를 잃습니다. 분모가 양수인 곳에서는 v_m[n]=w[n−mR]/Σ_j w²[n−jR]가 유효한 dual입니다.

코드는 별도 길이 16의 주기 기록, 길이 8 창, 완전한 frame 변환과 overlap-add를 씁니다. 기존 0 연장 단일 frame 실습을 몰래 바꾼 것이 아닙니다.

| 분석·합성에 각각 적용한 창 | Hop R | 최소 Σw² | 최대 Σw² |
|---|---|---|---|
| 분모 L인 periodic Hann | 4 | .5 | 1 |
| 분모 L−1인 symmetric Hann | 4 | .409079 | .903421 |
| Periodic Hann의 제곱근 | 4 | 1 | 1 |
| 겹침 없는 periodic Hann | 8 | 0 | 1 |

반씩 겹친 periodic Hann은 Σw=1이지만 두 번 적용하면 Σw²는 일정하지 않습니다. 양의 overlap 합으로 나누거나 제곱근 창 쌍을 쓰면 교정됩니다. 실습의 symmetric Hann은 같은 이름이라는 이유만으로 periodic 식을 물려받지 않습니다.

## 유한 Haar 변환을 L² 기저로 연결하기

[0,1)에서 φ=1, 첫 절반에서는 1·둘째 절반에서는 −1인 ψ를 둡니다. 각 dyadic 셀에 지지된 ψ_jk(t)=2^(j/2)ψ(2^jt−k)는 노름 1이고 적분 0입니다. 같은 층은 지지가 분리되어 직교합니다. 더 세밀한 wavelet은 거친 wavelet이 상수인 곳에서 적분 0이어서 층 사이에도 직교합니다. φ와 j≥0,0≤k<2^j를 함께 씁니다.

이들의 span은 모든 dyadic 계단 함수를 포함합니다. 연속 함수의 균등 근사와 연속 함수의 L² 조밀성으로 이 계단 함수들도 L²[0,1]에 조밀합니다. 따라서 구간의 완전한 정규직교 기저입니다. 유한 평균/detail 재귀 변환은 그 유한 차원 대응이며, 계수 index는 규모·위치 좌표이지 Fourier 주파수가 아닙니다.

**확인 문제:** 순방향 STFT spectrum이 잘 나왔다고 정확한 역복원이 증명되나요? **해설:** 아닙니다. 창의 덮임·합성 가중치·hop과 경계가 역조건을 결정합니다.

```python
from math import cos, sin, pi, sqrt
from cmath import exp

N,L=16,8
signal=[1+sin(.4*n) for n in range(N)]
def dft(x,inverse=False):
    n=len(x)
    return [sum(v*exp((1 if inverse else -1)*2j*pi*k*j/n) for j,v in enumerate(x))/(n if inverse else 1) for k in range(n)]
for name,hop,window in [
    ('periodic Hann',4,[.5-.5*cos(2*pi*j/L) for j in range(L)]),
    ('symmetric Hann',4,[.5-.5*cos(2*pi*j/(L-1)) for j in range(L)]),
    ('root periodic Hann',4,[sqrt(.5-.5*cos(2*pi*j/L)) for j in range(L)]),
    ('periodic Hann, no overlap',8,[.5-.5*cos(2*pi*j/L) for j in range(L)])]:
    output=[0.0]*N
    D=[0.0]*N
    for start in range(0,N,hop):
        frame=[signal[(start+j)%N]*window[j] for j in range(L)]
        restored=dft(dft(frame),True)
        for j in range(L):
            index=(start+j)%N
            output[index]+=restored[j].real*window[j]
            D[index]+=window[j]**2
    if min(D)>1e-14:
        recovered=[a/b for a,b in zip(output,D)]
        assert max(abs(a-b) for a,b in zip(recovered,signal))<1e-13
    else: assert D[0]==0  # An input coordinate is unobserved by all analysis windows.
    print(name,round(min(D),6),round(max(D),6),min(D)>1e-14)
# Periodic framing here is separate from the live lab's finite zero-extended single frame.
```

## 직접 풀기와 적용 범위

[1,1]은 [√2,0], [1,−1]은 [0,√2]로 바뀝니다. 차이가 국소 변화를, 평균이 거친 수준을 읽는 이유를 설명해 보세요. 창을 기록 끝으로 옮겨 padding 개수도 확인하세요.

STFT와 wavelet은 음성·과학 AI에서 국소·다중 규모 특징을 만듭니다. 이 실험은 이산 정규직교 Haar와 개별 STFT frame을 구현합니다. 모든 wavelet family나 overlap-add 역변환 구현은 아닙니다. STFT의 정확한 복원에는 분석·합성 창과 hop 조건을 추가로 확인해야 합니다.

추가 읽기: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
