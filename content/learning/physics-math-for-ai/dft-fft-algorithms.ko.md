## 같은 변환을 두 알고리즘으로 계산합니다

DFT는 N개 복소 표본을 N개 복소 계수로 바꿉니다. 여기서는 정변환에 배율을 붙이지 않고 역변환에서 N으로 나눕니다.

$$
X[k]=\sum_{n=0}^{N-1}x[n]e^{-2\pi i kn/N},\qquad
x[n]=\frac1N\sum_{k=0}^{N-1}X[k]e^{2\pi i kn/N}.
$$

FFT는 이 식을 분해해 빠르게 계산하는 알고리즘입니다. 새로운 변환이 아니며 입력에 없던 정보를 늘리지도 않습니다. 실험은 직접 DFT와 radix-two FFT를 모두 계산하고 복소 계수 차이까지 비교한 다음 크기를 그립니다.

## cosine으로 숫자를 검산하기

N=8, 창당 한 주기, 위상 0이면 x[n]=cos(2πn/8)입니다. Euler 항등식으로 양·음 주파수 지수함수로 반씩 나누면 X[1]=4, X[7]=4이고 나머지는 0입니다. 그림은 두 쪽 크기 |X|/N이므로 두 봉우리 높이는 각각 0.5입니다. 내부 bin을 두 배로 만드는 단측 진폭 그림과 다릅니다.

표에는 부호 있는 창당 주기 수도 있습니다. bin 7은 −1을 나타냅니다. 짝수 N의 bin N/2는 하나뿐인 Nyquist bin이므로 두 번 세지 않습니다. 위상을 바꾸면 복소 계수가 회전하는데 크기 그림만으로는 보이지 않을 수 있습니다. 실수부·허수부도 함께 확인하세요.

## radix two가 계산을 줄이는 방식

입력을 짝수 index와 홀수 index로 나누어 각각 절반 길이로 변환합니다.

$$
X[k]=E[k]+W_N^kO[k],\qquad
X[k+N/2]=E[k]-W_N^kO[k],\quad W_N=e^{-2\pi i/N}.
$$

이 한 쌍을 butterfly라고 합니다. N=8이면 butterfly는 (N/2)log₂N=12개이며 직접 합의 항은 64개입니다. N=64에서는 각각 192개와 4096개입니다. 서로 다른 단위의 알고리즘 계수이므로 실행 시간 비율이나 동일 비용의 기계 연산 수로 해석하면 안 됩니다. 메모리 접근·상수·하드웨어도 속도에 영향을 줍니다.

## 정수가 아닌 주기를 넣으면

창당 주기를 1.25로 바꾸세요. 관측 창을 주기 연장할 때 대개 경계가 어긋나 여러 bin으로 성분이 퍼집니다. 이것은 FFT 오차가 아닙니다. 직접 DFT도 같은 결과를 냅니다. 창당 주기를 고정한 채 N만 늘리면 정규화한 창을 더 촘촘히 표본화하는 것이며 같은 물리 tone을 더 오래 관측하는 상황이 자동으로 되지는 않습니다.

## 유한 기하합으로 leakage 유도하기

복소 tone x[n]=exp(2πifn/f_s), 0≤n<M에서 주파수 ν의 정규화 유한 기록 변환은

$$
A_M(\nu)=\frac1M\sum_{n=0}^{M-1}e^{2\pi i(f-\nu)n/f_s}
=\frac{e^{\pi i(M-1)d}}M\frac{\sin(\pi Md)}{\sin(\pi d)},
\qquad d=\frac{f-\nu}{f_s}.
$$

기하합 항등식을 쓰고 d가 정수이면 제거 가능한 극한을 택합니다. 실수 cosine은 복소 위상을 가진 이런 두 항을 합하므로 lobe가 간섭할 수 있습니다. 고립된 복소 tone의 첫 영점은 ν=f±f_s/M이며 실제 M개 표본의 관측 시간 M/f_s가 이 간격을 결정합니다.

## Zero padding은 spectrum을 읽는 위치를 바꾼다

M=8,f_s=8 Hz,f=1.25 Hz를 고정한 뒤 L≥M으로 padding합니다. FFT는 같은 유한 기록 변환을 ν=kf_s/L에서 읽습니다. 격자 간격은 f_s/L이지만 첫 영점은 .25와 2.25 Hz로 유지됩니다.

| Padding한 변환 길이 L | 격자 간격, Hz | 가장 큰 표본 주파수, Hz | 정규화 크기 |
|---|---|---|---|
| 8 | 1 | 1 | .90176420 |
| 32 | .25 | 1.25 | 1 |
| 128 | .0625 | 1.25 | 1 |

봉우리가 높아진 것은 같은 lobe를 더 잘 읽은 결과이지 새로운 자료나 더 좁은 lobe가 아닙니다. 코드는 세 길이를 모두 직접 복소 합으로 계산하고 두 해석적 영점을 검산합니다. 한 lobe를 분리하려고 복소 tone을 썼으며 실습의 실수 cosine에는 켤레 성분이 있습니다.

Radix-two의 butterfly 점화식은 B(N)=2B(N/2)+N/2,B(1)=0입니다. log₂N개 층마다 N/2개이므로 B(N)=(N/2)log₂N을 얻습니다. 역 butterfly를 재귀 층마다 2로 나누면 전체 배율은 1/N입니다. 마지막에 한 번 나누는 구현도 일관되게 적용하면 같습니다.

**확인 문제:** L을 8에서 128로 늘리면 관측 시간이 열여섯 배가 되나요? **해설:** 아닙니다. 같은 f_s에서 실제 표본을 더 얻어야 시간이 늘어납니다. Padding에는 관측이 추가되지 않습니다.

```python
from cmath import exp
from math import pi, sin

M,fs,f=8,8,1.25
samples=[exp(2j*pi*f*n/fs) for n in range(M)]
def slice_value(nu):
    return sum(v*exp(-2j*pi*nu*n/fs) for n,v in enumerate(samples))/M
for L in [8,32,128]:
    # Zero-pad the same eight observations. No extra observation time is added.
    padded=samples+[0j]*(L-M)
    spectrum=[sum(v*exp(-2j*pi*k*n/L) for n,v in enumerate(padded))/M for k in range(L)]
    k=max(range(L),key=lambda k:abs(spectrum[k]))
    print('FFT grid length, spacing Hz, peak Hz, magnitude:',
          L,fs/L,k*fs/L,round(abs(spectrum[k]),8))
for nu in [f-fs/M,f+fs/M]: assert abs(slice_value(nu))<1e-14
# These first zeros remain at f+-1 Hz for every padded length.
```

## 직접 풀기와 단위 확인

주기 수와 위상을 모두 0으로 두면 상수 입력입니다. X[0]=N만 남아 정규화 크기는 1이어야 합니다. 두 알고리즘과 역변환 오차로 검산하세요. 실제 Hz를 붙이려면 표본 간격이 필요하며 bin k는 부호를 접기 전 kfₛ/N입니다. index·창당 주기·radian/sample·Hz를 변환 계수 없이 섞지 마세요.

추가 읽기: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
