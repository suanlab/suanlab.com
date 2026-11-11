## 국소 규칙에는 경계 규칙이 필요합니다

Convolution은 같은 선형 규칙을 위치마다 반복하는 연산입니다. impulse response h는 입력 한 점이 주변 출력으로 어떻게 퍼지는지 나타냅니다. 유한 지지 신호를 무한한 정수 축에 0으로 연장하면 선형 convolution이 되고, 고정 길이 N에서 끝을 처음으로 감으면 순환 convolution이 됩니다.

$$
(x*h)[n]=\sum_k x[k]h[n-k],\qquad
(x*_Nh)[n]=\sum_{k=0}^{N-1}x[k]h[(n-k)\bmod N].
$$

두 식은 서로 다른 연산자입니다. AI 라이브러리에서 convolution이라고 부르는 correlation도 뒤집기·켤레 관례가 다를 수 있으므로 실제 정의를 확인해야 합니다.

## 네 점 입력을 손으로 계산하기

x=[1,0,0,2], h=[1,r]로 둡니다. r=0.5이면 선형 출력은 [1,0.5,0,2,1]입니다. 마지막 1은 입력의 마지막 2가 지연 tap 0.5와 곱해진 결과입니다. 길이 4의 순환 출력은 [2,0.5,0,2]입니다. 마지막 1이 index 0으로 감겨 기존 1에 더해졌습니다.

파랑은 선형 출력 다섯 점, 주황은 순환 출력 네 점입니다. 선택한 출력 길이 밖의 index는 이 보기에서 정의되지 않으며 새로 관측된 0이 아닙니다. 순환 모드에서 index 4를 질의하면 선택 출력 수치가 비어 있는 이유입니다.

## 주파수 곱셈 정리가 적용되는 경계

DFT는 순환 이동을 대각화합니다. convolution 합을 전개한 뒤 유한 주기 합의 index를 다시 정리하면 다음 관계를 얻습니다.

$$
\operatorname{DFT}_N(x*_Nh)[k]=X[k]H[k].
$$

같은 방법으로 선형 출력을 얻으려면 두 입력을 최소 4+2−1=5 길이로 zero padding해야 합니다. 실험은 radix-two FFT를 쓰기 위해 길이 8을 택합니다. 역변환하면 선형 출력 다섯 점 뒤에 0 세 점이 나옵니다. 길이 4로만 padding하면 앞서 본 wrap이 생깁니다. 재구성 오차는 선택한 경계의 참조 출력과 복소수 역변환값을 비교합니다.

## 합과 적분을 교환할 수 있는 적분 가능성부터 명시하기

f,h∈L¹(R)이면 Tonelli로 ∫∫|f(u)h(t−u)|du dt=‖f‖₁‖h‖₁입니다. 따라서 convolution은 거의 모든 곳에서 존재하고 L¹에 속하며, Fourier 변환에 t=u+v를 대입하는 Fubini 교환이 정당화됩니다. 실습의 각주파수 관례에서는 추가 배율 없이 F(f*h)=F(f)F(h)입니다. 시간 함수의 곱을 변환하는 convolution 공식은 다르며 별도 가정이 필요합니다.

h∈L¹,f∈L²이면 가중 Cauchy–Schwarz로 |h*f(t)|²≤‖h‖₁∫|h(u)||f(t−u)|²du입니다. 다시 적분하면 Young 경계를 얻습니다.

$$
\|h*f\|_2\le\|h\|_1\|f\|_2.
$$

유한 지지의 이산 impulse response에는 대응하는 합의 논증이 성립합니다. 임의의 비적분 kernel에는 다른 연산자·분포 틀이 필요하며 형식적인 순서 교환만으로 정리를 증명하지 못합니다.

## 잘라낸 Toeplitz 연산자와 circulant 연산자 비교하기

S를 네 좌표의 순환 한 칸 이동으로 둡니다. h=[1,r]의 주기 연산자 C=I+rS는 S와 가환하고 DFT 모드로 대각화됩니다. 0으로 연장한 출력을 네 위치로 자른 행렬 T는 아래쪽 두 대각선의 Toeplitz 행렬이며 맨 위 오른쪽 wrap 원소가 없습니다. 다른 경계의 유한 연산자이지 같은 출력의 두 알고리즘이 아닙니다.

| r | 0 연장 후 crop 출력 | 주기 출력 | ‖TS−ST‖F |
|---|---|---|---|
| 0 | (1,0,0,2) | (1,0,0,2) | 0 |
| 1/2 | (1,1/2,0,2) | (2,1/2,0,2) | .707107 |
| 1 | (1,1,0,2) | (3,1,0,2) | 1.414214 |

여기서 x=(1,0,0,2), ‖TS−ST‖F=√2|r|이고 CS−SC=0입니다. 코드는 행렬과 commutator를 정확분수로 만듭니다. 유한 crop이 순환 이동 등변성을 깨뜨린다는 내용이며, 모든 정수 위의 완전한 0 연장 convolution이 보통의 평행이동 등변성을 깨뜨린다는 뜻은 아닙니다.

**확인 문제:** 전체 선형 convolution에 FFT를 쓰는 데 zero-padding이면 충분한 이유는? **해설:** 주기가 입력 길이+필터 길이−1 이상이면 비영 출력이 다른 출력 index로 감기지 않기 때문입니다.

```python
from fractions import Fraction as F
from math import sqrt

N=4
S=[[F(int(i==(j+1)%N)) for j in range(N)] for i in range(N)]
x=list(map(F,[1,0,0,2]))
def matmul(A,B): return [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]
for r in [F(0),F(1,2),F(1)]:
    T=[[F(i==j)+r*int(i==j+1) for j in range(N)] for i in range(N)]
    C=[[F(i==j)+r*int(i==(j+1)%N) for j in range(N)] for i in range(N)]
    def commutator(A):
        AS,SA=matmul(A,S),matmul(S,A)
        return sum((AS[i][j]-SA[i][j])**2 for i in range(N) for j in range(N))
    assert commutator(C)==0 and commutator(T)==2*r*r
    T_y=[sum(a*b for a,b in zip(row,x)) for row in T]
    C_y=[sum(a*b for a,b in zip(row,x)) for row in C]
    print('r, cropped, periodic, cropped commutator norm:',str(r),
          [str(v) for v in T_y],[str(v) for v in C_y],round(sqrt(float(2*r*r)),6))
```

## 직접 풀기와 연구에서 확인할 것

r=−1일 때 첫 출력을 예측하세요. 선형 index 0은 1이고, 순환 index 0은 마지막 기여 −2가 더해져 −1입니다. 두 모드를 바꿔 표로 검산하세요. r=0이면 공통 index의 값은 같아지지만 출력 길이는 여전히 다릅니다.

영상 복원이나 Fourier neural operator에서 주파수 곱셈은 경계 관례를 내포합니다. 같은 내부 kernel이라도 zero padding·주기 연장·다른 경계 연장은 서로 다른 가장자리 결과를 만듭니다. 빠른 계산이 가능하다는 사실만으로 그 경계가 데이터에 적합하다고 결론낼 수 없습니다. 정확도를 비교하기 전에 연장과 crop 규칙까지 모형의 일부로 명시하세요.

추가 읽기: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
