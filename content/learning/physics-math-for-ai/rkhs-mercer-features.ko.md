## Kernel은 특징 내적과 함수 공간을 연결한다

양의 준정부호 kernel k는 임의의 유한 입력과 실수 계수에 대해 ∑cᵢcⱼk(xᵢ,xⱼ)≥0을 만족합니다. 모든 원소가 양수라는 조건과 다릅니다. RKHS는 점 평가가 연속인 Hilbert 함수 공간이며 kernel 절편 kₓ=k(·,x)가 평가의 대표가 됩니다.

$$
f(x)=\langle f,k_x\rangle_H,\qquad \|k_x\|_H^2=k(x,x).
$$

따라서 |f(x)|≤‖f‖H√k(x,x)입니다. 앞서 일반 L2에서 실패한 점 평가가 여기서는 공간의 구조에 포함됩니다.

## 두 특징으로 모든 것을 계산하기

[-1,1]의 kρ(x,z)=1+ρxz, ρ>0을 보겠습니다. 명시적 특징은 φ(x)=(1,√ρx)입니다. 함수 f=a+bx를 계수 w=(a,b/√ρ)로 표현하므로 다음 노름을 얻습니다.

$$
\|a+bx\|_H^2=a^2+b^2/\rho.
$$

ρ=1, f=1+x, 질의 x=.5이면 제곱 노름은 2, 평가 대표 노름은 √1.25, 재생 내적은 1.5입니다. Kernel을 바꾸면 같은 함수의 노름도 달라집니다. ρ=0에서는 상수만 남아 b≠0인 대상은 공간 밖입니다. 실습의 ∞는 이 경우 실제 유한 RKHS 노름을 부여하지 않는 확장 표기이며 재생 내적은 —입니다.

## Mercer의 측도와 고유값

Compact 영역의 연속 대칭 PSD kernel과 적절한 유한 Borel 측도를 가정하면 적분 연산자의 spectral expansion을 통해 kernel을 볼 수 있습니다. 이 예는 균등 확률측도 dx/2를 고정하며 정규직교 함수 e₀=1, e₁=√3x의 고유값은 1, ρ/3입니다.

$$
k(x,z)=1\,e_0(x)e_0(z)+\frac\rho3e_1(x)e_1(z).
$$

입력점 −1,0,1에서의 유한 Gram 행렬 고유값은 3,2ρ,0입니다. 적분 연산자 고유값과 같은 숫자가 아닙니다. 표본 개수·측도·가중치가 다르기 때문입니다.

## 그림과 AI 해석

파랑은 질의점의 kernel 절편, 주황은 대상 함수입니다. 두 곡선이 같은 함수일 필요는 없습니다. 재생은 그 두 함수의 **RKHS 내적**에 관한 주장이지 두 곡선의 높이가 같다는 주장이 아닙니다.

Kernel 폭이나 특징 배율은 어떤 함수를 짧은 노름으로 표현할지에 대한 prior를 만듭니다. Gaussian kernel처럼 무한 차원인 RKHS와 이 두 특징의 유한 공간을 구분하고, 일반 Mercer 전개에는 연속성·영역·측도 가정을 확인해야 합니다.

## 재생 성질을 쓰기 전에 공간을 구성하기

실수 대칭 PSD kernel에서 유한합 f=∑a_i k(·,x_i)로 시작하고 내적을 ∑a_i b_j k(x_i,z_j)로 정의합니다. Gram 이차형식은 seminorm입니다. Cauchy–Schwarz에 의해 노름 0인 벡터는 모든 점에서 값도 0이므로 영 벡터들을 몫으로 묶어도 평가와 모순되지 않습니다. 이 공간을 완비화합니다. |f(x)|≤‖f‖√k(x,x) 경계가 평가를 완비화까지 연속 확장하고 kernel 절편을 Riesz 대표로 줍니다. 이것이 RKHS 구성입니다. 원소별 양수만으로 이 논증을 할 수는 없습니다.

두 특징 kernel의 ρ>0에서는 전체 구간의 함수 1과 x가 독립입니다. a+bx의 계수가 유일하므로 유클리드 특징 노름에서 공식이 나옵니다. ρ가 0으로 갈 때 고정한 기울기 함수의 값은 유계여도 RKHS 노름은 발산합니다. ρ=0에서는 소속 공간이 달라지며 단지 아주 큰 유한 노름을 주는 것이 아닙니다.

## Mercer 복원에서 측도의 support를 명시하기

Compact metric 영역, 연속 대칭 PSD k, full support인 유한 Borel 측도는 유용한 충분조건입니다. Compact 양의 적분 연산자로부터 연속 대표를 쓰는 통상적인 Mercer kernel 전개를 얻습니다. 측도는 연산자와 고유값을 결정합니다. 양의 고유값의 고유함수는 치역 폐포를 생성하고 필요하면 kernel 보완 공간을 따로 둡니다. 양의 고유값만으로 모든 L²를 생성한다는 뜻은 아닙니다.

k=1+xz에서 네 측도를 비교합니다. 코드는 유한 Gram 고유벡터와 dx/2의 적분 모멘트 ∫x=0, ∫x²=1/3을 분수 계산으로 대조합니다.

| 측도 | 양의 적분 고유값 | 범위 |
|---|---|---|
| [−1,1]의 dx/2 | 1, 1/3 | 전체 구간 support |
| [−1,1]의 dx | 2, 2/3 | 전체 질량 두 배 |
| −1,0,1의 균등 측도 | 1, 2/3 | 세 점의 가중 Gram |
| δ₀ | 1 | 상수만 관측 |

δ₀에서는 x 특징이 L²(δ₀)의 영 벡터입니다. 양의 고유값 전개는 1만 주고 support 밖 k(1,1)=2를 복원하지 못합니다. L²의 항등식이 더 큰 영역의 모든 점에서 자동 성립하는 것은 아닙니다.

매우 작은 양의 ρ에서는 실습이 기울기를 √ρ로 나눈 뒤 제곱합니다. ρ=10⁻³⁰⁰,b=10⁻²⁰⁰,a=0이면 제곱 노름은 10⁻¹⁰⁰입니다. 유한 노름이 binary64 범위 밖이면 —와 수치 범위 안내를 표시하며, 여기서 ∞는 공간 밖의 대상과 구별합니다. 계수가 모두 정확히 0이면 노름도 0입니다.

**확인 문제:** 비가중 Gram 고유값을 Mercer 고유값이라고 부를 수 있나요? **해설:** 경험 측도의 정규화를 먼저 정해야 합니다. 세 점의 균등 적분 연산자는 G/3을 사용합니다.

```python
from fractions import Fraction as F

rho=F(1)
def moment(power): return F(1-(-1)**(power+1),2*(power+1))
assert moment(0)==1 and moment(1)==0 and moment(2)==F(1,3)
nodes=[F(-1),F(0),F(1)]
G=[[1+rho*x*z for z in nodes] for x in nodes]
for vector,eigenvalue in [([1,1,1],F(3)),([-1,0,1],2*rho),([1,-2,1],F(0))]:
    assert [sum(a*b for a,b in zip(row,vector)) for row in G]==[eigenvalue*b for b in vector]
print('dx/2 integral eigenvalues:',F(1),rho/3)
print('dx integral eigenvalues:',F(2),2*rho/3)
print('uniform 3-point integral eigenvalues:',F(1),2*rho/3,F(0))
# Under delta_0 the x feature is zero almost everywhere for that measure.
print('delta_0 positive integral eigenvalue:',F(1))
assert 1+rho*F(1)*F(1)!=1  # Its eigen-expansion cannot recover k(1,1) off support.
```

## 확인 문제와 해설

1. ρ=.5에서 f=x의 제곱 노름은? **2입니다.** 기울기 비용이 1/ρ입니다.
2. Gram 행렬에 0 고유값이 있으면 kernel이 잘못됐나요? **아니요.** PSD는 singular도 허용합니다.
3. 측도를 dx로 바꾸면 적분 고유값은? **현재 값의 두 배입니다.** Kernel 자체의 식은 그대로여도 적분 연산자는 달라집니다.

추가 읽기: [MIT 9.520, RKHS and Mercer](https://ocw.mit.edu/courses/9-520-statistical-learning-theory-and-applications-spring-2003/resources/class14/).
