## 1. 회로는 순서가 있는 계산이다

양자 회로에는 어떤 연산을 어느 선에 언제 적용하는지가 함께 들어 있다. 게이트 이름만 나열해서는 계산을 정의할 수 없다. 각각 올바른 게이트라도 순서를 바꾸면 결과가 달라진다. 양자 머신러닝의 학습 회로를 이해하거나 시뮬레이터의 선 번호 오류를 찾으려면 이 구별이 먼저 필요하다.

복소 벡터, 텐서곱, 유니터리 행렬을 안다고 가정한다. 먼저 잡음이 없는 순수 상태를 다루고, 잡음과 측정은 다음 채널 단원으로 연결한다. 유한 실험은 모든 진폭을 확인할 수 있도록 1–3큐비트를 사용한다. 이론 자체가 세 큐비트로 제한되는 것은 아니다.

A를 먼저, B를 나중에 적용하면 결과는 B A|ψ〉이다. 보통 회로 그림의 시간은 왼쪽에서 오른쪽으로 흐르지만, 행렬곱에서는 오른쪽 연산부터 작용한다. 계산 전에 말로 적힌 순서를 식으로 옮기자.

$$
|\psi_k\rangle=U_k|\psi_{k-1}\rangle,\qquad
|\psi_m\rangle=U_m\cdots U_2U_1|\psi_0\rangle.
$$

물체를 서로 다른 축으로 차례로 회전시키는 장면을 떠올려 보자. 각 회전은 되돌릴 수 있어도 회전 순서를 바꾸면 최종 방향이 달라질 수 있다. 이 비유는 합성 순서를 설명한다. 양자 진폭에는 여기에 상대 위상이라는 정보도 있다.

## 2. 간섭을 보려면 경로를 다시 합쳐야 한다

Hadamard 게이트는 기저 상태를 두 진폭으로 나눈다. 위상 게이트는 그 진폭들이 다음 Hadamard에서 합쳐지는 방식을 바꾼다. 중간 상태를 확률만으로 적으면 이 정보를 잃는다.

$$
H=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad
H|0\rangle=\frac{|0\rangle+|1\rangle}{\sqrt2}.
$$

|0〉에 H, Z, H를 적용하자. 첫 H는 더하기 상태를 만들고, Z는 빼기 상태로 바꾸며, 마지막 H는 |1〉을 만든다. 반면 Z, H, H는 |0〉으로 돌아온다. 같은 게이트들을 썼지만 출력은 다르다. 한 입력의 확률만 확인하지 말고 HZH=X라는 연산자 항등식도 확인하자.

T=diag(1,e^{iπ/4})로 두면 다음 예에서는 위상차가 최종 확률에 나타난다.

$$
HTH|0\rangle=\frac{1+e^{i\pi/4}}2|0\rangle+
\frac{1-e^{i\pi/4}}2|1\rangle,\qquad
p(1)=\frac{2-\sqrt2}{4}.
$$

H 직후와 TH 직후에 계산 기저로 측정할 확률은 모두 1/2이다. 그런데 마지막 H가 상대 위상을 진폭의 합과 차로 바꾸면서 출력 확률이 달라진다. 계산 전에 결과를 예상해 보자. 마지막 H를 제거하면 이 방식으로 위상을 읽을 수 없게 되지만, 상태의 위상 자체가 사라지는 것은 아니다.

## 3. 선 번호는 텐서 인자의 위치를 뜻한다

여기서는 |b₀b₁b₂〉의 site0을 최상위 비트로 둔다. 따라서 |101〉의 좌표 인덱스는5이고, X₀=X⊗I⊗I, X₂=I⊗I⊗X이다. 라이브러리가 다른 비트 순서를 사용해도 틀린 것은 아니다. 답을 비교하기 전에 좌표를 같은 관례로 변환해야 한다.

제어0·대상1인 CX는 |10〉을 |11〉로 바꾸고 |01〉은 그대로 둔다. 제어와 대상을 바꾸면 두 결과가 모두 바뀐다. CZ는 |11〉의 부호만 뒤집는다. SWAP은 두 선의 상태를 교환하며, 페르미온 모드 교환의 반대칭 부호를 추가하지 않는다.

|00〉에서 선0에 H를 적용한 뒤 CX₀→₁을 적용하면 (|00〉+|11〉)/√2가 된다. 각 선의 계산 기저 결과는 균등하지만 두 결과는 완벽하게 상관되어 있다. 주변 분포를 곱해서는 이 결합 상태를 복원할 수 없다. 독립적인 확률 막대 두 개를 전체 상태로 해석하지 말고 결합 진폭을 확인하자.

## 4. 전체 위상과 제어된 구현

상태 전체에 e^{iα}를 곱해도 밀도행렬은 변하지 않는다. 한 갈래에만 위상을 곱하면 일반적으로 간섭이 바뀐다. 시뮬레이터에서도 진폭이 정확히 같은 것과 공통 위상까지 허용하면 같은 것을 구별해야 한다.

더 미묘한 점도 있다. U와 e^{iα}U는 계에 같은 채널을 만들지만, controlled-U와 controlled-(e^{iα}U)는 제어 큐비트에서 다른 결과를 만들 수 있다. U=I, α=π일 때 두 번째 연산을 |+〉⊗|0〉에 적용하면 제어 상태가 |−〉로 바뀐다. 따라서 계의 채널만 주어졌다고 그 채널을 일관되게 제어하는 구현까지 정해지는 것은 아니다. 회로 변환이나 새 학습 구조를 비교할 때 필요한 주의점이다.

## 5. 검증과 계산 비용

역회로는 게이트를 역순으로 배열하고 각각의 수반을 적용한다. H, X, Y, Z, CX, CZ, SWAP은 자기 자신이 역연산이지만 S와 T에는 S†, T†가 필요하다. 원래 순서로 게이트를 되돌리는 것은 일반적으로 잘못이다. 복원된 노름뿐 아니라 복소 진폭도 비교하자.

계산 모델은 H와 T를 포함한 계수를 Q(√2,i)에서 정확히 유지하고, 정규화 전 진폭과 제곱 노름을 반환한다. 입력이 정규화되지 않았다면 확률은 제곱 노름으로 나누어야 한다. 영벡터에는 정규화된 상태가 없으므로 거부한다. 정확 연산은 작은 잔차를 임의로0으로 만드는 문제를 피하지만, 상태 크기의 지수적 증가는 없애지 못한다. n큐비트에는 2ⁿ개 진폭이 있다. 작은 실험 성공이 확장성의 증거는 아니다.

독립 검산은 전체 공간에 삽입한 행렬을 먼저 만든다. 서로 다른 계산 경로의 일치, 역회로 복원, 모든 계산 기저 입력에서의 작용을 확인해야 한다. 잘못된 유니터리도 노름은 보존하므로 노름 검증만으로 충분하지 않다.

## 6. 확률·국소 상태·결합 상관관계를 함께 읽는다

정규화하지 않은 영이 아닌 벡터 z에서 시작해도 된다. 다만 제곱 노름 N을 이용해 성분의 의미를 해석해야 한다. 회로에서 나온 진폭이라고 해서 자동으로 정규화된 것은 아니다.

$$
N=z^\dagger z,\qquad p_b=\frac{|z_b|^2}{N},\qquad
\rho=\frac{zz^\dagger}{N},\qquad \rho_A=\operatorname{Tr}_{\bar A}\rho.
$$

z=(1,i)이면 N=2이고 밀도행렬의 비대각 성분은 −i/2와 i/2이다. 계산 기저 확률은1/2,1/2이지만 최대혼합 상태는 아니다. 두 확률이 담지 못한 정보가 비대각 성분에 있다.

단일 큐비트의 축약 상태는 Bloch 좌표로 쓸 수 있다. 이 세 좌표는 각각의 기댓값이며 서로 독립적인 확률 세 개가 아니다.

$$
\rho_i=\frac{I+r_xX+r_yY+r_zZ}{2},\qquad
r_a=\operatorname{Tr}(\rho_i\sigma_a),\qquad
\operatorname{Tr}(\rho_i^2)=\frac{1+\|r\|^2}{2}.
$$

z=(1,i) 예는 r=(0,1,0)이다. Y의 부호를 확인하면 복소켤레 방향의 오류를 찾기 좋다. Bloch 구면 위의 점은 순수 상태다. 구 내부의 축약 상태는 혼합 상태일 수 있으며, 전체 상태가 순수해도 그렇다. Bell 상태에서는 각 Bloch 벡터가0이고 국소 순도는1/2이지만 결합 순도는1이다. 이 국소 벡터들을 독립적인 순수 상태로 취급하면 얽힘을 잃는다.

두 지점의 기댓값은 빠진 결합 정보의 일부를 드러낸다. 연결 상관관계는 개별 평균으로 설명되는 부분을 뺀다.

$$
C_{ab}^{ij}=\langle\sigma_i^a\sigma_j^b\rangle,
\qquad C_{ab}^{ij,\mathrm{conn}}=
C_{ab}^{ij}-\langle\sigma_i^a\rangle\langle\sigma_j^b\rangle.
$$

더하기 Bell 상태에서는 XX=1, YY=−1, ZZ=1이고 개별 평균은 모두0이다. 반면 곱 입력에서도 두 지점 기댓값이0이 아닐 수 있다. |+i〉⊗|+〉의 Y₀X₁=1은 Y₀=1과 X₁=1의 곱으로 완전히 설명되므로 연결 상관관계는0이다. 원래 상관관계가0이 아니라는 사실만으로 얽힘을 인증할 수는 없다.

모든 두 지점 상관관계를 알아도 세 큐비트 상태가 정해지지 않을 수 있다. GHZ 상태와 |000〉·|111〉의 균등한 고전 혼합은 단일·두 큐비트 축약 상태가 같다. 그러나 세 지점 XXX 기댓값은 GHZ에서1, 혼합에서0이다. 결합 상태가 같은지를 판단할 때에는 전체 밀도행렬을 유지해야 한다.

마지막으로 fidelity의 관례를 지정하자. 여기서는 순수 상태 중첩의 제곱을 사용한다.

$$
F_2(z,w)=\frac{|z^\dagger w|^2}{(z^\dagger z)(w^\dagger w)}.
$$

전체 위상만 다른 상태에서는1이고 |0〉과 |+〉 사이에서는1/2이다. 어떤 교재는 이 값의 제곱근을 fidelity라고 부른다. 수치를 비교하기 전에 정의식을 확인하자. 축약 상태와 제곱근 fidelity 관례의 배경은 [IBM density matrices](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/density-matrices), [IBM purifications and fidelity](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/purifications-and-fidelity)를 참고한다.

## 연구 확장: 게이트와 매개변수를 구별하며 회로를 미분한다

학습 가능한 양자 모형에는 올바른 전방 상태뿐 아니라 기댓값의 미분이 필요하다. 입력·관측량·다른 모든 매개변수를 고정하자. 미분할 게이트 하나를 고정된 두 unitary 사이에 놓고 두 지수함수를 미분하면 교환자가 나온다. G와 O가 Hermitian일 때 i 인자가 그 기댓값을 실수로 만든다. 각도는 무차원 radian이다.

$$
U(\theta)=W e^{-i\theta G}V,\quad G=G^\dagger,\quad |\phi\rangle=V|\psi\rangle,\quad A=W^\dagger O W,\quad
 f(\theta)=\langle\phi|e^{i\theta G}Ae^{-i\theta G}|\phi\rangle,\quad
 f'(\theta)=i\langle\phi|e^{i\theta G}[G,A]e^{-i\theta G}|\phi\rangle.
$$

### 서로 다른 생성자 고유값 두 개는 주파수 하나를 준다

생성자에서 cI를 빼면 이 통상적인 비제어 기댓값에서는 전체 위상만 바뀐다. 남은 생성자는 rP이고 P의 제곱은 항등연산자다. 지수함수를 cosine과 sine으로 전개하면 상수와 주파수 2r 하나만 남는다. 이동시킨 두 삼각식을 빼면 정확한 미분을 얻는다. Ry에서는 생성자가 Y/2이므로 r=1/2, 이동량 pi/2, 계수 1/2이다. 작은 간격의 유한 차분이 아니다.

$$
G=cI+rP,\quad P^2=I,\quad r>0:\qquad
 f(\theta)=a+b\cos(2r\theta)+d\sin(2r\theta),\quad
 f'(\theta)=r\left[f\left(\theta+\frac\pi{4r}\right)-f\left(\theta-\frac\pi{4r}\right)\right].
$$

### 순서를 명시한 얽힘 회로를 미분한다

00에서 선0을 theta만큼 회전하고 제어0인 CX를 적용한 뒤 선1을 phi만큼 회전한다. 선1의 Z를 측정하자. 처음 두 게이트가 cos(theta/2)|00>+sin(theta/2)|11>을 만들고 마지막 국소 회전이 Z 기댓값을 아래처럼 바꾼다. 독립 행렬 계산도 같은 시간 순서와 텐서 번호를 사용한다.

$$
|\psi(\theta,\varphi)\rangle=(I\otimes R_y(\varphi))\,\mathrm{CX}_{0\to1}\,(R_y(\theta)\otimes I)|00\rangle,\quad
 R_y(t)=e^{-itY/2},\quad O=I\otimes Z,\quad
 f=\cos\theta\cos\varphi,\quad
 \partial_\theta f=-\sin\theta\cos\varphi,\quad\partial_\varphi f=-\cos\theta\sin\varphi.
$$

### 매개변수 공유와 손실 처리는 문제를 바꾼다

Theta와 phi를 같은 변수 t에 묶으면 두 등장 위치의 편미분을 더해야 한다. 두 위치를 동시에 pi/2씩 옮기면 차이가 0이 되어 실제 도함수를 놓친다. 비선형 고전 손실도 연쇄법칙이 필요하다. f=cos(theta), L=f 제곱만 생각해도 pi/4에서 손실 미분은 −1인데 손실값 둘을 직접 이동시켜 빼면 0이다. 기댓값의 정확한 규칙이 기댓값의 임의 함수로 자동 확장되지는 않는다.

$$
g(t)=f(t,t)=\cos^2t,\quad g'(t)=\partial_\theta f(t,t)+\partial_\varphi f(t,t)=-\sin(2t),\quad
 \frac{g(t+\pi/2)-g(t-\pi/2)}2=0;\qquad
 L=(f-y)^2:\quad\partial_\theta L=2(f-y)\partial_\theta f.
$$

### 정확한 공식에도 많은 측정이 필요할 수 있다

독립 shot으로 Pauli 관측량을 측정하면 결과는 +1 또는 −1이다. 표본 평균의 분산은 (1−f 제곱)/m이다. 두 이동 회로를 독립적으로 추정하면 아래 gradient 분산을 얻는다. 배분 공식은 두 분산이 양수이며 알려져 있을 때 양의 실수 shot 수에 대해 최소화한 결과다. 장비의 횟수는 정수이고 알려지지 않은 분산에는 예비 추정이 필요하다. 영분산 끝점과 반올림은 따로 다룬다. 매개변수와 무관한 잡음은 고정 준비·읽기 map에 흡수할 수 있지만 매개변수 의존 잡음은 이 이상적 규칙을 따르지 않을 수 있다.

$$
\widehat{\partial_\theta f}=(\bar O_+-\bar O_-)/2,\quad O^2=I,\quad
 \operatorname{Var}(\widehat{\partial_\theta f})=\frac{1-f_+^2}{4m_+}+\frac{1-f_-^2}{4m_-},\quad
 m_++m_-=M,\quad \sigma_\pm=\sqrt{1-f_\pm^2},\quad
 m_\pm=\frac{M\sigma_\pm}{\sigma_++\sigma_-},\quad
 \operatorname{Var}_{\min}=\frac{(\sigma_++\sigma_-)^2}{4M}.
$$

### 제어 회전은 생성자 spectrum이 다르다

Controlled Ry는 서로 다른 생성자 고유값이 세 개다. 따라서 가능한 비영 주파수 크기도 1/2과 1 두 가지다. 제어를 plus, 대상을 zero로 준비하고 제어의 X를 읽으면 대상 갈래의 중첩에서 cos(theta/2)가 나온다. 통상 Ry 규칙을 쓰면 필요한 1/2 대신 1/sqrt(2)가 붙는다. 제어된 구현은 미분할 게이트 자체가 다르다.

$$
C R_y(\theta)=e^{-i\theta G_c},\quad G_c=|1\rangle\langle1|\otimes Y/2,\quad
 \operatorname{spec}G_c=\{0,-1/2,1/2\},\quad
 |\psi_0\rangle=|+0\rangle,\quad O=X\otimes I:\qquad
 f_c(\theta)=\cos(\theta/2),\quad f_c'=-\tfrac12\sin(\theta/2),\quad
 \tfrac12[f_c(\theta+\pi/2)-f_c(\theta-\pi/2)]=-\tfrac1{\sqrt2}\sin(\theta/2).
$$

### 실제 주파수에서 규칙을 복원한다

이 제어 게이트가 고정 연산 사이에 한 번 등장하면 기댓값 주파수는 생성자 고유값 차이들이다. 차이 공식이 주파수 1/2과 1 모두를 정확히 미분하도록 두 선형 방정식을 풀자. 부호 있는 가중치로 회로 네 번을 평가하는 규칙이 나온다. 이 주파수 집합에 대한 것이며 모든 제어 게이트나 재사용 매개변수에 대한 보편 공식은 아니다.

$$
D_s f=f(\theta+s)-f(\theta-s),\quad
 f\in\operatorname{span}_{\mathbb R}\{1,\cos(\theta/2),\sin(\theta/2),\cos\theta,\sin\theta\}:\qquad
 f'(\theta)=\tfrac12D_{\pi/2}f+\frac{1-\sqrt2}{4}D_\pi f.
$$

### 유리수 각도로 표를 정확히 확인한다

Tan(phi/2)=1/2를 고정하고 t=tan(theta/2)로 놓자. Sine·cosine·두 편미분과 독립적인 100+100 shot 추정기의 분산이 모두 유리수다. Gradient가 0이라는 사실은 추정기 분산도 0이라는 뜻이 아니다. t=0에서 미분은 사라지지만 분산은 1/200이다. Shot을 늘리면 이상적 미분이 바뀌는 대신 추정 불확실성이 줄어든다.

$$
t=\tan(\theta/2),\quad\tan(\varphi/2)=1/2,\quad
 \cos\theta=\frac{1-t^2}{1+t^2},\quad\sin\theta=\frac{2t}{1+t^2},\quad
 \cos\varphi=3/5,\quad\sin\varphi=4/5,\quad
 f_\pm=\pm\partial_\theta f,\quad m_+=m_-=100,\quad
 \operatorname{Var}(\widehat{\partial_\theta f})=\frac{1-(\partial_\theta f)^2}{200}.
$$

```python
from fractions import Fraction as F
for t in [F(0),F(1,3),F(1,2),F(1)]:
    cos=(1-t*t)/(1+t*t)
    sin=2*t/(1+t*t)
    f=F(3,5)*cos
    dtheta=-F(3,5)*sin
    dphi=-F(4,5)*cos
    variance=(1-dtheta*dtheta)/200
    print(t,f,dtheta,dphi,variance)
```

| t | f | theta 편미분 | phi 편미분 | 분산, M=200 |
| --- | --- | --- | --- | --- |
| 0 | 3/5 | 0 | -4/5 | 1/200 |
| 1/3 | 12/25 | -9/25 | -16/25 | 68/15625 |
| 1/2 | 9/25 | -12/25 | -12/25 | 481/125000 |
| 1 | 0 | -3/5 | 0 | 2/625 |

![Controlled Ry에서 통상 두 이동 미분이 실패하고 네 번 평가 주파수 규칙이 해석적 도함수와 일치하는 모습 및 측정 횟수에 따른 분산 감소.](/assets/learning/quantum-compression-gradient-research-parameter-shift-frequency.ko.svg)

위 패널은 보통 회전이 아니라 controlled Ry 뒤 제어 X를 읽은 것이다. 아래는 고정 phi에서 독립적으로 똑같이 배분한 Pauli shot 추정기의 정확 분산이다. 해석적 기준 예제이며 reader의 기존 정확 H/S/T 회로 제어를 유지한다. 연속 optimizer나 장비 sampler를 추가한 것은 아니다.

#### 연구 확인 1: 공유 변수에는 편미분 둘이 필요하다

Theta=phi=pi/4에서 각 편미분은 −1/2이므로 묶인 변수의 도함수는 −1이다. 두 각도를 동시에 ±pi/2 옮기면 두 회로의 값이 모두 1/2여서 잘못된 영미분을 얻는다.

#### 연구 확인 2: 제어 회전과 보통 회전은 다르다

Theta=pi에서 controlled Ry의 제어 X 기댓값 미분은 −1/2이다. 통상 규칙은 −1/sqrt(2)를 반환한다. 네 번 평가 공식은 가능한 두 비영 주파수에 맞췄기 때문에 −1/2를 반환한다.

#### 연구 확인 3: 영 gradient에도 shot 잡음이 있다

Theta=0, cos(phi)=3/5이면 두 이동 Z 기댓값은 0이다. 각 100 shot의 독립 평균은 각각 분산 1/100을 가지며 그 차이를 2로 나누면 분산 1/200이다. 기댓값이 정확히 0이라고 개별 Pauli 결과가 확정적이지는 않다.

배경은 [Schuld 외, Evaluating analytic gradients on quantum hardware, §II](https://arxiv.org/pdf/1811.11184)의 두 고유값 parameter-shift 규칙이다. 공유 변수 반례·제어 주파수 공식·shot 계산은 직접 전개한 유한 예제이며 학습 확장성이나 barren plateau 진단을 주장하지 않는다.

## 7. 해설이 있는 연습

**예측:** Z가 |+〉의 계산 기저 확률을 바꾸는가? 아니다. 상대 부호를 바꾸며, X기저 측정은 그 차이를 구별한다. 계산 기저 측정 전에 H를 넣으면 이 정보를 읽을 수 있다.

**계산:** |01〉에 제어1·대상0인 CX를 적용하자. 제어가1이므로 |11〉이 된다. 제어0·대상1이었다면 |01〉 그대로다. 비트 순서와 제어의 의미를 함께 확인하는 예다.

**오류 찾기:** 노름은 맞지만 Bell 상관관계가 틀리면 정규화부터 고쳐야 하는가? 아니다. 초기 텐서 순서, 제어·대상 번호, 게이트 시간 순서를 확인하자. 잘못된 유니터리도 모든 노름을 보존하면서 다른 상태를 계산할 수 있다.

## 8. 연결과 출처

매개변수 회로에는 미분 가능한 게이트와 목적함수가 추가되고, 잡음을 넣으면 상태 벡터 대신 밀도연산자가 필요하다. 다음 단원에서는 어떤 잡음 갱신이 물리적으로 가능한지 묻는다. 국소 통계만으로 회로를 설명하기 전에 유니터리·얽힘 단원을 복습하자. 표현의 배경은 [IBM Quantum Learning](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels)을 참고했다. HTH와 선 순서 계산은 이 교재의 독립 예제다.
