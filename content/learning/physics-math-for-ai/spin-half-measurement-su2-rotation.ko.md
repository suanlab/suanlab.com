## 1. 스핀은 내부 자유도다

스핀은 내부 양자 상태가 회전 아래에서 어떻게 변하고 어떤 각운동량 성분을 측정할 수 있는지 기술한다. 작은 공이 자기 표면을 따라 기계적으로 자전한다는 모형이 아니다. 스핀1/2의 상태 공간은 C²다. 비영 원시 벡터 z에서 정규화 대표 ψ=z/||z||를 만들고 첫째·둘째 표준 기저를 e₀·e₁로 쓴다.

두 성분은 진폭이지 공간의 두 지점이 아니다. 물리적으로 구현한 모든 qubit의 공간 회전이 스핀1/2 입자와 정확히 같다고 가정하지도 않는다. 여기서는 스핀1/2 표현을 명시적으로 선택한다. 준비·능동 회전·마지막 측정은 서로 다른 동작이다. 회전 표본을 여러 개 본다고 하나의 실험 중간에 측정을 반복해서 삽입하는 것은 아니다.

## 2. Pauli 행렬이 스핀의 척도를 정한다

오른손 방향의 세 축을 선택한다. 각 스핀 성분은 Pauli 행렬에 비례하는 Hermitian 행렬이며 단위는 ℏ와 같은 작용이다.

$$
\sigma_x=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
\sigma_y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
\sigma_z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad S_i=\frac{\hbar}{2}\sigma_i.
$$

각 성분의 결과는 +ℏ/2와 −ℏ/2다.1/2은 중요한 계수다. σ 자체를 물리적 스핀으로 사용하면 평균은 두 배, 분산은 네 배가 된다. σy의 허수 성분도 생략하면 안 된다. adjoint는 전치뿐 아니라 복소 켤레를 포함한다.

Sz 고유상태는 z 결과가 정해져 있어도 x·y 측정에서는 각각 불확실할 수 있다. 같은 준비를 별도로 만든 실험들의 결과에 관한 말이다. 하나의 고전적 물체가 세 성분의 값을 동시에 가지고 있다는 뜻으로 해석하지 않는다.

## 3. 측정 방향은 두 사영을 선택한다

n을 실수 단위 벡터, r을 성분 ψ†σiψ로 이루어진 Bloch 벡터라고 하자. 순수 C² 상태에서는 r의 길이가1이다. n 방향 측정의 사영·확률·모멘트는 다음과 같다.

$$
S_{\mathbf n}=\frac{\hbar}{2}\mathbf n\cdot\boldsymbol\sigma,\quad
P_\pm=\frac{I\pm\mathbf n\cdot\boldsymbol\sigma}{2},\quad
p_\pm=\frac{1\pm\mathbf n\cdot\mathbf r}{2},\qquad
\langle S_{\mathbf n}\rangle=\frac{\hbar}{2}\mathbf n\cdot\mathbf r,\quad
\operatorname{Var}(S_{\mathbf n})=\frac{\hbar^2}{4}\left[1-(\mathbf n\cdot\mathbf r)^2\right].
$$

Sₙ=(ℏ/2)P+−(ℏ/2)P−라는 스펙트럼 식에서 평균은 두 확률의 부호 있는 차이가 된다. 두 결과를 제곱하면 모두 ℏ²/4이므로2차 모멘트는 ℏ²/4다. 여기서 평균의 제곱을 빼면 분산을 얻는다. 기대값0은 한 번의 측정 결과가0이라는 뜻이 아니다.

e₀에서는 r이 +z를 가리킨다. z 측정은 +ℏ/2를 확실히 반환하지만 x·y는 같은 두 확률을 갖는다. (e₀+e₁)/√2는 +x를 가리킨다. 이 상태와 (e₀−e₁)/√2는 z histogram이 같지만 x 결과는 반대다. histogram 하나로 ray를 정할 수 없다.

## 4. 성분 제곱과 전체 스핀 제곱은 다르다

Pauli 행렬을 곱하면 각운동량 교환관계와 전체 스핀 제곱이 나온다. 나머지 두 교환관계는 축을 순환시키면 얻는다.

$$
[S_x,S_y]=i\hbar S_z,\qquad
S_x^2=S_y^2=S_z^2=\frac{\hbar^2}{4}I,\qquad
\mathbf S^2=S_x^2+S_y^2+S_z^2=\frac{3\hbar^2}{4}I.
$$

이 표현에서 전체 스핀 제곱의 고유값은3ℏ²/4 하나다. 그 제곱근은 √3ℏ/2지만, 성분 측정 결과의 크기는 ℏ/2다. 서로 다른 관측량이다. 세 성분에 동시에 고정된 값을 부여한 고전적 벡터로 비가환 연산자들을 대신할 수 없다.

e₀의 평균 벡터는(0,0,ℏ/2)이므로 길이 제곱은 ℏ²/4다. 전체2차 모멘트와의 차이 ℏ²/2은 x·y 분산에서 온다. 일반적인 순수 스핀1/2 상태도 평균 벡터 길이 제곱은 ℏ²/4이며 Casimir 값3ℏ²/4와 구별해야 한다. 전체 연산자의 제곱과 평균한 벡터를 제곱하는 순서를 바꾸면 다른 질문이 된다.

## 5. 회전에는 반각이 들어간다

능동 회전은 실험실의 측정 축을 고정한 채 상태를 바꾼다. 단위 축 a 주위의 각도 φ 회전은 그 방향의 스핀 성분이 생성한다. (a·σ)²=I이므로 exponential의 짝수·홀수 거듭제곱을 나눠 계산할 수 있다.

$$
U(\mathbf a,\phi)=\exp\left(-\frac{i\phi}{\hbar}\mathbf a\cdot\mathbf S\right)
=\cos\frac{\phi}{2}\,I-i\sin\frac{\phi}{2}\,\mathbf a\cdot\boldsymbol\sigma,
\qquad \phi=2\pi\nu.
$$

반각은 S=ℏσ/2에서 나온다. 애니메이션의 속도를 임의로 절반으로 정했다는 뜻이 아니다. 기하학적인 회전 행렬에서는 ℏ가 소거된다. 같은 각도에서 ℏ의 수치를 바꾸면 스핀 단위의 척도는 바뀌지만 상태 변환 자체는 유지된다.

ν는 공간 회전의 한 바퀴를 단위로 센 값이다. 경과한 물리 시간이 아니다. Hamiltonian과 지속 시간을 지정하면 관련된 회전을 동역학으로 만들 수 있지만 추가적인 가정이 필요하다. 여기서는 slider를 회전 각도의 선택으로 읽는다. Bloch 구의 카메라를 드래그하는 것은 보는 방향만 바꾼다.

## 6. 양의 X 회전을 손으로 계산하자

e₀에서 시작하여 +x 주위로 능동 회전한다. 앞의 행렬을 곱하면 두 진폭과 Bloch 벡터를 바로 얻는다.

$$
\psi(\phi)=\begin{pmatrix}\cos(\phi/2)\\-i\sin(\phi/2)\end{pmatrix},\qquad
\mathbf r(\phi)=(0,-\sin\phi,\cos\phi),\qquad
p_Z(+)=\cos^2(\phi/2),\quad p_Y(+)=\frac{1-\sin\phi}{2}.
$$

φ=π/2에서는 상태가(1,−i)/√2이며 Bloch 구의 −y를 가리킨다. 각도가 양수라고 y 성분도 양수인 것은 아니다. 오른손 좌표계에서는 x×z=−y다. φ=π에서는 상태가 −ie₁이 되고 z의 음의 결과가 확률1이다.

| 공간 각도 | 상태 대표 | Z 양의 결과 확률 | Sz 평균 |
| --- | --- | --- | --- |
| 0 | e₀ | 1 | ℏ/2 |
| π/2 | (e₀−ie₁)/√2 | 1/2 | 0 |
| π | −ie₁ | 0 | −ℏ/2 |
| 2π | −e₀ | 1 | ℏ/2 |
| 4π | e₀ | 1 | ℏ/2 |

진폭 화살표·Bloch 좌표·확률 막대를 함께 비교하면 이해하기 쉽다. 복소 대표는 이 histogram 하나보다 많은 정보를 담는다. 반대로 Bloch 벡터는 대표의 공통 위상을 지운다. 표의 각 행을 볼 때 벡터의 같은 부호와 같은 측정 법칙을 따로 판단해보자.

## 7. 공간 두 바퀴에서 spinor가 복원된다

반각 식에 대입하면 공간 한 바퀴 뒤에는 부호가 바뀌고 두 바퀴 뒤에는 대표가 복원된다. 정규화 입력이 무엇이든 첫 부호 변화에서 local spin 확률은 유지된다.

$$
U(\mathbf a,2\pi)=-I,\qquad U(\mathbf a,4\pi)=I,\qquad
|\langle\psi,-\psi\rangle|^2=1,\qquad
\|(-\psi)-\psi\|^2=4.
$$

마지막 두 양은 다른 질문에 답한다. fidelity는 공통 위상을 무시하고 ray를 비교한다. 대표 벡터 차이 제곱은 그 위상 규약을 유지한다. 그림이나 수치 비교에서 한 지표를 다른 지표로 몰래 바꾸면 안 된다.

SU(2)는 spinor에 직접 작용한다. U와 −U는 SO(3)에서 같은 Bloch 벡터 회전을 만든다. 이것이 double cover의 구별이다.2π 뒤 Bloch 화살표가 돌아왔다고 spinor 대표도 돌아온 것은 아니다. 행렬이 norm을 보존한다는 사실만으로 의도한 회전을 표현한다고 판단할 수도 없다.

## 8. 경로 사이에서는 공통 위상이 상대 위상이 될 수 있다

하나의 고립된 스핀 전체에 곱해진 부호는 그 스핀만 측정해서 볼 수 없다. 그렇다고 모든 coherent 제어 실험에서 무의미한 것은 아니다. 직교하는 두 경로 a·b에 같은 스핀을 준비하고 경로 b의 스핀만2π 회전한다고 생각하자.

$$
\frac{|a\rangle+|b\rangle}{\sqrt2}\otimes\psi
\longmapsto
\frac{|a\rangle-|b\rangle}{\sqrt2}\otimes\psi.
$$

마지막 스핀 인자는 여전히 ψ지만 두 경로 중첩은 서로 직교한다. 경로를 합/차 기저에서 다시 합치면 둘을 구별할 수 있다. 위상이 이제 전체 준비의 공통 인자가 아니라 경로 사이의 상대 위상이 되었기 때문이다.

이는 단일 스핀 예제를 해석적으로 확장한 사고 실험이다. 현재 수치 가족에 경로 register·controlled gate·간섭 장치를 추가로 제공한다는 뜻이 아니다. 복합계와 얽힘은 뒤에서 다룬다. 여기서는 local 확률에 대한 결론에 적용 범위를 밝혀야 한다는 점만 확인한다.

## 9. Spinor와3차원 회전을 대조한다

같은 능동 U는 Bloch 벡터를 보통의 오른손 Rodrigues 회전식으로 보낸다. 두 계산은 서로 다른 수학적 대상을 사용하므로 독립적으로 비교할 수 있다.

$$
\mathbf r'=\mathbf r\cos\phi+(\mathbf a\times\mathbf r)\sin\phi
+\mathbf a(\mathbf a\cdot\mathbf r)(1-\cos\phi),\qquad
r'_i=(U\psi)^\dagger\sigma_i(U\psi).
$$

+x 준비를 +z 주위로 회전하면 r′=(cosφ,sinφ,0)이며 π/2에서 +y를 가리킨다. 앞에서 +z 준비를 +x 주위로 회전했을 때의 부호와 비교해보자. 축·초기 상태·능동 회전 규약을 함께 명시해야 화살표가 단순히 시계 방향으로 돈다는 모호한 설명을 피할 수 있다.

상태 방향 r과 측정 방향 n을 함께 회전하면 내적은 유지된다. 공변성에 관한 말이다. n을 고정하고 상태만 돌리면 다른 실험이므로 확률이 변할 수 있다. 수동 좌표 변환은 상태와 관측량의 좌표를 함께 바꾸는 것이며 능동 상태 회전과 섞어 쓰지 않는다.

## 10. 정확한 turn과 구간의 의미

준비는 원시 상태·두 방향·부호 있는 유리수 turn·양의 ℏ·표본 수를 위한 제한된 정수17개를 사용한다. 원시 벡터들은 영이 아니어야 한다. 방향마다 P=vv†/(v†v)를 만들므로 Bloch 단위 방향과 초기 스핀 대수는 정확 유리수다. 스핀 평균은 작용 단위, 분산은 작용 제곱 단위다.

회전에서는 turn을 정확히 modulo2로 줄인다. 반 바퀴 기준점은 정확한 정수 조건으로만 인식한다. 일반 각도는 증명된 π 구간과 방향성 삼각함수로 계산하고 작업 연산은768bit, 공개 관측은 바깥 방향으로 반올림한512bit다.1turn 근처의 각도를 분류할 때1turn으로 반올림하지 않는다. 출력 norm은 검사하며 다시 정규화해서 보정하지 않는다.

돌아오는 시점 근처에서 fidelity가 화면에1로 보여도 정확한 재귀를 증명한 것은 아니다. 잔차 구간이0을 포함하는 사실과 구조적으로 정확히0인 사실도 다르다. 미해결 관측은 임의의 그림 좌표 대신 null을 유지할 수 있다. 독립 참조는 고정밀 일반 행렬 exponential과 기호 고유공간을 사용하며 생산 turn 축소나 π 함수를 재사용하지 않는다.

## 연구 확장: 회전하는 상태와 정보를 주는 판독은 다르다

### 회전각을 얼마나 잘 추론할 수 있는가?

Bloch 회전이 보여도 선택한 검출기에 유용한 신호가 생긴다는 보장은 없다. 매 시행에서 같은 알려진 초기 상태를 준비하고 능동 회전각 φ를 미지 매개변수로 보자. 본문의 부호 규약으로 상태 가족과 Bloch 미분을 얻는다.

$$
\psi_\varphi=e^{-i\varphi S_{\mathbf a}/\hbar}\psi_0,\qquad
 \partial_\varphi\psi_\varphi=-\frac i2(\mathbf a\cdot\boldsymbol\sigma)\psi_\varphi,
 \qquad \partial_\varphi\mathbf r_\varphi=\mathbf a\times\mathbf r_\varphi.
$$

전체 위상만 바꾸는 미분 성분을 제거하자. 이 순수 상태 가족의 quantum Fisher information은 그렇게 얻은 ray metric의 네 배다.

$$
g_\varphi=\langle\partial_\varphi\psi|\partial_\varphi\psi\rangle
 -|\langle\psi|\partial_\varphi\psi\rangle|^2
 =\frac{\operatorname{Var}(S_{\mathbf a})}{\hbar^2},\qquad
 \mathcal F_Q=4g_\varphi=1-(\mathbf a\cdot\mathbf r_0)^2.
$$

국소 각도 민감도를 정하는 것은 spin 평균이나 Casimir가 아니라 생성자의 분산이다. 초기 Bloch 벡터와 나란한 축의 회전은 대표 위상만 바꾸므로 F_Q=0이고, 수직 축에서는 F_Q=1이다. 단일 매개변수·순수 상태의 설명이며 임의의 혼합 모델 공식이 아니다.

정의는 [Paris §4.1의 식26–27](https://arxiv.org/pdf/0804.2981)과 일치한다. 여기서는 최적 측정을 이미 구현했다고 가정하지 않고 같은 유한 Bloch 기하로 구체적인 검출기 경계를 유도한다.

### 검출기는 사용 가능한 정보를 잃을 수 있다

고정 단위 방향 n의 spin을 측정하자. 부호 있는 평균을 m이라 쓰면 Bernoulli 확률과 미분은 다음과 같다.

$$
m=\mathbf n\cdot\mathbf r_\varphi,\quad p_\pm=\tfrac12(1\pm m),\quad
 \partial_\varphi p_\pm=\pm\tfrac12m',\quad m'=\mathbf n\cdot(\mathbf a\times\mathbf r_\varphi).
$$

두 확률이 양수일 때 classical Fisher information은 score의 제곱 기댓값이다. score는 로그 확률의 미분이다.

$$
\mathcal I_{\mathbf n}(\varphi)=\sum_{s=\pm}\frac{(\partial_\varphi p_s)^2}{p_s}
 =\frac{(m')^2}{1-m^2}\qquad(|m|<1).
$$

n을 r에 나란한 성분과 수직인 성분으로 나누자. a×r과의 내적에는 수직 성분만 기여한다. 그 길이는 √(1−m²)이고 |a×r|=√(1−(a·r)²)이므로 Cauchy–Schwarz로 다음을 얻는다.

$$
(m')^2\le(1-m^2)\bigl[1-(\mathbf a\cdot\mathbf r_\varphi)^2\bigr],
 \qquad 0\le\mathcal I_{\mathbf n}\le\mathcal F_Q\le1.
$$

ray 속도가 0이 아닌 알려진 동작각 φ₀에서 n을 접선 a×r_φ₀ 방향으로 택하면 m=0이며 국소적으로 I_n=F_Q를 달성한다. 동작점을 이용하는 선택이지 이 실습이 전역적으로 매개변수와 무관한 최적 전략을 제공했다는 뜻은 아니다.

보정된 같은 준비와 검출기로 N번 독립 반복하면 정보는 더해진다. 미분 가능성과 양의 support 가정 아래 국소 불편 추정량은 φ₀에서 평균이 φ₀이며 평균의 미분이1이다.

$$
\mathcal I_N=N\mathcal I_{\mathbf n},\qquad
 \operatorname{Var}_{\varphi_0}(\widehat\varphi)\ge\frac1{N\mathcal I_{\mathbf n}(\varphi_0)}
 \ge\frac1{N\mathcal F_Q}\quad(\mathcal I_{\mathbf n}(\varphi_0)>0).
$$

첫 분산 경계는 추정량 평균을 미분해 얻는다. N시행 score와의 공분산은1이고 score 분산은 N I_n이므로 Cauchy–Schwarz를 적용한다. 하한이지 유한 표본 추정량이 달성한다는 약속은 아니다. 주기성·nuisance 매개변수·보정 오차·전역 bias는 별도 분석이 필요하다.

### 영 확률 분기가 없어도 검출기는 보지 못할 수 있다

+x를 준비해 +z로 회전시키고 +z 쪽으로 기울어진 검출기를 사용하자.

$$
\mathbf a=(0,0,1),\quad \mathbf r_\varphi=(\cos\varphi,\sin\varphi,0),\quad
 \mathbf n=(0,3/5,4/5),\quad
 p_+=\tfrac12(1+\tfrac35\sin\varphi),\quad
 \mathcal I_{\mathbf n}=\frac{9\cos^2\varphi}{25-9\sin^2\varphi},\quad\mathcal F_Q=1.
$$

확률은1/5부터4/5 사이여서 경계 정규화가 필요 없다. φ=π/2에서는 검출기 일차 미분과 Fisher 정보가0이지만 상태의 F_Q는 여전히1이다. 검출기는 이차 변화에는 반응할 수 있다. 일차 정보가0이라는 말이 주변 확률 모델 전체가 상수라는 뜻은 아니다.

유리수 반각으로 정확하게 재현하자.

$$
t=\tan(\varphi/2),\quad c=\frac{1-t^2}{1+t^2},\quad s=\frac{2t}{1+t^2},\quad
 p_+=\tfrac12(1+3s/5),\quad \mathcal I_{\mathbf n}=\frac{9c^2}{25-9s^2}\quad(-\pi<\varphi<\pi).
$$

```python
from fractions import Fraction as Q
for t in [Q(0), Q(1, 2), Q(1), Q(2)]:
    cosine, sine = (1-t*t)/(1+t*t), 2*t/(1+t*t)
    probability = (1+Q(3, 5)*sine)/2
    information = 9*cosine*cosine/(25-9*sine*sine)
    assert Q(1, 5) <= probability <= Q(4, 5)
    print(t, probability, information, Q(1), 1-information)
```

| t | p_plus | classical FI | quantum FI | FI loss |
|---|---|---|---|---|
| 0 | 1/2 | 9/25 | 1 | 16/25 |
| 1/2 | 37/50 | 81/481 | 1 | 400/481 |
| 1 | 4/5 | 0 | 1 | 1 |
| 2 | 37/50 | 81/481 | 1 | 400/481 |

![회전각에 따른 기울어진 검출기의 확률과 classical 정보를 비교하며, quantum 정보는 1인데 검출기가 일차 변화를 보지 못하는 동작점을 표시한다.](/assets/learning/quantum-spin-research-readout-information.ko.svg)

t=1/2와 t=2의 두 행은 검출기 법칙은 같지만 Bloch 벡터는 다르다. 두 점에서 국소 정보가 양수여도 전역 식별이 보장되지 않는다. 다른 적절한 축을 측정하거나 정당화된 사전 범위로 제한하는 것은 추가 실험 선택이다. 현재 회전 reader에는 추정기·시행 표본기·Fisher 정보 제어가 없으며 독립 해석·Python 예제다.

#### 연구 확인 1: 둔감한 축 선택하기

+x를 +z로 회전시키면서 z를 측정하면 두 확률은 항상1/2이고 미분은0이다. 모든 각도에서 I_z=0이며 quantum 정보는1이다. 상태의 움직임과 정보를 주는 판독은 별개의 요건이다.

#### 연구 확인 2: 경계를 pseudocount로 대체하지 않기

동작점에서 n=r이면 한 분기 확률이0이고 정규 식은0/0이 된다. 정보를0이라고 선언하거나 임의의 epsilon으로 나누면 안 된다. 극한값에는 주변 모델 지정과 별도 정칙성 분석이 필요하며 앞의 정규 Cramér–Rao 논증을 그대로 경계에 대입할 수 없다.

#### 연구 확인 3: 국소 미분과 전역 역함수 구별하기

기울어진 검출기에서 φ와 π−φ는 sinφ가 같아 확률도 같다. 그 측정만 아무리 반복해도 두 각도를 구별할 수 없다. 양의 Fisher 정보는 선택한 점 주변의 민감도이지 전역적으로 유일한 역함수가 아니다.

## 11. 해설과 함께 풀어보기

**1. +z 준비의 x·z 측정 법칙은?**

**해설.** z는 +ℏ/2가 확률1이고 x는 ±ℏ/2가 각각1/2다. 하나의 고전적 물체에 두 성분 값을 동시에 지정하는 해석이 아니다.

**2. 한 성분의2차 모멘트가3ℏ²/4가 아니라 ℏ²/4인 이유는?**

**해설.** 성분의 두 결과를 제곱하면 모두 ℏ²/4다. 더 큰 값은 서로 다른 세 제곱 연산자를 합한 Casimir 관측량 S²의 값이다.

**3. 양의 π/2 X 회전은 e₀를 어디로 보낼까?**

**해설.** (1,−i)/√2로 보내고 Bloch 벡터는(0,−1,0)이다. 음의 y 부호는 선택한 능동·오른손 규약에서 나온다.

**4. 2π 회전 뒤의 −ψ를 local spin histogram으로 감지할 수 있을까?**

**해설.** 없다. 모든 local Born 확률과 ray fidelity는 같다. 대표 차이 제곱은4이며, 확장한 coherent 경로 실험에서는 경로별 부호가 상대 위상이 될 수 있다.

**5. 같은 기하 각도에서 ℏ 수치를 두 배로 하면?**

**해설.** 스핀 평균은 두 배, 분산은 네 배다. exponential에서 생성자 S와 분모의 ℏ가 소거되므로 회전 행렬과 진폭은 유지된다.

**6. 999999/1000000turn을 정확히1로 분류해도 될까?**

**해설.** 안 된다. 서로 다른 유리수 각도다. 반올림 뒤의 확률이나 fidelity가 시각적으로 같아 보여도 정확 분류와 구간 근거에서는 그 차이를 유지해야 한다.

## 12. AI 연구와 남은 범위

이 단원은 복소 내적·Hermitian 생성자·군 작용·위상에 민감한 비교를 연결한다. AI의 회전을 다루는 표현에서도 특징이 일반 벡터·tensor·spinor 중 무엇처럼 변환되는지 명시해야 한다. 특징 norm을 보존한다는 사실만으로 변환 법칙이나 equivariance가 증명되지는 않는다.3차원 그림은 표현을 검사하는 데 도움을 주지만 전체 상태 공간을 정의하지는 않는다.

다음 단원에서는 정수·반정수 예제를 포함해 일반 유한 스핀의 ladder 구조와 Casimir를 유도한다. 궤도 파동함수·연산자 정의역·일반 표현 분류·혼합 상태·복합계 동역학은 별도 주제로 남는다. 교재는 **도서 → Interactive Book**에서 연다.

회전·spinor의 배경 규약은 [Preskill §2.2.1](https://www.preskill.caltech.edu/ph219/chap2_15.pdf), 각도 구간의 급수는 [NIST DLMF §4.24](https://dlmf.nist.gov/4.24)에서 확인했다. 선택한 준비·회전 부호의 손계산·구간 계약·해설 문제는 이 교재를 위해 전개했다.
