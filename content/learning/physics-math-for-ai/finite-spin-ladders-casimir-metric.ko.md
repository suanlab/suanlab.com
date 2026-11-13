## 1. 스핀 표지는 연산자의 표현을 정한다

앞에서는2차원 스핀1/2을 다뤘다. 일반적으로 하나의 유한 기약 각운동량 표현은 j라는 표지와 자기 양자수 m으로 표시하는 기저를 갖는다. j는 Casimir 값과 차원을 정한다. m은 선택한 한 성분, 보통 Jz를 정한다. x·y의 값을 동시에 지정하지는 않는다.

먼저 일반 구조를 유도하고 유한 수치 예제를 살펴본다. 실험은 j=0,1/2,…,4를 지원한다. 이 경계는 행렬 크기의 제한이며 수학 이론의 제한이 아니다. j=0의 공간도 차원0이 아니라1이다. 더 높은 정수 j 예제가 올바른 표현이라고 해서 모든 표지에 해당하는 기본 입자가 자연에서 발견되었다는 뜻은 아니다. 궤도 각운동량과 내부 스핀은 대수를 공유하지만 물리적 구현은 다르다.

## 2. 교환관계에서 ladder를 정의한다

유한 내적 공간에서 Hermitian 생성자 Jx·Jy·Jz를 가정한다. 각운동량 대수와 raising/lowering이라는 선형 결합은 다음과 같다. 첫 교환관계에는 축을 순환시킨 두 식도 포함한다.

$$
[J_x,J_y]=i\hbar J_z,\qquad
J_\pm=J_x\pm iJ_y,\qquad
[J_z,J_\pm]=\pm\hbar J_\pm,\qquad
[J_+,J_-]=2\hbar J_z,\qquad J_+^\dagger=J_-.
$$

adjoint 관계에는 공간의 내적이 들어간다. 임의의 좌표에서 항상 보통의 conjugate transpose처럼 보일 필요는 없다. 모든 계수 기저가 정규직교라고 가정하지 않고 이 차이를 명시적으로 확인할 것이다.

Jzψ=ℏmψ라면 Jz를 J± 너머로 옮겨서, 영이 아닌 image의 성분값이 ℏ(m±1)임을 알 수 있다. ladder는 고유값 표지의 한 칸 이동을 말한다. 이것만으로 unitary 시간 진화나 측정 절차가 지정되지는 않는다.

## 3. 양성이 ladder 계수를 정한다

J²는 세 생성자와 가환하므로 Casimir가 하나의 scalar인 기약 sector 안에서 계산하자. 함께 대각화할 수 있는 관측량으로 ladder image의 norm 제곱을 표현할 수 있다.

$$
\mathbf J^2=J_z^2+\frac{J_+J_-+J_-J_+}{2},\qquad
\|J_\pm|j,m\rangle\|^2
=\hbar^2\left[j(j+1)-m(m\pm1)\right].
$$

둘째 식은 사다리의 종료로 정해질 Casimir 표지를 미리 쓴 것이다. 유도할 때는 먼저 Casimir 값을 c라고 두면 된다. 두 norm 제곱은 c−ℏ²m(m+1), c−ℏ²m(m−1)이고 모두 비음이어야 한다. 유한 사다리는 모든 m 너머로 계속될 수 없으므로 위끝은 J+가, 아래끝은 J−가 없앤다.

위끝 표지를 j로 쓰면 c=ℏ²j(j+1)이다. 아래끝은 m(m−1)=j(j+1)을 풀어 위끝보다 낮은 해 m=−j를 택한다. 두 끝 사이가1씩 이동하므로2j는 비음 정수다. 유한 차원·Hermiticity·기약성이 논증의 가정이다. 무한 차원의 미분 연산자 정의역까지 증명한 논증으로 읽지 않는다.

## 4. 일반 유한 사다리를 읽어보자

기저의 위상을 일관되게 고르면 정규화된 행렬 원소는 양의 제곱근 계수로 쓸 수 있다.

$$
J_\pm|j,m\rangle=\hbar\sqrt{(j\mp m)(j\pm m+1)}\,|j,m\pm1\rangle,\qquad
m=-j,-j+1,\ldots,j,\qquad d=2j+1.
$$

끝점에서는 계수가0이 되어 image 전체가 영벡터다. 사다리 밖 표지에 정규화된 상태가 있는 것은 아니다. 일반 입력은 여러 m의 중첩일 수 있으므로 J+를 적용해도 하나의 기저 상태만 나오는 것은 아니다.

j=3/2이면 Casimir는15ℏ²/4이고 세 연결의 크기는 √3ℏ·2ℏ·√3ℏ다. 표는 정규화된 기저 상태 사이의 연산자 계수를 표시하며, 0은 annihilation을 뜻한다.

| m | Raising 계수 | Lowering 계수 |
| --- | --- | --- |
| 3/2 | 0 | √3ℏ |
| 1/2 | √3ℏ | 2ℏ |
| −1/2 | 2ℏ | √3ℏ |
| −3/2 | √3ℏ | 0 |

사다리 그림에서는 선에 연산자 계수를, 꼭짓점에 m과 Jz=ℏm을 함께 표시해야 한다. 선의 크기는 전이 확률이 아니다. 반복 적용은 결국 모든 벡터를 없앤다. 비자명한 기약 ladder 연산자의 nilpotence 차수는 d이며 특정 입력은 그보다 일찍 영이 될 수 있다.

## 5. 정확한 다항식 좌표에는 metric이 필요하다

제곱근을 연산자 계수 대신 기저의 정규화로 옮길 수 있다. n=2j로 두고 두 보조 변수의 homogeneous polynomial을 사용하자. 이 변수들은 입자의 위치가 아니다. 선택한 내적에서 다항식 기저의 metric은 대각행렬이다.

$$
b_k=x^{n-k}y^k,\quad C_k=\binom nk,\quad G_{kk}=\frac1{C_k},\quad
e_k=\sqrt{C_k}\,b_k,\qquad
N=z^\dagger Gz,\quad u_k=\frac{z_k}{\sqrt{C_k}},\quad u^\dagger u=N.
$$

b 기저는 이 metric에서 서로 직교하지만 일반적으로 길이가1은 아니다. e 기저는 정규직교다. 원시 계수 z와 u는 같은 추상 벡터를 두 기저에서 기술한다. 정규화된 물리 진폭은 u/√N이다.

z 성분을 제곱하고 Euclidean 합으로 나누면 계수 histogram을 얻는다. 일반적으로 물리적인 Jz 법칙은 아니다. 계수가 같아도 물리 확률은 다를 수 있다. 좌표 배열과 그 벡터의 기하를 함께 지정해야 하는 구체적인 예다. 이 기저를 쓸 때의 편리한 유리수 계산과 물리적인 확률의 정의를 교환하면 안 된다.

## 6. 물리적인 adjoint와 행렬을 복원한다

다항식을 미분하면 유리수 ladder 성분과 대각인 Jz를 직접 얻는다.

$$
J_+=\hbar x\partial_y,\quad J_-=\hbar y\partial_x,\quad
J_z=\frac\hbar2(x\partial_x-y\partial_y),\qquad
J_+b_k=\hbar k b_{k-1},\quad J_-b_k=\hbar(n-k)b_{k+1},\quad
J_zb_k=\hbar(n/2-k)b_k.
$$

이 작용은 homogeneous 차수 n을 보존하므로 선택한 유한 표현 안에 머문다. 여기의 계수 k가 정규직교 기저의 제곱근 계수와 다른 것은 b 벡터들의 길이가 다르기 때문이다. 물리 법칙을 바꾼 것이 아니다.

B가 다항식 계수를 정규직교 좌표로 보낸다고 하자. 연산자와 adjoint는 그 metric에 맞추어 변환한다.

$$
B=\operatorname{diag}(C_k^{-1/2}),\quad G=B^\dagger B,\qquad
A_{\mathrm{phys}}=B A_b B^{-1},\quad
(A_{\mathrm{phys}})_{ik}=(A_b)_{ik}\sqrt{C_k/C_i},\qquad
A^{\dagger_G}=G^{-1}A_b^\dagger G.
$$

따라서 Jx·Jy·Jz는 가중 좌표에서 Hermitian이고 정규직교 좌표에서는 보통의 Hermitian 행렬이 된다. J+의 가중 adjoint는 J−다. 다항식 행렬에 단순 conjugate transpose를 시험해서 일치하지 않는다고 물리적인 Hermiticity가 실패한 것은 아니다. 행렬을 읽을 때 metric과 기저 변환을 함께 가져가야 한다.

## 7. 스핀1에서 다른 무게를 손으로 계산하자

j=1, ℏ=1, 다항식 계수 z=(1,1,1)을 선택한다. binomial 계수는(1,2,1)이므로 N=1+1/2+1=5/2다. 물리적인 정규화 벡터와 Jx 행렬은 다음과 같다.

$$
\psi=\begin{pmatrix}\sqrt{2/5}\\1/\sqrt5\\\sqrt{2/5}\end{pmatrix},\qquad
J_x=\frac1{\sqrt2}\begin{pmatrix}0&1&0\\1&0&1\\0&1&0\end{pmatrix},\qquad
(p_{m=1},p_{m=0},p_{m=-1})=(2/5,1/5,2/5).
$$

다항식 좌표의 Jx 세 행은(0,1/2,0), (1,0,1), (0,1/2,0)이다. z에 작용시키면(1/2,2,1/2)다. z와의 가중 내적은2이므로 평균은2/(5/2)=4/5다. image의 가중 norm 제곱은5/2이므로2차 모멘트는1, 분산은1−16/25=9/25다.

| 성분 | 평균 | 2차 모멘트 | 분산 |
| --- | --- | --- | --- |
| Jx | 4/5 | 1 | 9/25 |
| Jy | 0 | 1/5 | 1/5 |
| Jz | 0 | 4/5 | 4/5 |

단순 계수 histogram은(1/3,1/3,1/3)이겠지만 물리 법칙에는 inverse binomial 무게가 포함된다. 세2차 모멘트의 합은2로 j(j+1)과 같고, 분산의 합은34/25다. 이 계산은 좌표의 기하·정규화·모멘트 해석을 한꺼번에 확인한다. 측정 평균 하나만 맞는지 확인하는 것보다 여러 관계를 함께 검토하는 이유다.

## 8. 연산자 image의 norm은 확률이 아니다

같은 상태에 raising을 한 번 적용하면 다항식 image는(1,2,0)이다. 그 가중 norm 제곱은3이다. 원래 입력 norm5/2에 대한 action norm 제곱은6/5다. 이를 정규화하면 비교를 위한 다른 벡터를 얻는다.

$$
\|J_+\psi\|^2=\frac65,\qquad
\frac{J_+\psi}{\|J_+\psi\|}=\begin{pmatrix}1/\sqrt3\\\sqrt{2/3}\\0\end{pmatrix},\qquad
\frac{\|J_+^k z\|_G^2}{N}=1,\frac65,\frac85,0\quad(k=0,1,2,3).
$$

1보다 커도 잘못된 확률은 아니다. 확률이라고 정의한 적이 없기 때문이다. J+는 unitary가 아니며, 이 실험에는 그 결과를 표현하는 측정 instrument가 지정되지 않았다. 적절히 척도를 조정한 연산자를 쓰는 물리 절차를 따로 만들 수는 있지만 추가적인 모형이 필요하다.

원시 chain은 매번 정규화하지 않는다. k번 작용의 단위는 각운동량의 k제곱이고 norm 제곱은 작용의2k제곱이다. 처음 기록은 무차원이며 다음 기록은 작용² 단위다. ℏ 수치를 c배 하면 k번째 norm 제곱은 c^(2k)배가 된다. c가 양수이면 정규화된 image 방향은 유지된다. 영 image에는 정규화 벡터나 그 확률 법칙이 없으므로 null을 반환한다.

## 9. 평균 벡터는 고차원 스핀 상태를 결정하지 못한다

Casimir는2차 모멘트들의 합을 고정한다. 일반적으로 평균 벡터 길이의 제곱과 같지는 않다. 다음 관계와 길이 상한이 유용한 점검 기준이다.

$$
\sum_{i=x,y,z}\operatorname{Var}(J_i)
=\hbar^2j(j+1)-|\langle\mathbf J\rangle|^2,\qquad
|\langle\mathbf J\rangle|\le\hbar j.
$$

평균 벡터가 영이 아니면 그 방향으로 측정 축을 고른다. 그 성분의 기대값은 평균 벡터 길이이며 가능한 최대 projection 고유값 ℏj를 넘지 못한다. 영 평균에서는 상한이 바로 성립한다.

+x 방향 spin-coherent 상태의 Jx 평균은 ℏj이고 분산은0이다. 나머지 두 분산의 합은 ℏ²j다. 스핀1의 예제는 정규직교 좌표(1/2,1/√2,1/2)이며 다항식 계수는(1,2,1)에 비례한다. Casimir는 여전히2ℏ²지만 평균 벡터 길이는 ℏ다.

모든 순수 고차원 스핀이 coherent한 것은 아니다. 스핀1의 m=0 상태와 m=+1·m=−1의 같은 무게 중첩은 평균 벡터가 둘 다0이지만 Jz 법칙은(0,1,0)과(1/2,0,1/2)로 다르다.3차원 평균 화살표가 전체 상태를 표현하지 못한다. 순수 qubit의 Bloch 구 그림을 모든 j로 확대하면 안 된다.

## 10. 수치 좌표와 조건화

준비는2j·ℏ 비·최대9개 복소 원시 계수를 위한 소유 정수21개를 사용한다. 활성 좌표는 k=2j까지이며 그 뒤 성분은 모두0이어야 한다. j를 줄인다고 채워진 성분을 조용히 버리지 않는다. 선택한 행렬 차원은 최대9이며 앞의 일반적인 유도와는 별도의 계산 범위다.

가중 행렬·확률·모멘트·원시 ladder chain은 정확한 복소 유리수다. 정규직교 행렬과 정규화 진폭에는 제곱근이 들어갈 수 있어 방향성768bit 작업 연산과 바깥 반올림한512bit 관측을 쓴다. 정확한 대수적 영과0을 포함하는 수치 잔차 구간을 구별한다.

ℏ가 매우 작으면 반복한 원시 image가 작아진다. 정규화 좌표를 표시할 때는 모든 성분을 같은 양의 정확 유리수 척도로 나누고 제곱근을 계산한다. 수학적으로 같은 대표이며 복소 위상도 보존한다. a/√M=(a/L)/√(M/L²)가 식을 바꿔도 image가 같다는 이유다. L을 최대 실수·허수 성분 절댓값으로 택하면 선택 차원에서 척도 조정한 가중 질량은1/70부터18 사이다. 원시 chain·차원을 가진 norm 계정·정밀도 기준은 유지하며 매 단계 norm을 보정하지 않는다. 독립 참조는 정규직교 제곱근 ladder를 먼저 만들고 역변환하며 생산 다항식 미분 행렬을 재사용하지 않는다.

## 연구 확장: 높은 spin의 민감도와 반복되는 각도 alias

### 평균 화살표는 민감도의 인증이 아니다

능동 z 회전에서는 3차원 평균 화살표만이 아니라 전체 상태와 Jz 확률을 유지하자. 앞의 순수 상태 quantum 정보는 다음과 같다.

$$
\psi_\varphi=e^{-i\varphi J_z/\hbar}\psi_0,\qquad
 \mathcal F_Q=\frac{4\operatorname{Var}(J_z)}{\hbar^2}
 =4\left[\sum_m m^2p_m-\left(\sum_m mp_m\right)^2\right].
$$

무차원 생성자가 Jz/ℏ이므로 선언한 ℏ 척도는 소거된다. 회전 중 Jz 확률 자체는 고정된다. 생성자 분산을 정하지만 Jz 검출기만으로 누적 상대 위상을 볼 수는 없다. 정보는 적절한 판독이 국소적으로 접근할 수 있는 양이며 현재 검출기의 보증이 아니다.

### 이항분포를 갖는 coherent 준비

+x를 향하는 spin-coherent 상태의 정규직교 진폭은 다음과 같다. polynomial 좌표에는 본문의 metric을 사용한다.

$$
|{+x};j\rangle=2^{-j}\sum_{k=0}^{2j}\sqrt{\binom{2j}{k}}\,|j,j-k\rangle,
 \qquad p_k=2^{-2j}\binom{2j}{k},\qquad (z_b)_k=\binom{2j}{k}.
$$

마지막 벡터는 정규화 전 raw 계수다. z_k를 √C_k로 나누면 √C_k이고 제곱 노름은 이항합 2^(2j)다. raw 계수를 모두 같게 두면 다른 상태를 준비한다. 익숙한 이항분포를 reader에 구현된 가중 좌표계와 연결하는 예제다.

k는 Binomial(2j,1/2), m=j−k이므로 평균·분산은 다음을 준다.

$$
\mathbb E(m)=0,\quad \operatorname{Var}(m)=j/2,\quad
 \operatorname{Var}(J_z)=\hbar^2j/2,\quad \mathcal F_Q^{\rm coh}=2j.
$$

coherent 내적도 tensor product를 구성하지 않고 유도할 수 있다. p_k exp[−i(j−k)φ]를 더하고 exp(−ijφ)를 묶은 뒤 이항정리를 적용하자.

$$
\langle{+x};j|e^{-i\varphi J_z/\hbar}|{+x};j\rangle
 =\cos^{2j}(\varphi/2),\qquad F_{\rm coh}(\varphi)=\cos^{4j}(\varphi/2).
$$

fidelity의 국소 곡률은 j에 선형으로 증가한다. 단일 irreducible 표현들의 가족에 관한 설명이며 다입자 자원 모델이나 상태 준비 알고리즘을 제공한 것은 아니다.

### 두 끝점 중첩은 곡률과 모호성을 함께 바꾼다

j>0이면 최고·최저 자기 상태는 서로 다르고 직교한다. 같은 무게로 중첩하자.

$$
|{\rm cat};j\rangle=\frac{|j,j\rangle+|j,-j\rangle}{\sqrt2}\quad(j>0),\quad
 \operatorname{Var}(J_z)=\hbar^2j^2,\quad \mathcal F_Q^{\rm cat}=4j^2.
$$

초기 중첩으로 사영하는 검출기에서는 간섭 fringe가 보인다.

$$
F_{\rm cat}(\varphi)=\cos^2(j\varphi),\quad
 p_B(\varphi)=F_{\rm cat}(\varphi),\quad B=|{\rm cat};j\rangle\langle{\rm cat};j|,\quad
 \mathcal I_B=4j^2\quad(0<p_B<1).
$$

p_B를 미분해 Bernoulli 정보 식에 넣으면 확률이0 또는1이 아닌 점에서 마지막 식을 얻는다. 반대로 Jz만 측정하면 두 확률이 항상1/2여서 각도 정보가 없다. 큰 생성자 분산이 coherent 판독을 대신 구현하지는 않는다.

coherent·cat 가족에서 작은 각도 전개는 짝함수다.

$$
F(\varphi)=1-\tfrac14\mathcal F_Q\varphi^2+O(\varphi^4)
 \quad(\varphi\to0,\ j\ \text{fixed}),\qquad
 \frac{\mathcal F_Q^{\rm cat}}{\mathcal F_Q^{\rm coh}}=2j\quad(j>0).
$$

나머지는 j를 고정한 식이며 j도 함께 커지면 계수가 균일하게 제한된다는 보장은 없다. 같은 j에서 cat의 국소 quantum 정보가 더 크지만 j=1/2에서는 coherent 준비와 같은 +x 상태다. j≥1이면 cat의 spin 평균 벡터는0이지만 각도 민감도는 양수다. 평균 화살표나 순수성만으로 이 정보를 식별할 수 없다.

이제 전체 각도 범위를 보자.

$$
\psi_{\rm cat}(\varphi+\pi/j)=-\psi_{\rm cat}(\varphi),\qquad
 U_j(2\pi)=(-1)^{2j}I,\qquad
 \text{cat-ray return interval}=\pi/j\quad(j>0).
$$

cat 준비는 매 π/j마다 같은 ray를 만들고 전체 spin-j 표현은 정수/반정수에 따른 통상적인 2π 부호를 갖는다. 서로 다른 대상을 말하는 식이다. fringe가 좁아지면 고정 각도 구간에서 반복 alias도 늘어난다. 정확히 같은 ray를 만드는 두 각도는 이 준비에서 최적 측정으로도 구별할 수 없다. 전역 추론을 주장하려면 정당화된 사전 범위나 다른 준비가 필요하다.

### 정보와 alias 간격을 재현하기

정확한 코드는 이항 모멘트와 양 끝점 무게를 사용한다. 주기 열은 π로 나눈 값이므로 π 근사를 사용하지 않는다.

```python
from fractions import Fraction as Q
from math import comb
for j in [Q(1, 2), Q(1), Q(2), Q(4)]:
    n = int(2*j)
    probabilities = [Q(comb(n, k), 2**n) for k in range(n+1)]
    mean = sum((j-k)*p for k, p in enumerate(probabilities))
    variance = sum((j-k)**2*p for k, p in enumerate(probabilities))-mean**2
    coherent = 4*variance
    cat = 4*(Q(1, 2)*j*j + Q(1, 2)*(-j)**2)
    assert mean == 0 and coherent == 2*j
    print(j, coherent, cat, 1/j, cat/coherent)
```

| j | coherent FI | cat FI | cat period / pi | FI ratio |
|---|---|---|---|---|
| 1/2 | 1 | 1 | 2 | 1 |
| 1 | 2 | 4 | 1 | 2 |
| 2 | 4 | 16 | 1/2 | 4 |
| 4 | 8 | 64 | 1/4 | 8 |

![Coherent·두 끝점 spin-j 준비의 국소 정보와 생존 fringe를 비교하며, 더 촘촘한 cat fringe가 같은 ray를 더 자주 반복함을 보여 준다.](/assets/learning/quantum-spin-research-coherent-cat-aliases.ko.svg)

j=2에서는 π/2만 회전해도 cat ray가 돌아오지만 +x coherent 상태는 일반적으로 달라진다. 더 큰 국소 정보가 같은 총 실험 비용에서 무조건 유리하다는 뜻은 아니다. 준비 노력·검출기 가용성·시행 횟수·사전 범위를 같게 두지 않았다. 순수 상태 정보의 규약은 [Paris §4.1](https://arxiv.org/pdf/0804.2981)을 따른다. 이항·끝점 계산은 본문의 유한 표현을 이용해 직접 유도했다.

지원하는 j≤4에서는 두 준비 모두 기존 polynomial 입력으로 표현할 수 있다. 회전각 추정과 coherent 중첩 판독은 독립 확장이다. 현재 ladder 실습에 임의 사영이나 각도 추정 제어를 추가하지 않았다. [MIT 각운동량 강의노트](https://ocw.mit.edu/courses/8-321-quantum-theory-i-fall-2017/c491464fb065aa4cc8a99cc562475183_MIT8_321F17_lec19.pdf)는 생성자와 자기 라벨 규약의 배경이다.

#### 연구 확인 1: singlet은 따로 처리하기

j=0에서는 Jz=0인 상태 하나뿐이며 quantum 정보도0이고 회전에 의한 ray 변화가 없다. 두 끝점 cat 공식은 서로 다른 끝점을 가정하므로 적용하면 안 된다. 같은 벡터를 두 번 더해 √2로 나누어도 정규화되지 않는다.

#### 연구 확인 2: 같은 영 평균과 다른 정보

spin one의 |1,0⟩와 (|1,1⟩+|1,−1⟩)/√2는 평균 벡터가 모두0이다. Jz 분산은 각각0과 ℏ²이므로 z각 quantum 정보는0과4다. 평균 벡터는 상태도 국소 민감도도 정하지 못했다.

#### 연구 확인 3: 큰 미분으로 정확한 alias를 지울 수 없다

j=4의 cat은 F_Q=64지만 매 π/4마다 같은 ray를 만든다. 어떤 검출기든 이 준비의 φ와 φ+π/4에서 같은 법칙을 얻는다. 작은 국소 분산 경계를 보고하면서 사전 각도 모호성을 밝히지 않으면 실제 추론보다 과장하게 된다.

## 11. 해설과 함께 풀어보기

**1. j=3/2 기저의 상태 수는?**

**해설.** m=3/2,1/2,−1/2,−3/2의 네 개다. 차원은2j+1이다. Casimir는15ℏ²/4이며 가장 큰 성분값을 제곱한 것과 다르다.

**2. 다항식 행렬에서 가중 adjoint가 필요한 이유는?**

**해설.** 계수 기저의 metric이 G이므로 그 내적이 정한 adjoint는 G⁻¹A†G다. B로 metric을 I로 만든 뒤에야 일반 conjugate transpose가 물리 adjoint의 좌표식이 된다.

**3. 스핀1의 다항식 z=(1,1,1)의 Jz 법칙은?**

**해설.** 무게가(1,1/2,1), N=5/2이므로(2/5,1/5,2/5)다. 이 기저에서는 같은 raw 계수가 같은 물리 확률을 뜻하지 않는다.

**4. Raising norm6/5은 확률 오차일까?**

**해설.** 아니다. 입력 정규화를 기준으로 한 연산자 image의 norm이다. 지정된 측정 사건이 아니다. 정규화된 image의 확률(1/3,2/3,0)은 다른 수학적 대상이다.

**5. J+가 최고 상태를 없애면 어떻게 할까?**

**해설.** 영벡터와 action norm0을 얻는다. 사다리 밖에 정규화된 image가 없으므로 epsilon으로 나누거나 대체 상태를 고르지 않고 null을 반환한다.

**6. 스핀 평균 벡터가0이면 상태를 식별할 수 있을까?**

**해설.** 없다.9절의 스핀1 예제들은 세 평균이 같아도 서로 다른 순수 상태다. Jz 분포가 둘을 구별한다. 평균 화살표를 넘어 더 높은 모멘트나 전체 상태가 필요하다.

## 12. AI 연구와 남은 범위

이 단원은 표현론·metric을 반영한 선형대수·양의 이차형식·수치 조건화를 연결한다. AI의 좌표 의존적인 행렬도 내적과 기저 변환을 함께 해석해야 한다. 계수 기저에서 비대칭으로 보이는 행렬이 self-adjoint 연산자를 표현할 수 있고, 한 입력에서 norm이 잘 동작해도 연산자 전체 구조가 증명되지는 않는다. 정확한 계정 항등식을 보존하는 것과 안정적인 표시 좌표를 유지하는 것은 다른 요구다.

실험은 제한된 유한 행렬로 임의 j의 식을 예시한다. 궤도 구면조화함수·미분 연산자 정의역·tensor product 분해·일반 혼합 상태·각운동량 결합은 구현하지 않는다. 이 주제들과 남은 연구 깊이는 뒤에서 계속 다룬다. 교재는 **도서 → Interactive Book**에서 연다.

각운동량 대수와 종료의 배경은 [MIT8.321 Lecture19](https://ocw.mit.edu/courses/8-321-quantum-theory-i-fall-2017/c491464fb065aa4cc8a99cc562475183_MIT8_321F17_lec19.pdf), 스핀 표현의 문맥은 [Preskill §2.2.1](https://www.preskill.caltech.edu/ph219/chap2_15.pdf)에서 확인했다. 다항식 metric·손계산 상태·조건화 계약·해설 문제는 이 교재를 위해 전개했다.
