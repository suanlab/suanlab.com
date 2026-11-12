## 같은 점에서도 면의 방향이 중요하다

두꺼운 책을 위에서 누르는 힘과 옆으로 미는 힘을 생각해 보세요. 한 점의 내부 힘을 숫자 하나로 표현하면 서로 다른 면에서 받는 힘의 방향을 설명하기 어렵습니다. 응력은 면의 단위 법선을 받아 단위 면적당 힘인 traction을 만드는 사상입니다. 먼저 어떤 점의 어떤 현재 면인지 정한 뒤 힘의 법선 성분과 접선 성분을 분리해야 합니다. 이 단원은 응력의 정의와 힘의 수지를 다루며 재료의 변위를 자동으로 계산하지 않습니다.

## 선수 지식·부호·실제 단위

행렬과 벡터의 곱, 내적, 모멘트, 이전 운동량 보존 단원을 복습하세요. 응력은 Pa, traction도 Pa이지만 실제 힘은 N입니다. 힘의 모멘트는 N·m이고 체적력 밀도는 N/m³입니다. 인장을 양의 법선 응력으로 정합니다. 압력을 받는 면의 외향 법선과 힘 방향은 반대일 수 있습니다. 응력 값과 힘을 같은 세로축에서 빼거나, 이차원 그림의 선 길이를 그대로 면적으로 사용하지 마세요.

$$
[\sigma]=[\mathbf t]=\mathrm{Pa}=\mathrm{N/m^2},\qquad [\mathbf F]=\mathrm N,\quad [M_z]=\mathrm{N\,m},\quad [\mathbf f]=\mathrm{N/m^3}.
$$

## 법선에서 traction으로

현재 면의 단위 법선을 n이라 하면 traction은 sigma n입니다. 면적 벡터를 넣는 경우에는 결과가 힘이므로 입력의 정의를 먼저 확인해야 합니다. 법선 n과−n은 같은 위치의 서로 반대 면을 가리킵니다. 같은 응력에서 법선을 뒤집으면 traction도 뒤집혀 작용·반작용을 표현합니다. 이번 계산은 평면 내 대칭 응력이며 평면 밖 전단과 법선 응력은0으로 두는 plane-stress 예제입니다.

$$
\mathbf t(\mathbf n)=\boldsymbol\sigma\mathbf n,\qquad \|\mathbf n\|=1,\qquad \mathbf t(-\mathbf n)=-\mathbf t(\mathbf n).
$$

## 법선 응력과 부호 있는 전단

법선 방향 성분은 n과 traction의 내적으로 구합니다. 남는 벡터는 면을 따라 작용하는 전단 traction입니다. 이차원에서는 법선을 반시계 방향으로90도 돌린 접선 m을 정해 부호 있는 전단을 정의할 수 있습니다. 접선의 선택을 반대로 하면 스칼라 전단의 부호도 반대지만 물리적 전단 벡터는 같습니다. 접선 방향과 부호 약속을 숨긴 채 크기만 비교하면 두 계산이 실제로 일치하는지 판단하기 어렵습니다.

$$
\sigma_n=\mathbf n^T\boldsymbol\sigma\mathbf n,\qquad \tau=\mathbf m^T\boldsymbol\sigma\mathbf n,\qquad \mathbf t=\sigma_n\mathbf n+\tau\mathbf m.
$$

## 응력 대칭성과 가정의 범위

일반적인 couple 없는 Cauchy 연속체에서는 각운동량 수지로 응력의 대칭성을 얻습니다. 모든 고급 연속체 모형에 이를 무조건 강제하는 것은 아닙니다. 이번 예제는 대칭인 이차원 응력장을 선택하므로 면 힘의 중심 모멘트를 따로 계산해 대칭성과 적분 부호를 점검할 수 있습니다. 작은 수치 모멘트는 선택한 가족의 진단이지 일반화 연속체의 couple나 회전 관성을 검증한 결과가 아닙니다.

$$
\boldsymbol\sigma=\begin{pmatrix}\sigma_{xx}&\tau_{xy}\\\tau_{xy}&\sigma_{yy}\end{pmatrix}=\boldsymbol\sigma^T.
$$

## 손계산: 기울어진 면의 힘

sigma=[[3,1],[1,−1]]Pa와 n=(1,1)/√2를 고릅니다. traction은(2√2,0)Pa이며 n 방향 응력은2Pa입니다. m=(−1,1)/√2를 쓰면 전단은−2Pa입니다. 면적2m²의 힘은(4√2,0)N입니다. traction이 x축을 향하더라도 면의 법선은45도라는 점을 확인하세요. 힘의 방향, 면의 방향과 법선·전단의 스칼라 값은 서로 다른 정보입니다.

$$
\boldsymbol\sigma=\begin{pmatrix}3&1\\1&-1\end{pmatrix}\mathrm{Pa},\quad \mathbf t=(2\sqrt2,0)\mathrm{Pa},\quad \sigma_n=2\mathrm{Pa},\quad\tau=-2\mathrm{Pa},\quad\mathbf F=(4\sqrt2,0)\mathrm N.
$$

## 평면 내 주응력과 비유일한 방향

법선과 traction이 나란한 방향에서는 전단이0입니다. 대칭 행렬의 고유벡터가 그 방향이며 고유값이 평면 내 주응력입니다. 두 값이 같으면 선호되는 평면 내 방향이 없으므로 방향을 임의의0도로 보고하지 않습니다. plane-stress의 세 번째 주응력0까지 포함한 삼차원 최대값과도 구분해야 합니다. 같은 물리적 축은 n과−n으로 나타낼 수 있어 각도 자체는 π를 주기로 읽습니다.

$$
\sigma_{\pm}=\frac{\sigma_{xx}+\sigma_{yy}}2\pm\sqrt{\left(\frac{\sigma_{xx}-\sigma_{yy}}2\right)^2+\tau_{xy}^2}.
$$

작은 주응력은 비슷한 두 반올림 수를 차감할 때 사라질 수 있습니다. 대각 성분이1Pa와10⁻²⁰Pa인 텐서의 고유값은 이미 그 두 숫자이지만, 평균과 반차를 거쳐 빼면 작은 값을 잘못0으로 바꿀 수 있습니다. 계산은 대각 텐서의 값을 직접 유지합니다. 일반적인 대칭 텐서는 성분의 크기를 정규화해 절댓값이 큰 근을 구하고, 다른 근은 아직 표시 범위로 반올림하지 않은 행렬식을 그 근으로 나누어 구합니다. 이때 나눗셈의 근도 최종 표시 반올림 전에 유지합니다. 실제 특이 텐서의 영주응력과 약한 응력 방향을 구분하기 위한 계산입니다.

## 평면 내 압축과 삼차원 정수압

sigma2D=−pI2이면 모든 평면 내 법선에서 traction이−pn이고 전단은0입니다. 이것은 정수압의 평면 내 방향성을 설명하는 좋은 예제입니다. 하지만 이번 plane-stress 모형은 sigma_zz=0이므로 전체 삼차원 정수압 상태라고 부르면 안 됩니다. 진짜 삼차원 정수압에서는 sigma_zz도−p입니다. 이 차이는 체적 응력의 평균이나 세 주응력을 해석할 때 중요합니다.

$$
\boldsymbol\sigma_{2D}=-pI_2\quad\Longrightarrow\quad\mathbf t=-p\mathbf n,\quad\tau=0;\qquad \text{3D hydrostatic: }\boldsymbol\sigma=-pI_3.
$$

## 공간에 따라 변하는 응력 가족

이제 현재 좌표의 응력이 위치에 선형으로 변하도록 합니다. 법선 성분과 전단 성분의 기울기를 나누면 힘이 생기는 원인을 추적하기 쉽습니다. gradX, gradY, crossGradient는 Pa/m의 응력 기울기이며 힘 자체가 아닙니다. 이 가족의 발산은 상수이지만 각 면에 작용하는 힘은 위치와 면 방향에 따라 달라집니다. 응력을 균일하게 만드는 것과 힘이 균형을 이루는 것은 같은 조작이 아닙니다.

$$
\boldsymbol\sigma(x,y)=\begin{pmatrix}a+g_x x&b+ky\\b+ky&d+g_y y\end{pmatrix},\qquad\nabla\!\cdot\boldsymbol\sigma=(g_x+k,g_y).
$$

## 실제 직사각형 경계의 적분

중심(cx,cy), 반폭 hx,hy와 깊이 B인 직사각형 프리즘을 생각합니다. 좌우 면적은2hyB, 위아래 면적은2hxB입니다. 각 면에서 바깥 법선과 현재 위치를 이용해 traction을 적분해야 합니다. 같은 힘의 크기를 모두 더하는 대신 방향 있는 벡터 합을 구하세요. 모멘트의 기준점은 직사각형 중심으로 고정합니다. 이 예제에서 일정한 체적력은 중심에 대한 총모멘트가0입니다.

$$
V=4h_xh_yB,\qquad\mathbf F_s=\int_{\partial V}\boldsymbol\sigma\mathbf n\,dA,\qquad M_z=\int_{\partial V}\bigl((\mathbf x-\mathbf c)\times\boldsymbol\sigma\mathbf n\bigr)_z\,dA.
$$

## 체적력과 정적 평형의 손계산

기울기를 gx=1, k=2, gy=−0.5Pa/m로 고르고 hx=1m, hy=0.5m, B=2m로 둡니다. 부피4m³에 대한 표면 힘은(12,−2)N입니다. 체적력 밀도를(−3,0.5)N/m³로 명시하면 체적력은(−12,2)N이 되어 정적 순힘이0입니다. 같은 응력장에서 체적력만 누락한 대조는(12,−2)N을 남깁니다. 이 대조는 정적 평형 가정의 실패이며 밀도 없이 실제 가속도까지 결정하지 않습니다.

$$
\mathbf F_s=V\nabla\!\cdot\boldsymbol\sigma=(12,-2)\mathrm N,\qquad\mathbf F_b=V\mathbf f=(-12,2)\mathrm N,\qquad\mathbf F_s+\mathbf F_b=0.
$$

## 앞 단원의 운동량 유속과 부호

이전 기체 단원에서는 운동량 유속에+p가 들어갔지만 압력 traction에는−p가 들어갑니다. 이는 모순이 아닙니다. 보존식의 바깥 유속과 물질에 작용하는 표면 힘은 적분식에서 서로 반대편에 놓입니다. 일반적인 대류 운동량 유속에서 Cauchy 응력을 빼면 압력 항이 더해집니다. 이번 정적 응력 실험은 대류 속도를 따로 풀지 않으므로 그 항까지 계산했다고 주장하지 않습니다.

$$
\text{momentum flux}=\rho\mathbf u\otimes\mathbf u-\boldsymbol\sigma;\qquad\boldsymbol\sigma=-pI\ \Longrightarrow\ \rho\mathbf u\otimes\mathbf u+pI.
$$

## 좌표 회전과 같은 물리 상태

벡터와 응력장을 같은 능동 회전 Q로 돌리면 traction도 Q만큼 회전해야 합니다. 법선 응력과 접선 약속을 함께 돌린 전단 값, 주응력은 유지됩니다. 행렬 성분이 바뀌었다고 다른 하중이 생긴 것은 아닙니다. 반대로 응력만 돌리고 법선을 그대로 두면 일반적으로 다른 면의 관측이 됩니다. 기준계 변환을 AI 모델의 검사에 사용할 때도 입력과 출력 중 어떤 것이 변해야 하는지 먼저 정해야 합니다.

$$
\boldsymbol\sigma^{\prime}=Q\boldsymbol\sigma Q^T,\qquad\mathbf n^{\prime}=Q\mathbf n,\qquad\mathbf t^{\prime}=Q\mathbf t.
$$

## 영면적·작은 기하·산술 미정

면적0의 적분 힘은0이지만 수학적으로 주어진 법선에서의 traction 질의는 별개입니다. 무너진 직사각형에는 정상적인 닫힌 제어면 관측을 만들지 않습니다. 양의 반폭이10⁻⁸m 아래이면 지원 기하 밖으로 표시하며 실제0으로 바꾸거나 넓히지 않습니다. 작은 비영 곱이 부동소수점 범위 아래로 사라지면 그 관측은 미정입니다. 반올림과 상쇄를 전부 회복하는 임의 정밀도 솔버는 아닙니다.

## 검증·구성 관계·AI의 한계

먼저 기울어진 면 손계산과 법선 반전을 검사하고, 평면 내 압축·순수 전단을 비교하세요. 다음에는 affine 응력의 표면 힘·모멘트와 체적 발산 적분, 체적력 누락 대조를 봅니다. 실제 장을 적분한 별도 참조와 같은 식의 대입 잔차는 구분합니다. 정적 평형만으로 실제 변위나 재료 상수를 찾을 수 없으며 변형 적합성·경계 조건·구성 관계가 필요합니다. 다음 단원에서는 실제 선분과 격자를 변형해 변형률의 정의와 근사를 대조합니다.

## 연구 확장: 방향 관측으로 응력 식별하기

### 미지수와 측정 성분 정하기

같은 한 점의 대칭 평면 Cauchy 응력을 유지합니다. 보충 관측은 전체 traction 벡터 대신 법선 응력만 기록한다고 가정합니다. 아래 세 미지수는 Pa 단위입니다. 여기의 p는 평면 내 평균 응력이며 압축을 양수로 둔 유체 압력이 아닙니다. tau는 고정 좌표의 xy 성분이지 모든 기울어진 면의 부호 있는 전단은 아닙니다. Frobenius 노름은 이 매개변수의 Euclidean 노름에 비례합니다. 조건수를 논하기 전에 비교 척도를 선언하는 이유입니다.

$$
\sigma=\begin{pmatrix}p+d&\tau\\\tau&p-d\end{pmatrix},\qquad s=(p,d,\tau)^T,\qquad\|\sigma\|_F^2=2\|s\|_2^2.
$$

### 스칼라 관측 하나 유도하기

단위 법선을 법선 투영의 이차식에 대입하고 배각 공식을 쓰면 표시한 전방 행입니다. 반대 법선의 전체 traction은 반전하지만 스칼라 관측은 같습니다. 자료는 같은 점의 바뀌지 않은 물리 상태를 읽으며 재료 하중을 바꾼 뒤 측정한 값은 다른 역문제입니다. 실제 시편을 반복 절단해도 응력이 유지된다거나 이 식이 실제 계측기를 모형화한다고 주장하지 않습니다.

$$
n(\theta)=(\cos\theta,\sin\theta)^T:\qquad m(\theta)=n^T\sigma n=p+d\cos2\theta+\tau\sin2\theta,\quad m(\theta+\pi)=m(\theta).
$$

### 읽음 수 대신 관측 정보를 역산하기

알려진 방향을 A에 쌓습니다. 독립·동일 분산 오차에서 세 매개변수 방향이 관측 가능하면 아래 최소 제곱 추정입니다. pi를 주기로 서로 다른 세 축은 배각 단위 원의 다른 세 점입니다. 한 직선은 원과 최대 두 점에서 만나므로 세 행은 독립입니다. 같은 축의 반복은 빠진 방향을 만들지 않습니다. 반복 관측을 더하면 보이는 성분의 잡음은 줄일 수 있어도 그것만으로 순위를 복원하지 못합니다.

$$
m=As+e,\quad A_{j,:}=(1,\cos2\theta_j,\sin2\theta_j),\qquad\widehat s=(A^TA)^{-1}A^Tm\quad(\operatorname{rank}A=3).
$$

### 전체 traction은 다른 관측이다

전체 traction 하나는 두 성분을 주지만 대칭 이차원 응력에는 여전히 보이지 않는 한 방향이 있습니다. 아래 접선 dyad의 배수를 더해도 그 traction은 바뀌지 않습니다. 비평행한 두 법선의 전체 traction은 교차 투영의 대칭 일관성을 만족할 때 사상을 정할 수 있습니다. 기록하는 자료가 다르므로 스칼라 세 개의 결과와 모순이 아닙니다. 관측되지 않는 방향의 정규화는 응력을 선택할 뿐 측정이 그것을 식별했다고 증명하지 않습니다.

$$
t=\sigma n,\quad n^Tz=0,\quad\|n\|=\|z\|=1:\qquad (\sigma+\beta zz^T)n=t\quad\text{for every real }\beta.
$$

### 유일성과 잡음 안정성 구분하기

최소 제곱식에서 실제 전방 관계를 빼면 pseudoinverse 노름의 결정적 오차 상한입니다. 독립·영평균·동일 분산 오차에서는 공분산도 얻습니다. 순위가 충분해도 최소 특이값이 작으면 잡음이 크게 증폭됩니다. 상한은 보정한 방향과 가산 응력 오차를 가정하며 방향 불확실성·계측기 응답·모형 불일치는 생략합니다. 상관 잡음에서는 선언한 모형에 맞춰 공분산과 가중 추정기를 다시 유도해야 합니다.

$$
\mathbb E e=0,\quad\operatorname{Cov}(e)=\nu^2I:\quad\operatorname{Cov}(\widehat s)=\nu^2(A^TA)^{-1},\quad\|\widehat s-s\|_2\le\frac{\|e\|_2}{\sigma_{\min}(A)},\quad[\nu]=\mathrm{Pa}.
$$

### 방향을 분산하여 정보 균형 맞추기

0도·60도·120도의 세 축에서는 배각 열을 합하면 Gram 행렬이 대각입니다. 명시한 잡음 모형에서 복원된 평균과 두 편차 좌표의 오차는 서로 무상관입니다. 이 대칭 설계는 조건수 제곱근 2를 이룹니다. 모든 관측에 평균 열이 들어가지만 두 각도 성분은 고정된 제곱 길이를 나누므로 1이 아닙니다. 또한 조건수만으로 절대 응력 불확실성의 크기까지 정해지지는 않습니다.

$$
(\theta_1,\theta_2,\theta_3)=(0,\pi/3,2\pi/3):\quad A^TA=\operatorname{diag}(3,3/2,3/2),\qquad\kappa_2(A)=\sqrt2.
$$

### 선언한 척도에서의 최적성을 증명하기

세 행을 어떻게 고르든 평균 열의 제곱 노름이 3이므로 Gram 최대 고윳값은 3 이상입니다. 다른 두 매개변수 좌표로 제한한 2차원 block의 trace는 3이어서 작은 Rayleigh 값은 3/2 이하입니다. 전체 행렬의 최소 고윳값은 그보다 클 수 없습니다. 따라서 앞의 축이 달성하는 아래 하한입니다. 이 변수 척도의 스칼라 세 관측에 대한 것이며 모든 센서 비용·잡음 상관·실험 목표의 최적성을 뜻하지 않습니다.

$$
\lambda_{\max}(A^TA)\ge3,\qquad\lambda_{\min}(A^TA)\le3/2\quad\Longrightarrow\quad\kappa_2(A)\ge\sqrt2\quad\text{for three full-rank scalar observations}.
$$

### 거의 같은 방향은 순위를 유지해도 정밀도를 잃는다

축을 0·delta·2delta로 고릅니다. 다른 행에서 첫 행을 빼고 삼각 항등식을 쓰면 아래 행렬식입니다. 선언한 구간에서는 양수이지만 delta가 줄면 삼차로 사라집니다. 정확한 delta=0은 별도의 순위 부족 입력입니다. 작은 비영 행렬식만으로 조건수 계산이 끝나는 것은 아니며 반올림한 각도·행렬식을 정확히 반복된 축으로 몰래 바꾸면 안 됩니다. 다음 식에서 특이값을 직접 구합니다.

$$
(\theta_1,\theta_2,\theta_3)=(0,\delta,2\delta),\quad0<\delta<\pi/2:\quad\det A=4\sin^2\delta\sin2\delta>0,\qquad\det A\sim8\delta^3\quad(\delta\to0).
$$

### 상쇄 없이 작은 고윳값 평가하기

세 축의 중심에 맞춰 편차 매개변수 좌표를 회전하면 Gram 행렬이 2x2 block과 스칼라로 나뉩니다. block의 trace가 T, 행렬식이 D입니다. 가까운 큰 두 수를 차감하는 대신 곱 관계로 작은 고윳값을 구하고 나머지 고윳값과 함께 최소 특이값을 계산합니다. 표는 행렬식·최소 특이값·조건수·nu 제곱으로 나눈 최악 방향 매개변수 분산입니다. 유한 수치 비교이며 순위 상실 근방의 보편 부동소수 오차 상한이 아닙니다.

$$
c=\cos2\delta,\quad T=4+2c^2,\quad D=8\sin^4\delta:\quad\lambda_+=\tfrac12(T+\sqrt{T^2-4D}),\quad\lambda_-=D/\lambda_+,\quad\lambda_0=2\sin^22\delta,\quad\sigma_{\min}(A)=\sqrt{\min(\lambda_-,\lambda_0)}.
$$

### 익숙한 세 축 예 손계산하기

0도·45도·90도에서는 행렬 풀이 없이 역산식을 씁니다. 원래 텐서 [[3,1],[1,−1]] Pa의 법선 관측은 (3,2,−1) Pa여서 p=1·d=2·tau=1 Pa입니다. 독립 계측 오차가 복원 매개변수의 독립 오차를 뜻하지는 않습니다. 표시한 공분산에는 비영 p–tau 항이 있습니다. 등방 응력은 성분을 정확히 읽어도 선호 주축이 없으며 이것은 관측 행렬의 순위 상실과 다른 물리적 비유일성입니다. 기존 traction·affine 힘 실습은 그대로 유지합니다.

$$
\theta=(0,\pi/4,\pi/2):\quad p=\tfrac12(m_1+m_3),\quad d=\tfrac12(m_1-m_3),\quad\tau=m_2-p,\quad\operatorname{Cov}(\widehat s)/\nu^2=\begin{pmatrix}1/2&0&-1/2\\0&1/2&0\\-1/2&0&3/2\end{pmatrix}.
$$

| delta (도) | A 행렬식 | 최소 특이값 | 조건수 | 최악 분산 / nu 제곱 |
| --- | --- | --- | --- | --- |
| 5.00000000 | 0.00527621 | 0.00881572 | 276.45298154 | 12867.20641698 |
| 15.00000000 | 0.13397460 | 0.08083776 | 28.99405294 | 153.02820107 |
| 30.00000000 | 0.86602540 | 0.33763747 | 6.20274201 | 8.77200187 |
| 60.00000000 | 2.59807621 | 1.22474487 | 1.41421356 | 0.66666667 |

```python
from math import sin, cos, pi, sqrt
for degrees in [5,15,30,60]:
    delta = degrees*pi/180
    c = cos(2*delta)
    trace = 4+2*c*c
    determinant = 8*sin(delta)**4
    large = (trace+sqrt(trace*trace-4*determinant))/2
    small = determinant/large
    third = 2*sin(2*delta)**2
    minimum, maximum = min(small,third), max(large,third)
    det_A = 4*sin(delta)**2*sin(2*delta)
    values = [float(degrees),det_A,sqrt(minimum),sqrt(maximum/minimum),1/minimum]
    print(" ".join(f"{x:.8f}" for x in values))
```

[Alberta의 Cauchy 응력 노트](https://engcourses-uofa.ca/stress/cauchy-stress-as-a-linear-map/)는 traction 사상·대칭성의 배경입니다. 스칼라 관측 모형·영방향·잡음 계산·조건부 설계 하한은 여기서 직접 유도하며 계측기를 인증하거나 구성 응답을 결정하지 않습니다.

![방향 응력 측정·군집 축의 조건수·서로 다른 관측 범위](/assets/learning/continuum-stress-directional-identification.ko.svg)

### 해설이 있는 연구 질문

**한 축을 반복 측정하면 모든 응력 성분이 식별되는가?** **답:** 아니다. 보이는 잡음은 줄여도 같은 미관측 매개변수 방향이 남는다.

**작은 비영 행렬식은 식별 불가능 텐서와 같은가?** **답:** 아니다. 군집 설계는 역산 가능해도 잡음을 크게 증폭한다. 정확한 순위 상실과 나쁜 조건수는 다르다.

**성분을 식별하면 주축도 항상 유일한가?** **답:** 아니다. 등방 평면 응력은 측정 설계가 성분을 식별해도 선호 축이 없다.

## 풀이가 있는 연습 문제

### 연습 1

기울어진 면 손계산의 traction·법선 응력·전단은?

**해설:** (2√2,0)Pa,2Pa,−2Pa입니다. 면적2m²의 힘은(4√2,0)N입니다.

### 연습 2

같은 응력에서 n을−n으로 바꾸면 무엇이 바뀌나요?

**해설:** traction과 전단 벡터는 반전합니다. 법선 응력은 같습니다. 접선도 함께 반전하는 약속에서는 스칼라 전단도 같습니다.

### 연습 3

sigma2D=−2I2인 plane-stress 예제가 완전한 삼차원 정수압인가요?

**해설:** 아닙니다. 평면 밖 응력은0으로 두었습니다. 삼차원 정수압이면 sigma_zz도−2Pa여야 합니다.

### 연습 4

부피4m³, 발산(3,−0.5)N/m³이면 표면 힘과 균형 체적력 밀도는?

**해설:** 표면 힘(12,−2)N, 체적력 밀도(−3,0.5)N/m³입니다.

### 연습 5

균일 응력에서 순힘0이면 모든 면의 힘이0인가요?

**해설:** 아닙니다. 서로 다른 면의 방향 있는 힘이 상쇄할 수 있습니다. 크기의 합과 벡터 합은 다릅니다.

### 연습 6

정적 평형과 응력만으로 실제 변위를 유일하게 구할 수 있나요?

**해설:** 아닙니다. 변형 적합성, 경계 조건과 재료 구성 관계 등이 더 필요합니다.

## 자료와 다음 연결

응력·평형의 배경은 MIT의 [Elements of Continuum Elasticity](https://www.ocw.mit.edu/courses/2-002-mechanics-and-materials-ii-spring-2004/f7b561fbe8762ff73da9236e6c0fff73_lec7.pdf)와 Alberta의 [Cauchy Stress Tensor](https://engcourses-uofa.ca/stress/cauchy-stress-as-a-linear-map/), 변형률 정의와 회전은 MIT의 [The Concept of Strain](https://ocw.mit.edu/courses/2-080j-structural-mechanics-fall-2013/71034688fba0ff26d25653a4bd236a4a_MIT2_080JF13_Lecture2.pdf) 및 Alberta의 [Strain Measures](https://engcourses-uofa.ca/displacement-and-strain/strain-measures/)를 참고하세요. 선택한 affine 장과 손계산·산술 관측은 이 교재의 예제이며 일반 재료 해석의 인증이 아닙니다.
