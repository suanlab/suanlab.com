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
