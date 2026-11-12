## 몸을 돌리는 것과 늘리는 것은 다르다

정사각형 종이를 책상 위에서 돌리면 꼭짓점 좌표와 변위는 크게 바뀌지만 변의 길이와 모서리 각도는 그대로입니다. 변위를 곧바로 변형률이라고 읽으면 이런 강체 운동을 재료의 늘어남으로 오해할 수 있습니다. 이번 단원은 실제 점과 선분을 이동시켜 길이·각도·면적을 관측하고, 그 결과를 소변형률과 유한 변형률의 식으로 비교합니다. 응력이나 탄성 계수를 지정해 변형을 푸는 재료 솔버는 아닙니다.

## 기준 위치와 현재 위치를 나누기

기준 좌표를 X, 현재 좌표를 x=chi(X)라고 씁니다. 변위 u=x−X는 길이 단위이고 변위 기울기 H는 무차원입니다. 변형 기울기 F는 현재 위치를 기준 위치로 미분한 것으로 I+H입니다. 일부 강의는 변위 기울기를 F라고 부르기도 하므로 기호만 보고 같은 식이라고 판단하면 안 됩니다. 이 교재에서는 H와 F를 일관되게 구분합니다.

$$
\mathbf x=\boldsymbol\chi(\mathbf X),\qquad\mathbf u=\boldsymbol\chi-\mathbf X,\qquad H=\nabla_X\mathbf u,\qquad F=\nabla_X\boldsymbol\chi=I+H.
$$

## 회전·전단·신장 매개변수

양의 a,b와 전단 g를 삼각 행렬 B에 넣고 공간 회전 R(theta)를 곱합니다. g=0이면 a,b는 각 기준 축의 신장입니다. g가 비영이면 기준 y방향의 실제 신장은√(g²+b²)이므로 b를 그대로 측정 신장이라고 부르면 안 됩니다. B는 일반적인 대칭 신장 텐서가 아니며 R도 전단이 있는 경우 극분해의 회전과 일반적으로 같지 않습니다. 두 대각값이 양수이므로 이 가족은 방향을 뒤집거나 특이해지지 않습니다.

$$
F=R(\theta)B,\qquad B=\begin{pmatrix}a&g\\0&b\end{pmatrix},\qquad a,b>0,\qquad\mathbf x=F\mathbf X+\mathbf c.
$$

## 실제 두 점으로 길이를 관측한다

기준의 두 점 차이가 dX이면 현재 차이는 F dX입니다. 공통 평행 이동 c는 차감되어 길이를 바꾸지 않습니다. 단위 방향 n의 정확한 신장은||Fn||이며 공학 변형률은 이 값에서1을 뺀 것입니다. 실제 점의 거리를 측정한 값은 변형률 텐서로 계산한 값과 대조할 독립적인 기하 관측이 됩니다. 반올림한 두 큰 좌표의 차감에서는 작은 변화가 사라질 수 있다는 점도 함께 기록합니다.

$$
\Delta\mathbf x=F\Delta\mathbf X,\qquad\lambda(\mathbf n)=\|F\mathbf n\|,\qquad e_{\rm eng}(\mathbf n)=\lambda(\mathbf n)-1.
$$

## Green 변형률은 내적의 변화를 잰다

변형 전후 길이의 제곱 차이를 쓰면 F^TF가 자연스럽게 나타납니다. 이 행렬에서 I를 빼고 절반으로 나눈 E가 Green–Lagrange 변형률입니다. E는 기준 좌표에 정의한 텐서입니다. 방향 n의 n^TEn은 길이 변화 자체가 아니라 길이 제곱 변화의 절반입니다. 0.2의 공학 변형률과0.22의 Green 성분이 동시에 맞을 수 있으므로 어떤 변형률을 보고하는지 명시해야 합니다.

$$
E=\tfrac12(F^TF-I),\qquad\|F\mathbf n\|^2-1=2\mathbf n^TE\mathbf n.
$$

## 소변형률에서 빠지는 항

F=I+H를 대입하면 E는 H의 대칭 부분과 H^TH/2의 합입니다. 소변형률 epsilon은 이차 항을 생략한 선형 근사입니다. epsilon의 비대각 성분과 공학 전단을 같은 숫자로 취급하면 배수 차이도 생깁니다. 근사의 적용 조건은 변위 자체가 작은지뿐 아니라 변위 기울기와 회전이 충분히 작은지에 관련됩니다. 큰 강체 평행 이동은 길이를 바꾸지 않지만 작은 몸체의 큰 회전은 선형화를 깨뜨릴 수 있습니다.

$$
\varepsilon=\tfrac12(H+H^T),\qquad E=\varepsilon+\tfrac12H^TH.
$$

## 이 가족의 안정적인 Green 성분

정확한 회전에서는 R^TR=I이므로 E를 B에서 직접 계산할 수 있습니다. a²−1은(a−1)(a+1)로 평가해1 근처의 작은 신장을 유지합니다. g²처럼 표현 범위 아래인 비영 항은 미정으로 남깁니다. 이 가족식과 반올림한 F에서 직접 계산한 Green 행렬을 별도로 기록합니다. 식이 공유하는 정확한 회전 성질을 일반적인 수치 행렬이나 임의 변형장의 인증으로 확장하지 않습니다.

$$
E=\frac12\begin{pmatrix}(a-1)(a+1)&ag\\ag&(b-1)(b+1)+g^2\end{pmatrix}.
$$

## 손계산: 순수 회전에서 생기는 가짜 압축

a=b=1,g=0,theta=60도로 정합니다. 실제 사각형의 변과 면적은 그대로이고 E=0입니다. 하지만 소변형률의 대각 성분은 cos(theta)−1=−0.5입니다. 이 값을 재료가50% 압축됐다고 해석하면 틀립니다. 선형 근사가 유한 회전의 이차 항을 버렸기 때문에 생긴 차이입니다. 회전한 격자의 실제 길이와 면적을 먼저 확인하고 두 변형률이 무엇을 측정하는지 설명하세요.

$$
F=R(\theta):\quad E=0,\quad\varepsilon=(\cos\theta-1)I;\qquad\theta=\pi/3:\ \varepsilon=-\tfrac12I.
$$

## 작은 회전의 작은 비영 보정

아주 작은 theta에서 cos(theta)는 컴퓨터에서1로 반올림될 수 있습니다. cos(theta)−1을 직접 차감하면 실제 음의 이차 항을 잃습니다. −2sin²(theta/2)를 사용하면 표현 가능한 작은 보정을 유지할 수 있습니다. theta/2나 그 제곱까지 표현 범위 아래이면 그 관측은 미정입니다. 정확한 강체 회전 가족의 E=0과 소변형 근사의 작은 비영 보정은 서로 다른 성질이므로 같은0으로 뭉개지 않습니다.

$$
\cos\theta-1=-2\sin^2(\theta/2).
$$

## 손계산: 단순 전단의 놓친 성분

a=b=1,theta=0,g=0.4를 고릅니다. epsilon의 비대각 값은0.2이고 두 대각은0입니다. E에서는 비대각0.2와 함께 E22=0.08이 남습니다. 처음 y방향 선분은(0.4,1)로 바뀌므로 길이비는√1.16이고 공학 변형률은 약0.07703입니다. b=1이라고 y방향 길이가 보존된다고 읽으면 안 됩니다. 선형 근사의 방향 변형률0과 실제 비영 신장을 같은 관측으로 취급하지 마세요.

$$
F=\begin{pmatrix}1&0.4\\0&1\end{pmatrix},\quad\varepsilon=\begin{pmatrix}0&0.2\\0.2&0\end{pmatrix},\quad E=\begin{pmatrix}0&0.2\\0.2&0.08\end{pmatrix},\quad e_{\rm eng}(\mathbf e_y)=\sqrt{1.16}-1.
$$

## 소변형 회전 증분은 유한 회전이 아니다

J=[[0,−1],[1,0]]에 대해 I+omega J를 생각해 보세요. 대칭 일차 변형률은0이지만 이 행렬은 유한 omega에서 직교 행렬이 아닙니다. 실제 길이비는√(1+omega²), Green 변형률은omega² I/2입니다. omega=0.5이면 E의 대각은0.125입니다. 이 지도를 정확한 회전 R(omega)와 비교하면 선형 회전 증분을 유한 회전으로 부르는 오류가 명확해집니다.

$$
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad F_{\rm control}=I+\omega J,\quad E_{\rm control}=\tfrac12\omega^2I,\quad\lambda_{\rm control}=\sqrt{1+\omega^2}.
$$

## 면적은 행렬식과 실제 다각형으로 비교한다

이번 이차원 변형에서 기준 정사각형[−1,1]²의 면적은4입니다. 현재 면적은4ab이며 전단과 공간 회전은 행렬식을 바꾸지 않습니다. 실제 변형한 네 꼭짓점으로 다각형 면적을 구해 별도로 비교합니다. 깊이를 일정하게 해석할 때 면적비와 부피비의 연결을 명시해야 하며, 이차원 그림만으로 임의 삼차원 압축성까지 결정하지 않습니다. 음의 또는0인 행렬식은 이 가족의 지원 범위 밖입니다.

$$
\det F=ab>0,\qquad A_{\rm current}=4ab,\qquad A_{\rm ref}=4.
$$

## 추가 강체 회전과 객관성

현재 상태에 직교 회전 Q를 추가하면 새 변형 기울기는 QF입니다. 기준 Green 텐서는 바뀌지 않으며 실제 길이도 같습니다. 반면 공간 좌표의 벡터 성분이나 Cauchy 응력 성분은 각각 Q와 Q sigma Q^T로 바뀝니다. 기준 텐서가 불변이라는 말과 모든 행렬 숫자가 동일해야 한다는 말은 다릅니다. 같은 E를 알았다는 이유로 재료의 현재 Cauchy 응력을 구성 관계 없이 결정하지 않습니다.

$$
E^{\prime}=\tfrac12((QF)^T(QF)-I)=E,\qquad Q^TQ=I.
$$

## 작은 길이 변화와 두 종류의 계산

큰 길이비에서1을 직접 빼면 작은 신장이 사라질 수 있습니다. q=n^TEn을 알고 있으면 제곱근 차이를 유리화한 식으로 표현 가능한 작은 공학 변형률을 계산할 수 있습니다. 이 가족 계산과 반올림한 좌표·행렬로 구한 값을 함께 남깁니다. 그래도 거의 상쇄하는 임의 방향의 모든 정보를 회복하는 것은 아닙니다. 원래 입력 해상도와 지원 범위, 미정 값과 정확한 영을 함께 보고해야 합니다.

$$
q=\mathbf n^TE\mathbf n,\qquad e_{\rm eng}=\frac{2q}{\sqrt{1+2q}+1}.
$$

한 성분이 표시 불가라고 모든 방향의 관측까지 미정인 것은 아닙니다. 주 신장 매개변수가1이고 전단이 극도로 작으면 E22는 표시 범위 아래여도 교차 항이 대각 방향의 표현 가능한 신장을 만들 수 있습니다. 따라서 성분을 반올림하기 전에 완전한 다항식에서 방향별 관측을 계산합니다. 또한 b=1.5이면 작은 전단 제곱을 표시할 수 없다는 이유로 횡방향 신장의 유한한 E22=0.625까지 지우면 안 됩니다. 표시한 텐서 성분과 별도로 계산한 방향 관측의 가용성이 다를 수 있지만 변형률의 정의를 바꾼 것은 아닙니다.

## 실험 순서와 AI 검증

먼저 평행 이동과 정확한 회전에서 실제 길이·면적 보존을 확인하세요. 회전각을 키워 소변형률과 Green 값의 차이를 보고 단순 전단과 축 신장으로 넘어갑니다. 다음에는 I+omega J 대조와 작은 회전의 안정 보정을 비교합니다. AI가 예측한 변위장에서는 적합한 변형률 정의, 실제 점의 거리, 행렬식·경계 조건·재료 관계를 함께 확인해야 합니다. 작은 공유 잔차만으로 전체 변형장이나 실제 응력을 인증하지 않습니다.

## 연구 확장: 척도 적합성과 전역 지도 점검

### 척도가 실제 지도에서 나올 수 있는지 묻기

기존 affine 가족은 변형 지도를 먼저 주므로 그 기울기의 적합성이 자동으로 성립합니다. 이번에는 매끄럽게 변하는 등방 길이 제곱 척도 C를 지정하고, 평면 기준 영역을 평면으로 보내는 매끄러운 방향 보존 지도가 이를 만들 수 있는지 거꾸로 묻습니다. C는 무차원이며 모든 점에서 양의 정부호이고 E는 대응하는 Green 목표입니다. 양수성은 국소 길이 제곱을 의미 있게 만들지만 변위를 제공하지는 않습니다. 이 새 운동학 문제에 구성 법칙·외력 해는 없습니다.

$$
C(X,Y)=e^{2\phi(X,Y)}I,\quad E=(C-I)/2,\quad\phi\in C^2,\quad C\succ0,\qquad[C]=[E]=[\phi]=1.
$$

### 모든 후보에는 국소 회전이 필요하다

이 등방 목표의 후보 F는 각 특이값이 exp(phi)여야 하므로 극분해가 표시한 척도와 proper 회전의 곱입니다. 위치에 따른 회전을 허용합니다. theta=0만 검사하면 한 후보만 제외할 뿐 모든 후보의 실패를 증명하지 못합니다. 양의 행렬식과 연속 각도 lift가 있는 작은 단순 연결 영역에서 다룹니다. 3차원 곡면 임베딩이나 결함을 처방한 물체는 다른 실현 문제이며 이 평면 계산 밖입니다.

$$
F=\nabla\chi,\quad\det F>0,\quad F^TF=C=e^{2\phi}I\quad\Longrightarrow\quad F=e^\phi R(\theta).
$$

### 혼합 미분이 회전을 제한한다

변형 기울기의 두 열은 같은 현재 위치 벡터의 편미분이므로 혼합 미분이 같아야 합니다. 첫 열을 exp(phi) 곱하기 (cos theta,sin theta), 둘째를 exp(phi) 곱하기 (−sin theta,cos theta)로 쓰고 미분한 뒤 두 방향에 투영하면 아래 스칼라 조건입니다. 각 행렬이 비특이하고 요구 길이비를 가져도 공간적으로 변하는 척도에 임의의 국소 회전을 붙일 수 없음을 보입니다.

$$
\partial_YF_{:,1}=\partial_XF_{:,2}\quad\Longrightarrow\quad\theta_X=-\phi_Y,\qquad\theta_Y=\phi_X.
$$

### 루프로 적분 장애 드러내기

두 각도 조건을 다시 미분하고 각도의 혼합 미분을 같게 두면 phi가 harmonic이어야 합니다. 또는 필요한 각도 미분을 작은 반시계 닫힌 고리에 적분하고 Green 정리를 사용합니다. 단일값 각도 lift는 같은 값으로 돌아와야 합니다. 아래는 필요조건이며 모든 위상에서 모든 양의 척도를 실현했다는 뜻은 아닙니다. 매끄럽고 수축 가능한 작은 영역에서는 각도를 2pi 주기로 읽는다고 연속적으로 조절되는 작은 루프 결함이 사라지지 않습니다.

$$
\theta_{XY}=\theta_{YX}\quad\Longrightarrow\quad\Delta\phi=0,\qquad\oint_{\partial A}(-\phi_Y\,dX+\phi_X\,dY)=\int_A\Delta\phi\,dX\,dY=0.
$$

### 양수이지만 부적합한 목표 만들기

무차원 alpha와 양의 길이 L로 아래 방사형 이차 phi를 고릅니다. Laplacian이 상수이므로 비영 alpha는 둘러싼 면적에 비례하는 비영 루프 결함을 만듭니다. 양 기준 좌표가 −L~L인 정사각형에서는 8alpha radian입니다. 특정 큰 고리의 값이 2pi 배수가 되어도 더 작은 수축 가능 고리에 비영 결함이 남습니다. 따라서 alpha가 비영이면 허용한 어떤 평면 국소 회전으로도 열린 영역에서 이 목표를 변형 기울기로 만들 수 없습니다.

$$
\phi=\frac{\alpha(X^2+Y^2)}{2L^2},\quad L>0:\qquad\Delta\phi=\frac{2\alpha}{L^2},\qquad\oint_{\partial[-L,L]^2}(-\phi_Y\,dX+\phi_X\,dY)=8\alpha.
$$

### 중심은 완전히 무변형처럼 보일 수 있다

모든 alpha에서 중심의 목표 척도는 항등, Green 변형률은 0입니다. 모서리의 척도도 alpha 부호와 무관하게 양수입니다. 이러한 점별 검사로는 공간 적분 가능성의 실패를 보지 못합니다. 표에는 중심 Green 성분·모서리 척도 성분·무차원 Laplacian·사각 고리의 각도 결함을 기록합니다. alpha=0도 별도 적합한 항등 사례로 포함합니다. 비영 alpha에서 행렬 지수로 양의 출력을 만들었다는 것만으로 AI가 실제 변위장을 예측했다고 볼 수 없습니다.

$$
C(0,0)=I,\quad E(0,0)=0,\qquad C(L,L)=e^{2\alpha}I\succ0\quad\text{for every finite real }\alpha.
$$

### 정의를 바꾸지 않고 소변형 적합성과 연결하기

별도의 소변형 목표 epsilon=phi I에서는 평면 Saint–Venant 조합이 같은 Laplacian으로 줄어듭니다. 충분히 매끄러운 변위의 대칭 기울기라면 0이어야 하므로 작은 변형에서 루프 장애와 연결됩니다. 그러나 이것이 임의 유한 Green 텐서의 적합성 공식은 아닙니다. E는 비선형 항과 원래 정의를 유지해야 합니다. 모든 유한 E에 선형 적합성 식을 적용하면 유효한 비선형 변형까지 거부할 수 있습니다.

$$
\varepsilon=\phi I:\quad\partial_{YY}\varepsilon_{XX}+\partial_{XX}\varepsilon_{YY}-2\partial_{XY}\varepsilon_{XY}=\Delta\phi\quad\text{(infinitesimal strain target)}.
$$

### 적합한 비균일 대조 구성하기

phi를 X에 선형으로 두면 Laplacian이 사라지고 대응하는 각도는 Y에 선형입니다. 아래 명시적 지도는 독립적인 양성 대조입니다. 미분하면 요구 척도를 정확히 주고 회전도 위치에 따라 달라집니다. 따라서 공간적으로 변하는 등방 신장 자체가 금지되는 것은 아닙니다. alpha=0은 표시한 나눗셈에 대입하지 않고 항등 지도 극한으로 다룹니다. 지도는 길이 단위이고 그 기울기·척도·각도는 무차원입니다.

$$
\phi=\alpha X/L,\quad\theta=\alpha Y/L,\quad\alpha\ne0:\qquad\chi(X,Y)=\frac L\alpha\begin{pmatrix}e^{\alpha X/L}\cos(\alpha Y/L)-1\\e^{\alpha X/L}\sin(\alpha Y/L)\end{pmatrix}.
$$

### 기울기와 행렬식을 직접 점검하기

두 현재 좌표를 미분해 F^TF와 행렬식을 구하면 아래 결과입니다. 수치 적합성 잔차와 별개로 국소 척도 실현·방향 보존을 확인합니다. 기준 좌표를 고정하고 alpha를 0으로 보내면 항등 지도입니다. 부동소수 차감으로 그 극한을 평가할 때는 expm1·삼각 차이 항등식이 필요할 수 있으며 모든 작은 alpha를 임의 정밀도로 평가한다고 주장하지 않습니다. 기존 affine 실습은 작은 회전·작은 변형률의 안정 계산을 유지합니다.

$$
(\nabla\chi)^T\nabla\chi=e^{2\alpha X/L}I,\quad\det\nabla\chi=e^{2\alpha X/L}>0,\qquad\alpha\to0:\ \chi(X,Y)\to(X,Y).
$$

### 국소 실현은 전역 단사성을 약속하지 않는다

alpha가 비영이면 적합한 지도도 아래 양의 Y 이동 주기로 반복됩니다. 따라서 대응하는 두 기준점을 포함한 영역은 어디서나 양의 Jacobian이어도 그 점들을 같은 현재 위치로 보냅니다. 충분히 좁은 띠에서는 이 반복을 제외할 수 있지만 국소 적합성만으로 전역 영역 제한이나 비침투 조건이 주어지지는 않습니다. 연구 검증은 목표 척도·미분 적합성·기준 위상·경계 자료·전역 지도 검사를 나누어 보고해야 합니다. 그림은 alpha=1·−1/2≤X/L≤1/2와 두 기준 띠 0≤Y/L≤2pi·2pi≤Y/L≤4pi를 사용합니다. 기존 affine 제어는 이미 적합한 별도 비교로 유지합니다.

$$
P_Y=2\pi L/|\alpha|,\quad\alpha\ne0:\qquad\chi(X,Y+P_Y)=\chi(X,Y).
$$

| alpha | 중심 Green E11 | 모서리 C11 | L 제곱 laplacian phi | 사각 각도 결함 (rad) |
| --- | --- | --- | --- | --- |
| -0.20000000 | 0.00000000 | 0.67032005 | -0.40000000 | -1.60000000 |
| 0.00000000 | 0.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| 0.10000000 | 0.00000000 | 1.22140276 | 0.20000000 | 0.80000000 |
| 0.30000000 | 0.00000000 | 1.82211880 | 0.60000000 | 2.40000000 |

```python
from fractions import Fraction as F
from math import exp
# Coordinates / L. Radial phi=alpha*(X^2+Y^2)/(2*L^2).
for alpha in [F(-1,5),F(0),F(1,10),F(3,10)]:
    centre_green = F(0)
    corner_metric = exp(float(2*alpha))
    laplacian_scaled = 2*alpha
    angle_defect = 8*alpha
    values = [float(alpha),float(centre_green),corner_metric,float(laplacian_scaled),float(angle_defect)]
    print(" ".join(f"{x:.8f}" for x in values))
```

[Bower 2.2.15절](https://solidmechanics.org/Text/Chapter2_2/Chapter2_2.htm)과 [MIT 16.20 Unit 4](https://ocw.mit.edu/courses/16-20-structural-mechanics-fall-2002/eaf046c2970c263862e604e70f89763e_unit4.pdf)는 소변형 적합성의 배경입니다. 유한 등방 목표·각도 루프 유도·명시적 conformal 지도는 별도로 유도하며 선형 변형률 식으로 일반 유한 변형 검사를 대체하지 않습니다.

![루프 결함을 가진 양의 방사형 척도·국소 적합하지만 반복하는 지도](/assets/learning/continuum-metric-compatibility-loop.ko.svg)

### 해설이 있는 연구 질문

**모든 점의 양의 C가 평면 변위의 존재를 증명하는가?** **답:** 아니다. 방사형 목표는 비영 alpha에서 양의 국소 척도여도 필요한 각도 순환이 0이 아니다.

**소변형 적합성 공식을 모든 유한 Green 텐서에 쓰는가?** **답:** 아니다. 유한 관계에는 비선형 기하가 있다. 여기의 선형 연결은 별도로 선언한 소변형 목표에 적용한다.

**양의 Jacobian과 국소 척도 실현으로 전역 단사성이 증명되는가?** **답:** 아니다. 명시적 harmonic 대조는 충분히 넓은 영역에서 Y 방향으로 반복하여 다른 기준점을 같은 현재 점으로 보낸다.

## 풀이가 있는 연습 문제

### 연습 1

60도 정확한 회전에서 E와 epsilon의 대각 성분은?

**해설:** E는0, epsilon은−0.5입니다. 실제 길이는 보존되므로 후자를 물리적50% 압축으로 해석하지 않습니다.

### 연습 2

a=b=1,g=0.4에서 E22와 y방향 공학 변형률은?

**해설:** E22=0.08이고 공학 변형률은√1.16−1≈0.07703입니다. b 자체와 실제 y신장은 다릅니다.

### 연습 3

축 신장1.2이면 공학 변형률과 Green 변형률은 같은가요?

**해설:** 공학 변형률0.2, Green 변형률(1.2²−1)/2=0.22로 다릅니다.

### 연습 4

I+0.5J가 정확한 강체 회전인가요?

**해설:** 아닙니다. 길이비√1.25이고 Green 대각0.125입니다. 대칭 일차 변형률0만으로 회전이라고 판정하면 안 됩니다.

### 연습 5

같은 변형 뒤에 공간 회전 Q를 추가하면 기준 E와 면적은?

**해설:** 둘 다 유지됩니다. 공간 벡터·응력의 성분은 올바른 회전 법칙에 따라 바뀝니다.

### 연습 6

cos(theta)가1로 반올림되면 소변형 회전 보정도 실제0인가요?

**해설:** 그렇지 않을 수 있습니다. −2sin²(theta/2)로 표현 가능한 작은 비영 항을 유지하고 표현 불가이면 미정으로 둡니다.

## 자료와 다음 연결

응력·평형의 배경은 MIT의 [Elements of Continuum Elasticity](https://www.ocw.mit.edu/courses/2-002-mechanics-and-materials-ii-spring-2004/f7b561fbe8762ff73da9236e6c0fff73_lec7.pdf)와 Alberta의 [Cauchy Stress Tensor](https://engcourses-uofa.ca/stress/cauchy-stress-as-a-linear-map/), 변형률 정의와 회전은 MIT의 [The Concept of Strain](https://ocw.mit.edu/courses/2-080j-structural-mechanics-fall-2013/71034688fba0ff26d25653a4bd236a4a_MIT2_080JF13_Lecture2.pdf) 및 Alberta의 [Strain Measures](https://engcourses-uofa.ca/displacement-and-strain/strain-measures/)를 참고하세요. 선택한 affine 장과 손계산·산술 관측은 이 교재의 예제이며 일반 재료 해석의 인증이 아닙니다.
