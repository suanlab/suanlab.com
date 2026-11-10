## 같은 PDE를 세 가지 질문으로 바꾸기

Poisson 문제 -u″=f를 [0,1]에서 풉니다. 정확해를 먼저 u=L(1-x)+Rx+x(1-x)+A sin(πx)로 선택하고 f=2+Aπ²sin(πx)를 만듭니다. 이것은 제조해 방법입니다. 정확해를 알고 시작하면 경계 처리, 조립, 선형 해법, 이산화 오차를 각각 검산할 수 있습니다.

FD는 절점의 미분 근사, FV는 작은 셀의 흐름 수지, FEM은 시험 함수에 대한 약한 식에서 출발합니다. 세 이름을 하나의 stencil에 붙이는 대신 실제 미지수 위치와 load를 다르게 구성합니다. 위 위치 그림과 표를 함께 보면 FV의 셀 중심이 FD·FEM의 내부 절점과 다르다는 점이 드러납니다.

## 유한차분과 유한체적

균일 간격 h에서 절점 FD는 다음 식을 풉니다.

$$
\frac{2u_i-u_{i-1}-u_{i+1}}{h^2}=f(x_i).
$$

FV는 J=-u′에 대해 셀 적분을 J_right-J_left≈h f_center로 만듭니다. 내부 face에서는 같은 flux를 양옆 셀이 반대 부호로 사용합니다. 경계 셀의 중심에서 Dirichlet 경계까지 거리는 h가 아니라 h/2입니다. 따라서 양 끝 셀의 대각 계수는 3/h², 내부는 2/h²가 되고 경계값의 우변 계수는 2/h²입니다. 이 실험의 FV 값은 셀 중심 점값의 근사로 해석하며 정확한 셀 평균이라고 부르지 않습니다.

## 유한요소의 약한 식

경계에서 0인 시험 함수 v로 적분하고 부분적분하면 다음을 얻습니다.

$$
\int_0^1u'v'\,dx=\int_0^1fv\,dx,\qquad
K_{ij}=\int\phi_i'\phi_j'\,dx,\quad b_i=\int f\phi_i\,dx.
$$

선형 hat 함수를 사용하면 균일 격자의 K는 대각 2/h, 이웃 -1/h입니다. Load는 단순히 f_i를 넣지 않고 각 요소에서 두 점 Gauss 적분으로 계산합니다. 다항식에서는 정확할 수 있지만 정현파 load의 적분은 여전히 근사입니다. K를 h로 나누어 FD와 비교해도 우변의 정의가 다릅니다.

## 손계산과 직접 풀기

A=L=R=0이면 정확해는 x(1-x), source는 2입니다. FD의 이차 차분은 이 이차 함수의 미분을 정확히 재현하므로 내부 절점 오차는 반올림 수준입니다. 선형 FEM도 이 1D 상수 계수 문제에서 정확한 load를 사용하면 절점 값이 정확하지만 요소 내부의 선형 보간은 곡선과 다릅니다.

직접 풀기: h=1/4인 FV 첫 중심 x=h/2에서 정확한 점값을 넣은 경계 stencil을 계산하세요. 풀이: (3u(h/2)-u(3h/2))/h²=3/2로 source 2와 다릅니다. 계산 FV 값은 이 특별한 이차 모형에서 정확한 점값보다 h²/4만큼 높습니다. 세 방법의 점값이 항상 같다고 가정하면 이 경계 효과를 놓칩니다.

## Simulator 학습과 방법 선택

Surrogate나 physics-informed 모델을 평가할 때 미지수가 점값인지 셀 평균인지, weak residual인지 point residual인지 구분해야 합니다. 작은 행렬 잔차는 조립한 이산 문제를 잘 풀었다는 뜻이며 연속 PDE 해의 정확도와는 별개입니다. 큰 차원의 FEM, 비구조 mesh, 비선형 flux는 추가 구현이 필요합니다. 현재 실습은 1차원 constant-coefficient Poisson 문제의 명확한 비교 기준입니다.

참고: [MIT 수치 PDE](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/), [Clawpack 저자들의 유한체적 설명](https://www.clawpack.org/riemann_book/html/Approximate_solvers.html).
