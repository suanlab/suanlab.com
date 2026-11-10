## 조건 사건 대신 정보 전체에 조건 걸기

조건부기대 E[X|G]는 관측 정보 G로 표현할 수 있는 확률변수입니다. 하나의 숫자가 아니라 표본점마다 값을 갖습니다. 적분가능한 X와 부분 σ-대수 G를 가정합니다. 정의는 G-가측성과 모든 G 사건에서의 적분 보존입니다. 관측하지 못하는 원자 내부를 구분하는 값을 내면 정보 G만 사용하는 예측이 아닙니다.

$$
Y=E[X\mid\mathcal G]:\quad Y\text{ is }\mathcal G\text{-measurable},\quad\int_A Y\,dP=\int_A X\,dP\ \forall A\in\mathcal G.
$$

## 유한 원자에서 직접 구성하기

기준 X=(0,2,1,3), P=(0.1,0.2,0.3,0.4)이고 정보 원자는 {0,1}, {2,3}입니다. 각 원자 안에서는 상수여야 하므로 원자의 가중평균을 택합니다. 첫 값은 (0·0.1+2·0.2)/0.3=4/3이고 둘째는 (1·0.3+3·0.4)/0.7=15/7입니다. 각 원자에서 적분이 보존되므로 그 합집합인 모든 G 사건에서도 보존됩니다.

$$
Y(\omega)=\frac{\sum_{i\in A}p_iX_i}{P(A)}\quad(\omega\in A,\ P(A)>0).
$$

## 기대의 tower와 정보 해상도

전체 공간을 사건으로 넣으면 E[E[X|G]]=E[X]=1.9입니다. 모든 정보를 주면 조건부기대는 원래 X가 되고, 자명한 정보만 주면 전체 평균이라는 상수가 됩니다. 더 작은 정보 H가 G 안에 포함되면 G로 평균낸 뒤 H로 다시 평균낸 결과는 처음부터 H로 평균낸 것과 같습니다. 일반 공간의 존재 정리는 Radon–Nikodym 정리로 연결되며 유일성은 거의 확실한 의미입니다.

$$
\mathcal H\subseteq\mathcal G\implies E[E[X\mid\mathcal G]\mid\mathcal H]=E[X\mid\mathcal H].
$$

## 영 원자와 최소제곱 해석

원자의 확률이 0이면 비율은 정의되지 않지만 그 원자 위 어떤 상수 버전을 택해도 적분 보존에는 영향이 없습니다. 실습은 명시적으로 0을 택합니다. 이것이 영 사건의 초등 조건부확률을 0으로 정의한다는 뜻은 아닙니다. X가 L2이면 조건부기대는 G로 예측할 수 있는 변수들 중 제곱오차를 최소화합니다. 가중 잔차가 모든 원자에서 적분 0이 되는 직교성을 확인하면 이 해석을 볼 수 있습니다.

$$
E[(X-Z)^2]=E[(X-Y)^2]+E[(Y-Z)^2]\quad(Y=E[X\mid\mathcal G],\ Z\in L^2(\mathcal G)).
$$

## 확인 문제와 해설

1. 전체 평균은 보존되나요? **예. G에 Ω가 포함됩니다.**
2. 영 원자의 선택값은 유일한가요? **아니요. 조건부기대의 버전은 영집합에서 달라도 됩니다.**
3. 정보가 늘면 최적 제곱오차가 증가하나요? **아니요. 이전 예측도 새 정보에서 허용됩니다.**

추가 읽기: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
