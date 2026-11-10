## 두 Index는 다른 표현을 만듭니다

벡터는 v'=gv로 변하지만 rank-two tensor는 두 index 모두에 작용을 받습니다. 정규직교 Euclidean 좌표의 이 예제는 active 법칙 T'=gTgᵀ를 씁니다. 그러면 (gv)ᵀT'(gv)=vᵀTv입니다. passive 좌표 변화는 다른 관례이므로 역변환과 index 종류를 추적하지 않고 섞으면 안 됩니다.

조작값은 임의의 실수 대칭 T=[[p,q],[q,r]]입니다. 반드시 PSD가 아니므로 모든 설정을 covariance라 부르지 않습니다. 이차형식은 px²+2qxy+ry²라 비대각 성분은 두 번 기여합니다.

## Scalar와 Traceless 분해

모든 2×2 대칭 tensor는 유일하게 다음과 같이 나뉩니다.

$$
T=sI+\begin{pmatrix}a&b\\b&-a\end{pmatrix},\qquad
s=\frac{p+r}{2},\quad a=\frac{p-r}{2},\quad b=q.
$$

trace 성분 s는 직교 켤레 변환에 불변입니다. RθTRθᵀ를 직접 곱하면 traceless 쌍에 두 배 각도가 작용합니다.

$$
\begin{pmatrix}a'\\b'\end{pmatrix}
=\begin{pmatrix}\cos2\theta&-\sin2\theta\\
\sin2\theta&\cos2\theta\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}.
$$

수평 반사는 먼저 b를 −b로 바꾼 뒤 회전합니다. 실제 tensor 곱과 예측 계수 법칙을 비교하고 작은 차이는 부동소수점 오차로 보고합니다. 변환법칙을 맞추려 결과를 임의로 고치지 않습니다.

## 계수 그림 읽기

파랑·주황 벡터는 물리 위치가 아니라 (a,b) 계수 평면에 있습니다. T=diag(2,0)이면 s=1,(a,b)=(1,0)입니다. 물리 π/4 회전 후 [[1,1],[1,1]], 계수는 (0,1)로 90도 돌아갑니다. 물리 π/2 회전 후 diag(0,2), 계수는 (−1,0)입니다.

trace·determinant·고유값은 유지됩니다. 고유값 s±√(a²+b²)로 PSD 조건은 s≥√(a²+b²)입니다. T=I이면 traceless 성분이 0이라 어떤 회전에도 같습니다.

## 군에 따라 불가약성이 달라집니다

연속 SO2에서는 비영 weight의 두 배 각도 평면이 실수 불가약입니다. 물리 π/2를 생성자로 하는 C4에 제한하면 이 평면의 생성자는 −I뿐입니다. 모든 직선이 불변이므로 alternating 1차원 character 두 복사본으로 나뉩니다. 앞 단원의 한 배 각도 C4 불가약 평면과 다릅니다.

## 직접 풀기와 풀이

T=[[1,1],[1,−1]],θ=0에서 반사를 켜면 scalar 0, 계수 (1,1)→(1,−1)이고 trace 0·determinant −2는 같습니다. 고유값 ±√2라 covariance가 될 수 없는 이유를 설명하세요. 이어 항등 tensor를 택해 모든 각도에서 불변인지 확인하세요.

## 과학 AI와의 연결

Covariance·응력·확산 tensor·고차 신경 특징에는 index에 맞는 변환법칙이 필요합니다. 모든 channel을 무관한 scalar로 다루면 이 법칙을 잃습니다. tensor 차수·대칭성·metric 관례·양성 조건을 정한 뒤 등변 층을 설계하거나 tensor field를 해석하세요.

추가 읽기: [Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/books.html).
