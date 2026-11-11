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

## 두 Index의 곱에서 Scalar·Vector·고차 Block으로

전체 tensor product는 대칭 tensor만 포함하지 않습니다. 3차원의 모든 실수 행렬은 scalar 항등·반대칭·대칭 traceless 부분으로 유일하게 나뉩니다.

$$
T=\frac{\operatorname{tr}T}{3}I+\frac{T-T^T}{2}
+\left(\frac{T+T^T}{2}-\frac{\operatorname{tr}T}{3}I\right).
$$

Frobenius 직교인 세 block의 차원은 1,3,5입니다. 직교 g의 켤레 작용은 각 block을 보존합니다. 반대칭 부분을 A=hat(a)로 쓰면 SO3에서 a는 벡터로 변하지만 O3에서는 a'=det(g)ga인 axial vector입니다. 이 추가 determinant 때문에 벡터·tensor channel을 섞을 때 반사를 명시적으로 처리해야 합니다. proper 회전의 익숙한 3⊗3 분해이며 반사의 parity는 따로 정한 것입니다.

5차원 block은 SO3에서 불가약입니다. 비영 불변 부분공간의 tensor 하나를 proper 회전으로 대각화하세요. 부호 있는 순열 켤레 행렬들의 차로 비영 대각 차이를 얻고, 축 순열로 traceless 대각 공간 전체를 생성합니다. 대각 차이를 45도 회전하면 비대각 대칭 기저 tensor가 됩니다. 축을 순열하면 비대각 기저 세 개를 모두 얻으므로 불변 부분공간은 다섯 차원 전체를 포함합니다.

2차원에서는 대응하는 반대칭 직선이 J로 생성됩니다. 회전에서는 trivial이지만 반사에서 부호가 바뀝니다. 대칭 trace 직선은 둘 다에서 불변입니다. 실습은 대칭 2×2 tensor의 3차원 공간을 다루며, 아래 코드는 별도의 3×3 확장입니다.

| 3차원 Block | 차원 | SO3 변환 |
|---|---:|---|
| Scalar 항등 | 1 | Scalar |
| 반대칭 | 3 | Axial vector, proper 회전에서는 보통 벡터와 같음 |
| 대칭 Traceless | 5 | Rank-two 불가약 Block |

정확분수로 복원·직교성과 axial 반사 법칙을 확인하세요. Covariance tensor의 양성은 별도 제약이며 표현 분해만으로 따라오지 않습니다.

## Python으로 계산 재현하기

```python
from fractions import Fraction as F
T=[[F(v) for v in row] for row in [[1,2,0],[0,-1,3],[4,0,2]]]
scalar=sum(T[i][i] for i in range(3))/3
C=[[scalar*int(i==j) for j in range(3)] for i in range(3)]
A=[[(T[i][j]-T[j][i])/2 for j in range(3)] for i in range(3)]
S=[[(T[i][j]+T[j][i])/2-C[i][j] for j in range(3)] for i in range(3)]
assert [[C[i][j]+A[i][j]+S[i][j] for j in range(3)] for i in range(3)]==T
def inner(X,Y): return sum(X[i][j]*Y[i][j] for i in range(3) for j in range(3))
assert inner(C,A)==inner(C,S)==inner(A,S)==0
axial=[A[2][1],A[0][2],A[1][0]]
# Reflect z: g=diag(1,1,-1), det(g)=-1.
signs=[1,1,-1]; reflected=[[signs[i]*A[i][j]*signs[j] for j in range(3)] for i in range(3)]
axial_after=[reflected[2][1],reflected[0][2],reflected[1][0]]
assert axial_after==[-signs[i]*axial[i] for i in range(3)]
print('block squared norms:',inner(C,C),inner(A,A),inner(S,S))
print('axial before/after:',axial,axial_after)
```

## 직접 풀기와 풀이

T=[[1,1],[1,−1]],θ=0에서 반사를 켜면 scalar 0, 계수 (1,1)→(1,−1)이고 trace 0·determinant −2는 같습니다. 고유값 ±√2라 covariance가 될 수 없는 이유를 설명하세요. 이어 항등 tensor를 택해 모든 각도에서 불변인지 확인하세요.

## 과학 AI와의 연결

Covariance·응력·확산 tensor·고차 신경 특징에는 index에 맞는 변환법칙이 필요합니다. 모든 channel을 무관한 scalar로 다루면 이 법칙을 잃습니다. tensor 차수·대칭성·metric 관례·양성 조건을 정한 뒤 등변 층을 설계하거나 tensor field를 해석하세요.

추가 읽기: [Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/books.html).
