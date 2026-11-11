## 강체 Pose는 방향과 위치를 묶습니다

SO3는 RᵀR=I, detR=1인 3차원 proper 회전 군입니다. 직교성은 길이와 각도를 보존하고 행렬식 1은 반사를 제외합니다. 아홉 행렬 성분 안의 굽은 3차원 다양체이므로 성분을 모두 독립적인 무제약 변수로 학습하면 회전 조건을 잃을 수 있습니다.

SE3는 병진을 추가합니다. 점 p에 Rp+t로 작용하며 homogeneous 행렬로 표현합니다.

$$
T=\begin{pmatrix}R&t\\0&1\end{pmatrix},\qquad
T\begin{pmatrix}p\\1\end{pmatrix}
=\begin{pmatrix}Rp+t\\1\end{pmatrix}.
$$

방향 벡터는 마지막 성분이 0이므로 병진 영향을 받지 않습니다. 그림의 기준·변환 벡터는 각 frame 원점에서 시작하며 주황 원점도 병진과 함께 옮깁니다.

## 이동을 무조건 더하지 말고 합성 유도하기

T₂ 다음 T₁을 적용하면 다음과 같습니다.

$$
R_1(R_2p+t_2)+t_1
=(R_1R_2)p+(R_1t_2+t_1).
$$

따라서 곱은 (R₁R₂,R₁t₂+t₁)이고 오른쪽 변환부터 작용합니다. 두 번째 병진을 첫 회전으로 표현해야 하므로 t₁+t₂만 더하면 frame을 잘못 섞을 수 있습니다. 역은 p'=Rp+t를 풀어 (Rᵀ,−Rᵀt)입니다.

실험은 A=(Rz(α),t), B=(Rx(β),0)을 사용해 순서 효과를 쉽게 보입니다. 다음 단원에서는 임의 회전벡터 생성자도 씁니다. 여기서도 행렬 곱과 점 작용은 실제 3차원입니다.

## Quarter-turn 손계산

α=β=π/2, t=(1,0,0), p=(1,0,0)으로 둡니다. AB에서는 Rx가 p를 유지하고 Rz가 (0,1,0)으로 보낸 뒤 병진해 (1,1,0)입니다. BA에서는 A가 (1,1,0)을 만든 뒤 Rx가 (1,0,1)로 보냅니다. 차이는 √2이며 수치 정밀도 문제가 아닙니다.

아래 그림은 선택한 변환점의 xy·xz 투영이며 두 세로 좌표 의미가 다릅니다. 전체 xyz·행렬 행·determinant·직교 오차는 표에서 읽으세요. 위 비등각 그림의 각도와 길이는 실제 값이 아닙니다.

## Adjoint로 Twist의 Frame 바꾸기

SE3는 반직접곱입니다. 병진을 더하기 전에 회전이 병진에 작용합니다. 속도 생성자의 frame을 바꿀 때도 같은 결합이 나타납니다. T=(R,t)로 homogeneous 생성자를 켤레 변환하세요.

$$
T\begin{pmatrix}\widehat\omega&v\\0&0\end{pmatrix}T^{-1}
=\begin{pmatrix}\widehat{R\omega}&Rv+t\times R\omega\\0&0\end{pmatrix}.
$$

각 성분을 먼저 쓰면 adjoint는 Adₜ(ω,v)=(Rω,t×Rω+Rv)입니다. 병진 항은 lever-arm 속도입니다. 원점이 다른데 이 항 없이 두 3차원 벡터를 회전만 시키면 틀립니다. 미분 가능한 pose에서 spatial·body twist의 행렬 정의가 각각 ṪT⁻¹, T⁻¹Ṫ이면 Vₛ=AdₜVᵦ입니다.

R=I,t=(1,2,0),ω=(0,0,1),v=0이면 선형 성분은 (2,−1,0)입니다. 원래 점 집합을 이동한 것이 아니라 같은 무한소 강체 운동을 다른 원점 기준으로 표현한 것입니다. 정확 homogeneous 행렬 코드는 부호와 곱 순서를 검사합니다.

| 원점 차이 t | 각 성분 | 변환된 선형 성분 |
|---|---|---|
| (0,0,0) | (0,0,1) | (0,0,0) |
| (1,0,0) | (0,0,1) | (0,-1,0) |
| (1,2,0) | (0,0,1) | (2,-1,0) |

대응하는 Lie bracket은 (ω₁×ω₂,ω₁×v₂−ω₂×v₁)입니다. bracket과 adjoint 모두 frame을 반영해 pose 갱신을 유도하는 도구이며, 유한 회전이 교환한다고 가정해 회전벡터를 더하는 근거는 아닙니다.

## Python으로 계산 재현하기

```python
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
Xi=[[0,-1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]
for t in [(0,0,0),(1,0,0),(1,2,0)]:
    T=[[int(i==j) for j in range(4)] for i in range(4)]
    inv=[r[:] for r in T]
    for i in range(3): T[i][3]=t[i]; inv[i][3]=-t[i]
    out=mm(mm(T,Xi),inv)
    linear=[out[i][3] for i in range(3)]
    assert linear == [t[1],-t[0],0]
    print(t,linear)
```

## 직접 풀기와 풀이

두 각을 0, t=(1,2,−1)로 두면 기준점은 (2,2,−1)입니다. B가 항등이라 순서도 같습니다. 비영 각으로 돌아가 y 병진을 바꾸면 역순에서 병진도 회전하는지 확인하세요. 마지막으로 계산한 역을 적용해 기준점으로 돌아오는지 검산하세요.

## Embodied AI와의 연결

로봇 pose·카메라 extrinsic·공간 신경 특징에는 active/passive와 frame 관례가 필요합니다. 올바른 회전 행렬만으로 world→camera인지 반대인지는 알 수 없습니다. 예측을 합성하거나 pose 오차를 비교하기 전에 출발·도착 frame, 곱 순서, 점인지 방향인지부터 표시하세요.

추가 읽기: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
