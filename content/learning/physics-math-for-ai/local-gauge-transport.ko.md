## Base 기하를 유지한 채 국소 Frame을 바꿀 수 있습니다

전역 대칭은 영역 전체에 같은 변환을 적용하지만 국소 gauge는 위치마다 다른 feature frame을 택합니다. 실험의 삼각형 위치는 고정이며 각 hᵢ는 두 성분이고 gᵢ∈SO2입니다. hᵢ'=gᵢhᵢ라는 성분 관례를 명시합니다. 역을 쓰는 passive 관례도 가능하지만 일관되게 적용해야 합니다.

frame을 독립적으로 바꾸면 이웃 성분을 바로 더할 수 없습니다. Uᵢⱼ는 j의 성분을 i의 성분 공간으로 옮깁니다. 운반의 일관성을 요구하면 다음과 같습니다.

$$
U_{ij}'=g_iU_{ij}g_j^{-1},\qquad
U_{ij}'h_j'=g_i(U_{ij}h_j).
$$

입력 쪽의 역이 그 점의 frame 변화를 상쇄해 출력점 변화만 남깁니다. 운반 메시지가 gauge 공변일 수 있는 이유입니다.

## 메시지 손계산

h₁=(0.5,1),U₀₁=Rπ/2이면 m₀=(−1,0.5)입니다. g₀=Rπ/2,g₁=I로 바꾸고 link도 올바르게 바꾸면 m₀'=g₀m₀=(−0.5,−1)입니다. link를 고정하는 실수를 하면 메시지는 (−1,0.5)에 남아 오차 √2.5≈1.581입니다.

주황 선분은 바뀐 국소 성분을 공통 배율 1/4로 표시합니다. 삼각형 꼭짓점을 움직이는 실제 물리 변위가 아닙니다. 성분과 link 각은 변해도 적절한 norm scalar·loop trace는 유지될 수 있습니다.

## 폐곡선은 켤레 변환됩니다

link는 0←1,1←2,2←0을 따릅니다. L=U₀₁U₁₂U₂₀는 0번 점의 성분에 작용하며 중간 gauge 인자가 지워집니다.

$$
L'=g_0Lg_0^{-1},\qquad \operatorname{tr}(L')=\operatorname{tr}(L).
$$

SO2는 가환이라 이 예제의 loop 행렬 자체도 같습니다. 비가환 군에서는 일반적으로 켤레류가 유지될 뿐 모든 성분이 같은 것은 아닙니다. 원래 link 각 (π/2,π/2,0)이면 L=Rπ=−I, trace=−2입니다.

## 비가환 Loop에서 켤레류와 행렬의 동일함 구별하기

끝점 frame 인자의 상쇄에는 link의 가환성이 필요하지 않습니다. SO3 link를 가진 삼각형에서도 같은 곱셈으로 L'=g₀Lg₀⁻¹을 얻습니다. trace와 고유값은 같지만 loop 행렬과 회전축의 성분은 일반적으로 달라집니다. Wilson trace는 gauge 불변이지만 connection 전체를 완전히 기술하는 특징은 아닙니다.

코드는 정확한 quarter-turn 행렬을 사용합니다. 첫 link에는 z 회전, 둘째에는 x 회전, 셋째에는 항등을 둡니다. 서로 다른 국소 frame 회전 후 loop의 성분은 바뀌지만 trace는 계속 0입니다. 별도의 비가환 확장 예제이며 기존 제어는 명시한 SO2 삼각형을 유지합니다.

| 대상 | Gauge 변환 | 불변 결론 |
|---|---|---|
| Feature hᵢ | gᵢhᵢ | 직교 frame에서의 노름 |
| 메시지 Uᵢⱼhⱼ | gᵢUᵢⱼhⱼ | 받는 점에서의 공변성 |
| 시작점을 정한 Loop L | g₀Lg₀⁻¹ | 켤레류와 trace |

연결 graph의 connection에서는 spanning tree를 고르고 root에서 frame을 전파해 tree link를 모두 항등으로 만들 수 있습니다. 남은 간선은 root에서의 공통 켤레 변환을 제외한 loop 정보를 담습니다. 따라서 비영 link 좌표만으로 holonomy를 증명할 수 없지만, 비자명한 loop 곱은 frame 변경만으로 모두 없앨 수 없습니다. 이 graph 명제가 연속 곡률이나 mesh의 표본화 정리를 증명하는 것은 아닙니다.

## Python으로 계산 재현하기

```python
I=[[1,0,0],[0,1,0],[0,0,1]]
X=[[1,0,0],[0,0,-1],[0,1,0]]
Y=[[0,0,1],[0,1,0],[-1,0,0]]
Z=[[0,-1,0],[1,0,0],[0,0,1]]
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def tr(A): return sum(A[i][i] for i in range(3))
def transpose(A): return list(map(list,zip(*A)))
links=[Z,X,I]; gauges=[Y,Z,X]
changed=[mm(mm(gauges[i],links[i]),transpose(gauges[(i+1)%3])) for i in range(3)]
loop=mm(mm(links[0],links[1]),links[2]); new=mm(mm(changed[0],changed[1]),changed[2])
assert new==mm(mm(gauges[0],loop),transpose(gauges[0]))
assert loop!=new and tr(loop)==tr(new)==0
print('loop:',loop); print('changed:',new); print('traces:',tr(loop),tr(new))
```

## 직접 풀기와 풀이

기준 link 각을 0으로 하고 국소 각을 (π/2,π,−π/2)로 두세요. 변환된 link 각은 비영이어도 loop는 항등이고 trace 2입니다. 단순 frame 변화이므로 비영 link 좌표만으로 곡률을 주장할 수 없습니다. trace와 메시지 공변 오차를 확인하세요.

## 기하와 AI의 연결

굽은 표면의 이웃 접특징에는 호환되는 국소 frame과 운반이 필요합니다. gauge 등변 구성은 이 의존성을 정리합니다. 이산 connection은 실제 holonomy를 가질 수 있고 gauge 변화는 그 표현을 바꿉니다. 세 점 SO2 계산은 이 법칙의 도입이며 비가환 gauge field·연속 곡률 tensor·네트워크 전체를 구현한 것은 아닙니다.

추가 읽기: [Cohen et al., Gauge Equivariant Convolutional Networks](https://proceedings.mlr.press/v97/cohen19d.html).
