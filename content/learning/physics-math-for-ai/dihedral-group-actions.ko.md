## 군은 가역 변환의 합성 규칙입니다

정사각형에는 네 회전과 네 반사, 총 여덟 강체 대칭이 있습니다. 여기서는 이 군을 D4라 부릅니다. 같은 군을 D8이라고 쓰는 책도 있으므로 표기를 확인해야 합니다. 닫힘·결합법칙·항등원·역원은 예쁜 그림 목록이 아니라 변환을 합성하는 규칙입니다.

R은 반시계 π/2 회전, S는 수평축 반사입니다. 모든 원소는 RᵏSᵇ로 유일하게 쓰며 k=0,1,2,3, b=0,1입니다. 열벡터에 행렬을 곱하므로 오른쪽 변환부터 작용합니다.

$$
R=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad
S=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
R^4=S^2=I,\quad SRS=R^{-1}.
$$

반사가 회전 방향을 뒤집기 때문에 일반적으로 교환하지 않습니다. 쌍 표기의 곱은 (k,b)(l,c)=(k+(−1)ᵇl mod 4,b+c mod 2)입니다. 수치 검사는 이 규칙과 실제 행렬 곱을 별도로 비교합니다.

## 군 작용은 무엇을 변환할지 정합니다

왼쪽 작용은 e·x=x, (gh)·x=g·(h·x)를 만족합니다. 여기서는 물리 평면에 행렬로 작용합니다. 파란 정사각형은 기하 기준이고 파란 점들은 선택한 원본 점의 서로 다른 상 전체입니다. 주황은 선택한 군 원소의 상입니다.

같은 추상 군도 pixel·벡터·집합·함수 공간에 다르게 작용할 수 있습니다. 작용 선택도 모형의 일부입니다. 군 원소가 여덟 개라고 모든 점의 상이 여덟 개일 필요는 없습니다.

## 궤도와 안정자는 다른 것을 셉니다

궤도는 도달 가능한 상의 집합이고 안정자는 원래 점을 고정하는 부분군입니다. 유한 군에서는 다음과 같습니다.

$$
|Gx|=\frac{|G|}{|G_x|},\qquad
G_x=\{g:g\cdot x=x\}.
$$

같은 상을 만드는 군 원소들은 안정자의 coset을 이루므로 상마다 같은 개수의 원소가 대응합니다. 꼭짓점은 네 상을 각각 두 원소가 만들고, 중심은 여덟 원소가 같은 점을 만듭니다. (0.7,0.3) 같은 일반 점의 안정자는 항등원뿐이므로 상이 여덟 개입니다.

## Burnside 보조정리로 구성의 동치류 세기

![정사각형 꼭짓점 색칠의 서로 다른 여섯 대칭 궤도와 평균이 6이 되는 여덟 군 원소의 고정 색칠 수를 비교합니다.](/assets/learning/symmetry-burnside-ko.svg)

군은 점 하나뿐 아니라 구성 전체에도 작용합니다. 정사각형 꼭짓점 네 개를 이름 있는 두 색으로 칠하면 배치는 열여섯 개지만 회전·반사가 일부를 동일시합니다. gx=x인 쌍 (g,x)을 두 방법으로 세세요. g별 합은 고정 구성의 수이고, 궤도별 합은 궤도 크기와 안정자 크기의 곱인 |G|입니다. 따라서 다음을 얻습니다.

$$
|X/G|=\frac1{|G|}\sum_{g\in G}|\operatorname{Fix}(g)|.
$$

꼭짓점 순열에 cycle이 c개이면 고정된 색칠은 각 cycle에서 같은 색이어야 하므로 2ᶜ개입니다. 표의 합은 48이므로 궤도는 여섯 개입니다. 주황 꼭짓점 두 개가 인접한 배치와 마주 보는 배치는 색 개수는 같지만 궤도가 다릅니다. 색 개수만으로는 완전한 불변량이 아닙니다. 정확 계산 코드는 궤도 대표를 따로 열거해 평균과 비교합니다.

| 대칭 종류 | 원소 수 | 각 원소가 고정하는 색칠 수 |
|---|---:|---:|
| 항등 | 1 | 16 |
| Quarter-turn | 2 | 2 |
| Half-turn | 1 | 4 |
| 꼭짓점 둘을 고정하는 반사 | 2 | 8 |
| 꼭짓점을 고정하지 않는 반사 | 2 | 4 |

별도의 구성 작용 예제이며 기존 실습의 제어는 점 작용을 유지합니다. 데이터 증강이 덮는 범위는 실제 변환법칙의 궤도입니다. 서로 다른 구성 궤도까지 합치면 과제 정보를 잃을 수 있습니다.

## Python으로 계산 재현하기

```python
from itertools import product
vertices = [(1,1),(-1,1),(-1,-1),(1,-1)]
permutations = []
for reflect in [False, True]:
    for k in range(4):
        permutation = []
        for x,y in vertices:
            if reflect: y = -y
            for _ in range(k): x,y = -y,x
            permutation.append(vertices.index((x,y)))
        permutations.append(permutation)
colorings = list(product([0,1], repeat=4))
def act(p, c):
    out = [0]*4
    for i,j in enumerate(p): out[j] = c[i]
    return tuple(out)
fixed = [sum(act(p,c)==c for c in colorings) for p in permutations]
representatives = {min(act(p,c) for p in permutations) for c in colorings}
assert sum(fixed) == 8*len(representatives) == 48
print(fixed, 'orbits:', len(representatives))
for r in sorted(representatives): print(r)
```

## 직접 풀기와 풀이

변 중앙 (1,0)을 선택하세요. 항등원과 수평축 반사가 고정하므로 안정자 2, 궤도 4를 예상할 수 있습니다. 주황 점 하나만 보지 말고 표 전체로 확인하세요. 이어 중심으로 바꿔 안정자가 클수록 궤도가 작아지는 이유를 설명하세요.

## 학습과의 연결

가중치 공유와 데이터 증강은 작용과 궤도에 의존합니다. 대칭으로 묶은 표본이 같은 예측 규칙을 공유하려면 그 변환이 과제에 적합해야 합니다. 반사는 물체 종류는 보존해도 handedness label은 바꿀 수 있습니다. 무엇이 변하고 출력은 어떻게 변해야 하는지, 제안한 변환들이 실제 합성에 닫힌 군인지부터 정하세요.

추가 읽기: [Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/books.html).
