## 1. 텐서 네트워크는 진폭을 만드는 규칙을 저장한다

n개 qubit의 상태는 이진 문자열마다 복소 진폭 하나를 가진다. 세 qubit이면000부터111까지 여덟 문자열이며, 일반적으로는2ⁿ개다. Hamiltonian을 국소항의 합으로 짧게 적었다고 상태 벡터까지 자동으로 작아지지는 않는다. 상태에 어떤 구조가 있는지와 그 구조를 어떤 좌표로 표현할지를 따로 설명해야 한다.

Matrix product state, MPS는 작은 텐서들의 사슬을 수축하여 진폭을 만든다. 물리 비트가 각 위치의 행렬 하나를 선택한다고 생각하자. 선택한 행렬들을 곱한 결과가 전체 문자열의 진폭이다. 행렬 사이의 내부 색인은 합산하며, 추가로 측정하는 입자가 아니다. 그 내부 색인의 크기를 bond dimension이라 부른다.

$$
\psi_{s_0\cdots s_{n-1}}=\sum_{a_1,\ldots,a_{n-1}}A_0^{s_0}[1,a_1]A_1^{s_1}[a_1,a_2]\cdots A_{n-1}^{s_{n-1}}[a_{n-1},1],\qquad D_0=D_n=1.
$$

여기서는 열린 경계를 사용한다. 맨 왼쪽과 맨 오른쪽의 가상 색인은 크기1이므로 최종 곱은 스칼라다. 주기 경계의 행렬 곱에 trace를 취하는 표현과 혼동하지 않는다. 전체 배열에서 site0의 비트가 최상위 비트다. 이번 장은 표현·정확 수축·환경·가상 좌표 변경을 배우며, 최적 상태 탐색이나 Schmidt 성분을 버리는 작업은 구별한다.

## 2. 배열을 채우기 전에 다리의 역할을 그린다

각 site에는 물리 다리 s와 왼쪽 가상 다리 a, 오른쪽 가상 다리 b가 있다. 내부 선은 같은 차원의 두 다리를 연결하며 해당 색인을 합하라는 뜻이다. 바깥 물리 다리는 비트를 정하기 전까지 열려 있다. 그림의 선과 점이 계산이 되려면 색인의 역할·크기·순서를 먼저 고정해야 한다.

$$
A_i^s\in\mathbb C^{D_i\times D_{i+1}},\qquad P=\sum_{i=0}^{n-1}d_iD_iD_{i+1},\qquad P_{\mathrm{qubit},D}=4D+2(n-2)D^2\quad(n\ge2).
$$

dᵢ는 국소 물리 차원이며 계산기에서는2다. P는 gauge 중복을 제거하기 전 활성 복소 항의 개수다. 서로 다른 물리 상태들의 공간 차원과 같지 않다. n=3이고 두 bond가2이면 각 site의 항은4,8,4개로 총16개다. 전체 진폭은8개뿐이므로 이 작은 예제는 오히려 항이 많다. 텐서 네트워크라는 이름만으로 압축을 보장하지 않는다.

| 항목 | 뜻 | 세 qubit 예제 |
| --- | --- | --- |
| 물리 다리 | 국소 기저 선택 | s₀,s₁,s₂의 크기는 각각2 |
| 가상 다리 | 합산하는 표현 색인 | D₁=D₂=2 |
| Site 텐서 | s가 선택하는 두 행렬 | 크기1×2,2×2,2×1 |
| 출력 벡터 | 모든 물리 진폭 | 복소 항8개 |

출력은 A[site][physical][left][right] 순서다. 입력 키 t1Imag203은 site1·left2·physical0·right3을 뜻하므로 키를 읽는 순서와 출력 배열 순서를 같다고 추측하면 안 된다. 차원이 우연히 일치하더라도 transpose가 다른 상태를 만들 수 있다. 색인 convention은 구현의 부수 설명이 아니라 실험의 일부다.

## 3. 곱 상태에서 GHZ 경로 그림으로

모든 bond 차원이1이면 선택하는 행렬은 스칼라다. 전체 진폭이 국소 진폭의 곱으로 분리되므로 곱 상태를 나타낸다. 얽힌 상태를 표현하려면 적어도 어떤 cut에서는 독립적인 가상 경로가 둘 이상 필요하다. 그러나 경로를 많이 할당하는 것만으로 얽힘이 생기지는 않는다.

다음 세 site의 텐서로 구체적인 얽힌 상태를 만들자. 모든 항은0 또는1이며 물리 레이블이 위첨자의 행렬을 선택한다. 첫 텐서는 행벡터이고 마지막 텐서는 열벡터다. 가운데 텐서의 projector가 가상 레이블을 유지하거나 그 경로를 없앤다.

$$
A_0^0=(1,0),\quad A_0^1=(0,1),\quad A_1^0=\begin{pmatrix}1&0\\0&0\end{pmatrix},\quad A_1^1=\begin{pmatrix}0&0\\0&1\end{pmatrix},\quad A_2^0=\binom10,\quad A_2^1=\binom01.
$$

000에서는 가상 레이블0의 경로가 세 site를 통과해1을 준다. 111에서는 레이블1의 경로가1을 준다. 섞인 문자열은 중간 projector나 경계 벡터의 불일치로 사라진다. 원시 벡터는 |000〉+|111〉이고 노름 제곱은2다. 정규화하면 GHZ₃ 상태다. 텐서의 원시 정의와 물리 상태의 정규화를 별도로 기록하자.

| 비트 문자열 | 살아남는 가상 경로 | 원시 진폭 |
| --- | --- | --- |
| 000 | 0→0 | 1 |
| 111 | 1→1 | 1 |
| 001 | 완전한 경로 없음 | 0 |
| 010 | 완전한 경로 없음 | 0 |

101도 직접 따라가 보자. 첫 물리1은 가상 레이블1을 선택하지만 가운데 물리0의 행렬은 레이블0만 통과시킨다. 따라서 곱은0이다. 그림을 움직이기 전에 이 한 경로를 손으로 추적하면, 애니메이션이 무엇을 합산해야 하는지 예측할 수 있다. 한 경로의 실패와 모든 경로 합의 상쇄도 서로 다른 이유라는 점을 기억하자.

## 4. 사슬을 잘라 두 계수 사상을 만든다

k개 site 뒤에서 자르고 왼쪽 물리 문자열을 x, 오른쪽 문자열을 y로 묶자. 왼쪽 수축 L은 물리 prefix와 절단의 가상 색인을 연결한다. 오른쪽 수축 B는 가상 색인과 물리 suffix를 연결한다. 두 행렬의 곱은 그 물리적 양분할에서 전체 상태의 계수 행렬이다.

$$
C_{xy}=\sum_aL_{xa}B_{ay},\qquad C=LB,\qquad\operatorname{rank}C\le D_k,\qquad N=\sum_{x,y}|C_{xy}|^2.
$$

C의 각 열은 L의 열공간에 있으므로 rank가 Dₖ를 넘을 수 없다. 비영 순수 상태라면 계수 rank가 Schmidt rank다. 따라서 가상 차원은 실제 사용한 rank의 상한이다. 정규화는 모든 singular value에 공통 인자를 곱하며 rank를 바꾸지 않는다. 이 사실은 다음 장의 Schmidt 분해와 연결된다.

GHZ₃의 첫 cut에서 C는 행0·열00과 행1·열11에만1을 가진다. 두 행은 독립이므로 rank2이며 bond1로는 이 상태를 정확하게 표현할 수 없다. 반면 bond4에 저장한 상태의 rank는2나1일 수도 있다. 표현의 용량, 실제 사용한 rank, 양쪽 물리 공간의 크기를 각각 구별해야 잘못된 얽힘 해석을 피한다.

## 5. 환경은 convention을 가진 Gram 행렬이다

각 cut에서 양쪽 물리 문자열을 합산하여 left environment와 right environment를 만든다. 둘 다 가상 공간의 양의 준정부호 Gram 행렬이다. 노름을 닫는 데 필요한 블록 정보를 요약하지만, 원소 자체는 선택한 표현에 의존한다. 물리 입자의 정규화된 reduced density라고 부르지 않는다.

$$
E_{ab}=\sum_x\overline{L_{xa}}L_{xb},\qquad R_{ab}=\sum_y\overline{B_{ay}}B_{by},\qquad N=\sum_{a,b}E_{ab}R_{ab}=\operatorname{Tr}(ER^{\mathsf T}).
$$

오른쪽 정의에서 Rₐᵦ는 conj(Bₐᵧ)Bᵦᵧ를 합한다. C의 절댓값 제곱을 전개하면 같은 색인 순서의 EₐᵦRₐᵦ가 남는다. 따라서 Tr(ERᵀ)이며 일반적으로 Tr(ER)가 아니다. 실수 대각 예제만 쓰면 이 transpose를 빠뜨려도 통과할 수 있다. 두 행렬이 양의 준정부호라는 사실은 임의의 색인 연결을 정당화하지 않는다.

복소 위상 반례로 L=(1,i), B=(1,i)ᵀ를 사용하자. 두 site·bond2 표현에서 물리00만 활성화하면 유일한 진폭이1+i²=0이다. E와 R은 모두 [[1,i],[-i,1]]이다. 같은 색인 원소를 곱해 합하면0이지만 Tr(ER)는4다. 원시 텐서는 유효해도 정규화된 물리 상태는 정의되지 않는다. 잘못된 closure는 상쇄를 숨기고 존재하지 않는 양의 노름을 만들어낸다.

## 6. Transfer 수축은 부분합을 재사용한다

모든 가상 경로를 열거하는 방식은 기준 계산으로 명확하다. 환경은 앞에서 구한 부분합을 재사용하여 한 site씩 계산할 수 있다. 양 끝에서 스칼라1로 시작하고 왼쪽 또는 오른쪽 방향으로 진행한다. 어떤 물리 색인을 이미 합했는지 표시하면 재사용하는 정보의 의미가 분명해진다.

$$
E_{i+1}=\sum_s(A_i^s)^\dagger E_i A_i^s,\qquad R_i=\sum_s\overline{A_i^s}R_{i+1}(A_i^s)^{\mathsf T},\qquad E_0=R_n=(1).
$$

왼쪽 update에는 adjoint가 있고 오른쪽 update에는 R 정의에 맞는 conjugation과 transpose가 있다. 어느 cut에서 양쪽 환경을 닫아도 전체 진폭의 노름 N과 같아야 한다. 최종 스칼라 하나만 확인하는 대신 모든 cut을 검사하면 색인이나 켤레의 오류를 더 쉽게 찾는다. 두 실수가 마지막에서 우연히 상쇄되는 경우도 구별할 수 있다.

이 식을 그림으로 보면 ket 텐서 층과 그 복소 켤레 층이 겹쳐 있고 각 물리 다리가 국소적으로 닫힌다. 반면 prefix 진폭 수축은 물리 문자열을 열어 둔다. 같은 텐서를 써도 열린 다리와 저장할 배열의 크기가 다른 계산이다. 전체 진폭을 계속 보관하면서 계산했다면 환경만 계산하는 알고리즘의 비용을 그대로 주장할 수 없다.

## 7. 전체 진폭을 펼치지 않고 관측량을 삽입한다

특정 물리 다리의 identity 대신 관측 연산자를 넣는다. Bra와 ket의 두 물리 색인에 연산자의 행렬 원소를 곱하고 주변 환경을 닫은 뒤 전체 노름 제곱으로 나눈다. 분모는 반드시 양수여야 한다. 영 수축을 다른 상태로 바꾸거나 임의의 작은 수로 나누는 것은 정규화가 아니다.

$$
\langle O_i\rangle=\frac1N\sum_{a,a',b,b',s,t} E_i[a,a']\overline{A_i^s[a,b]}\,O_{st}A_i^t[a',b']R_{i+1}[b,b'],\qquad\langle Y\rangle=-2\operatorname{Im}\rho_{01}.
$$

Bra/ket 물리 색인을 열어 두면 국소 reduced density를 얻는다. 그 행과 열은 물리 qubit 기저를 가리키지만 E와 R의 행과 열은 가상 좌표를 가리킨다. 크기가 우연히 같아도 서로 바꿔 쓸 수 없다. 전체 진폭에서 직접 partial trace한 밀도와 환경 삽입 결과를 별도로 비교하는 이유다.

|0〉⊗(|0〉+i|1〉)의 노름 제곱은2다. 오른쪽 density는 [[1/2,−i/2],[i/2,1/2]]이므로 X=0,Y=+1,Z=0이고 purity1이다. 켤레를 잘못 넣으면 Y 부호가 바뀐다. 실수 GHZ 예제만으로는 이 오류를 드러낼 수 없으므로, 얽힌 예제 옆에 위상이 있는 곱 상태를 함께 둔다. 얽힘의 유무와 복소 위상 검산은 서로 다른 역할을 한다.

## 8. Gauge 변환은 가상 좌표를 바꾼다

공유 bond의 한쪽에 가역 행렬 G를 넣고 다른 쪽에 그 inverse를 넣는다. 모든 물리 레이블 조합에서 이웃 곱이 그대로이므로 전체 진폭도 모두 같다. G가 unitary일 필요도, 입력 상태가 정규화돼 있을 필요도 없다. 필요한 가정은 크기가 맞는 가역 행렬이라는 것이다.

$$
A_i^s\longmapsto A_i^sG,\qquad A_{i+1}^t\longmapsto G^{-1}A_{i+1}^t,\qquad (A_i^sG)(G^{-1}A_{i+1}^t)=A_i^sA_{i+1}^t.
$$

Gauge는 두 이웃 텐서를 함께 바꾸는 작업이다. 한 텐서만 G로 곱하면 일반적으로 물리 상태가 달라진다. 양쪽 모두 G를 곱해도 보통 틀리며, inverse가 올바른 위치에 있어야 상쇄된다. Singular 행렬에는 inverse가 없으므로 이 형태의 정확 gauge로 허용하지 않는다. 차원을 줄이는 projection과 가역 좌표 변환도 구별해야 한다.

이는 행렬 분해를 (UG)(G⁻¹V)로 다시 적는 표현의 중복과 닮았다. 물리적 부분계에 시간 진화를 적용한 것이 아니다. 따라서 전체 복소 진폭 벡터를 비교해야 한다. 노름이나 몇 개의 국소 관측량만 같다는 결과는 서로 다른 많은 물리 상태를 구별하지 못한다. Gauge 보존의 증거로 필요한 양을 처음부터 정확히 정하자.

## 9. Gauge가 텐서 좌표 길이를 바꾸는 예제

GHZ₃의 첫 bond에 scale2와 shear1/2를 적용한다. 순서는 diagonal scale times shear이며 아래 G와 inverse가 된다. 활성 텐서의 모든 복소 항의 절댓값 제곱 합을 K로 정의하자. K와 전체 물리 진폭의 노름 N은 서로 다른 다항식이다.

$$
G=\begin{pmatrix}2&1\\0&1\end{pmatrix},\quad G^{-1}=\begin{pmatrix}1/2&-1/2\\0&1\end{pmatrix},\qquad N'=N=2,\quad K=6,\quad K'=\frac{19}{2}.
$$

첫 site의 좌표 길이는2에서6으로 바뀐다. 가운데 site는2에서3/2가 되고 마지막은2를 유지하므로 K′=19/2다. 물리 노름 제곱은2이고 모든 진폭도 그대로다. 따라서 텐서 좌표의 Euclidean penalty는 같은 물리 상태를 표현하더라도 gauge에 따라 달라질 수 있다. 최적화의 변수에 거는 규제가 어떤 공간의 양인지 구별해야 한다.

| GHZ₃의 양 | Gauge 전 | Gauge 후 |
| --- | --- | --- |
| 물리 노름 제곱 N | 2 | 2 |
| 텐서 좌표 길이 K | 6 | 19/2 |
| 첫 cut의 계수 rank | 2 | 2 |
| 첫 left environment | I₂ | [[4,2],[2,2]] |

K의 증가는 얽힘 증가·에너지 변화·근사 개선의 증거가 아니다. 좌표 기하와 물리 상태의 기하가 다르다는 증거다. AI 연구에서 텐서 인자를 학습하거나 정규화 항을 설계할 때도, 규제하려는 대상이 좌표인지 표현된 함수나 상태인지 먼저 정해야 한다. 이 예제는 최적화 알고리즘의 수렴을 증명하지 않는다.

## 10. 환경의 covariance는 closure를 보존한다

변환한 cut에서 L은 LG, B는 G⁻¹B가 된다. 이를 Gram 정의에 대입하면 환경의 변환 법칙을 얻는다. 물리 상태가 같아도 두 환경은 일반적으로 달라진다. 좌표가 바뀌는데 두 metric을 고정된 숫자처럼 비교하면 잘못된 결론을 낼 수 있다.

$$
E'=G^\dagger EG,\qquad R'=\overline{G^{-1}}R(G^{-1})^{\mathsf T},\qquad\sum_{ab}E'_{ab}R'_{ab}=\sum_{ab}E_{ab}R_{ab}.
$$

위 실수 GHZ gauge에서는 E′=[[4,2],[2,2]], R′=[[1/2,−1/2],[−1/2,1]]이다. 같은 위치의 원소를 곱한 합은2로 원래 노름과 같다. 어느 환경도 identity일 필요는 없다. 각각은 좌표에 의존하는 블록 metric이며 독립적으로 보존되는 물리 관측량이 아니다.

복소 shear에서는 R′의 conjugation이 필수다. G가 실수인 예제만으로 켤레의 위치를 검증할 수 없다. 계산기는 inverse 잔차·두 covariance 행렬 잔차·전체 상태 차이를 모두 구한다. 이 잔차가0이라는 결과는 명시한 유한 대수의 확인이지, 아직 실행하지 않은 최적화 알고리즘이나 큰 계의 수렴을 보증하지 않는다.

## 11. Isometry가 정규형을 이해하게 한다

Left-isometric site는 왼쪽 가상 색인과 물리 색인을 묶어 행으로 볼 때 열들이 직교정규인 텐서다. Right-isometric site는 물리 색인과 오른쪽 가상 색인을 묶어 열로 볼 때 행들이 직교정규다. 두 identity의 크기가 다르므로 단순히 텐서가 정규화됐다는 한 표현으로 합치지 않는다.

$$
\sum_s(A_i^s)^\dagger A_i^s=I_{D_{i+1}}\quad\text{(left isometry)},\qquad\sum_s A_i^s(A_i^s)^\dagger=I_{D_i}\quad\text{(right isometry)}.
$$

원시 GHZ 텐서에서 첫째와 가운데 site는 left isometry지만 마지막 left Gram은 스칼라2다. 앞의 identity 환경이 전파되어 마지막 스칼라에서 노름 제곱2가 되는 것과 일치한다. 마지막 site의 두 행렬만√2로 나누면 전체 상태가 정규화되며 사슬 전체가 left-normalized가 된다. 이 해석적 변환은 현재 복소 유리수 입력 인터페이스와 구별한다.

한 site만 isometry라고 전체 노름이1인 것은 아니다. 열의 수가 결합한 왼쪽·물리 행 차원을 넘으면 left isometry 자체가 불가능하다. 중복 bond가 그런 상황을 만들 수도 있다. 정규형 진단은 실제 Gram과 잔차를 보고해야 하며, 어떤 차원도 가역 gauge만으로 whitening할 수 있다고 가정하지 않는다.

## 12. Canonicalization은 별도의 알고리즘이다

Left-isometric 텐서를 만들려면 site를 행렬로 reshape하고 rank를 고려한 QR 또는 SVD를 수행한다. 직교정규 인자는 새 site가 되고 나머지 인자는 다음 site로 흡수한다. 필요한 성분을 모두 유지한 정확 분해라면 물리 상태는 보존된다. 단순히 진단값을 계산하는 것과 실제 분해를 수행하는 것은 다른 작업이다.

$$
M_{(a,s),b}=A_i^s[a,b]=\sum_c Q_{(a,s),c}T_{cb},\qquad\sum_s(Q^s)^\dagger Q^s=I,\qquad A_{i+1}^t\longmapsto T A_{i+1}^t.
$$

열 rank가 부족하면 economy 분해에서 가상 차원을 줄일 수 있다. 이때 T는 직사각형일 수 있으므로 가역 정사각 gauge라고 부를 수 없다. 임의의 직교정규 completion을 유지하면 사용하지 않는 방향이 남을 수도 있다. 상태 보존·수치 안정성·차원 축소를 각각 확인해야 한다. 수학적으로 같은 상태라는 사실만으로 수치 conditioning까지 같아지는 것도 아니다.

현재 계산기는 canonicality를 측정하고 지정한 가역 scale/shear를 적용한다. QR·SVD·전체 sweep을 실행하지 않는다. 여기의 식은 뒤에 필요한 알고리즘과 가정을 설명한다. Gram 잔차가0이라 그림이 초록색으로 변하더라도, 그것은 진단이지 자동 canonicalization 구현의 증거가 아니다. 학습 화면의 상태 표시도 이 구별을 유지해야 한다.

## 13. 국소 요약과 전체 정보는 다르다

두 site의 Pauli를 transfer network에 삽입하면 pair 기대값을 얻는다. 여기에서 각각의 one-site 평균의 곱을 빼면 connected correlation이다. Pauli X/Y/Z는 무차원이며 실제 spin 성분에는ℏ/2 계수가 필요하다. 앞의 spin 장과 연결할 때 단위 convention을 유지하자.

$$
C_{ij}^{\alpha\beta}=\langle\sigma_i^\alpha\sigma_j^\beta\rangle-\langle\sigma_i^\alpha\rangle\langle\sigma_j^\beta\rangle,\qquad C_{01}^{ZZ}(\mathrm{GHZ}_3)=1.
$$

정규화한 GHZ₃의 모든 one-site Z 평균은0이고 모든 pair ZZ 평균은1이다. 따라서 connected ZZ도1이다. Bond1 곱 상태는 이 비영 connected correlation을 만들 수 없다. 반대로 곱 상태에 bond4를 할당해도 연결 상관이 생기거나 Schmidt rank가4가 되지 않는다. 표현 용량을 실제 사용한 상관으로 해석하지 않는 연습이다.

국소와 pair 요약만으로 전체 상태가 결정되지는 않는다. |000〉와|111〉의 같은 비율 비간섭 혼합은 GHZ₃와 모든 one-site 상태·two-site Pauli 평균이 같지만, 세 site XXX 평균은0으로 GHZ의1과 다르다. 이 혼합은 순수 MPS 입력 밖의 해석적 비교다. 국소 purity나 큰 ZZ만 보고 전역 coherence를 추론해서는 안 된다. 어떤 관측을 추가해야 차이를 드러내는지 묻는 것이 다음 연구 질문이 된다.

## 14. Bond 용량·수축 비용·압축 주장을 분리한다

모든 cut에서 계수 rank는 가상 차원을 넘지 않는다. 양쪽 물리 공간 차원도 rank의 상한이다. 이 조건들은 정확 표현을 제약하지만 singular value의 크기나 bond를 줄일 때의 오차를 아직 알려주지 않는다. Rank가 같다는 것과 절단하기 쉬운 상태라는 것도 다른 주장이다.

$$
\mathrm{GHZ}_3:\ D_1=D_2=2,\quad\operatorname{rank}C_1=\operatorname{rank}C_2=2;\qquad |000\rangle:\ D_1=D_2=4,\quad\operatorname{rank}C_1=\operatorname{rank}C_2=1.
$$

세 qubit 곱 상태를 bond4에 저장하면 활성 항은8+32+8=48개다. 전체 벡터는8개이며 많은 가상 방향이 중복이어도 할당한 배열의 공간은 필요하다. 큰 사슬에서 D를 적당한 상수로 유지할 수 있다면 텐서 저장량은 site 수에 선형일 수 있다. 그러나 임의의 강하게 얽힌 상태에는 D 자체가 지수적으로 커질 수 있으므로 D에 대한 조건을 생략할 수 없다.

| 세 site의 표현 | Bond | 활성 복소 항 | 전체 진폭 |
| --- | --- | --- | --- |
| 곱 상태 | (1,1) | 6 | 8 |
| GHZ 상태 | (2,2) | 16 | 8 |
| 중복 곱 상태 | (4,4) | 48 | 8 |

행렬 곱 순서를 잘 잡은 norm 수축은 n,d,D에 대해 알려진 다항 비용을 가진다. 현재 기준 친화적인 구현은 선택한 반복문의 dense summand slot 수도 별도로 표시한다. 이 수는 실제 실행 시간·최소 연산 수·bit complexity가 아니다. 정확 유리수의 분자와 분모도 커질 수 있다. 모든 진폭을 화면에 출력하라는 요구에는 그 자체로 지수적인 출력 개수가 필요하다. 한 노름의 효율적인 수축과 전체 벡터 출력의 비용을 섞지 않는다.

## 연구 확장: transfer 스펙트럼·상관 길이·gauge 조건

### 정확히 정규화한 bond2 가족 만들기

각 site의 물리 bit와 다음 비음수 행렬·열린 경계를 택하자.

$$
0\le p\le1,\quad T=\begin{pmatrix}p&1-p\\1-p&p\end{pmatrix},\quad
 A^0=\begin{pmatrix}\sqrt p&\sqrt{1-p}\\0&0\end{pmatrix},\quad
 A^1=\begin{pmatrix}0&0\\\sqrt{1-p}&\sqrt p\end{pmatrix},\quad
 L=(1,1)/\sqrt2,\quad R=\frac{(1,1)^{\mathsf T}}{\sqrt p+\sqrt{1-p}}.
$$

각 A는 영이 아닌 행이 하나다. 물리 문자열을 곱하면 그 행 경로가 정해져 N site에 대해 다음과 같다.

$$
\psi_{s_1\cdots s_N}=L A^{s_1}\cdots A^{s_N}R
 =\frac1{\sqrt2}\prod_{j=1}^{N-1}\sqrt{T_{s_j,s_{j+1}}},\qquad
 |\psi_{s_1\cdots s_N}|^2=\frac12\prod_{j=1}^{N-1}T_{s_j,s_{j+1}},\quad\|\psi\|^2=1.
$$

마지막 bit부터 합하면 T의 행 합을 반복해 정규화를 증명한다. Z 기저 확률은 정류 두 상태 Markov chain과 같지만 양자 상태는 그 확률의 coherent 제곱근 중첩이며 대각 고전 혼합이 아니다. 다른 측정 기저는 둘을 구별할 수 있다.

### 가상 transfer map으로 물리 상관 설명하기

아래 규약의 오른쪽→왼쪽 transfer map을 정의한다. 고정 행렬 I·고정 쌍대 함수 ell·스펙트럼은 모두 두-by-두 곱으로 확인할 수 있다.

$$
\mathcal E(X)=\sum_s A^sX(A^s)^\dagger,\quad
 \mathcal E(I)=I,\quad\mathcal E(Z)=\lambda Z,\quad\lambda=2p-1,\quad
 \operatorname{spec}\mathcal E=\{1,\lambda,0,0\},\quad
 \ell=I/2+\sqrt{p(1-p)}X,\quad\mathcal E^\dagger(\ell)=\ell,\quad\operatorname{Tr}\ell=1.
$$

Transfer 식의 X·Z는 가상 Pauli 행렬이다. Ell은 가상 trace 함수이며 곧바로 물리 site의 축약 밀도가 아니다. 명시한 유한 경계에서는 관측하지 않는 경계 site 하나를 수축하면 이 정류 환경에 도달한다. 유한 Z 상관은 Markov 확률 법칙에서도 직접 얻는다.

두 Kraus형 항의 차이로 물리 Z를 삽입한다. 두 site가 존재하는 거리 r>=1의 연결 상관은 다음과 같다.

$$
\mathcal E_Z(X)=A^0X(A^0)^\dagger-A^1X(A^1)^\dagger,\quad
 \mathcal E_Z(I)=Z,\quad\mathcal E_Z(Z)=\lambda I,\qquad
 \langle Z_i\rangle=0,\quad C_{ZZ}(r)=\lambda^r,\quad
 \xi=-\frac1{\ln|\lambda|}\quad(0<|\lambda|<1).
$$

첫 삽입이 가상 Z를 만들고 전파 단계마다 lambda를 곱하며 두 번째 삽입이 lambda I로 돌아간다. 국소 평균이0이라 원래·연결 ZZ 상관이 같다. 음수 lambda는 부호 교대를 만들고 절댓값은 envelope를 정한다. 거리·xi의 단위는 물리 시간이 아닌 격자 간격이다. Transfer 관점은 [Schollwöck §4.2.2](https://arxiv.org/pdf/1008.3477)를 참고하며 이 가족과 유한 경계의 증명은 여기서 직접 구성했다.

고정 bond capacity가 상관 길이의 균일한 상한을 정하지는 않는다.

$$
p=1-\varepsilon,\quad\varepsilon\downarrow0:\qquad
 \lambda=1-2\varepsilon,\quad\xi\sim\frac1{2\varepsilon},\quad
 D=2\ \text{throughout}.
$$

p=1 극한은 bond2로 장거리 ZZ 질서를 갖는다. p=1/2에서는 서로 다른 site 상관이 사라지므로 lambda=0의 로그 식은 그 직접 결과로 읽고 임의 작은 고유값으로 나누지 않는다.

### 가상 좌표 변경에서도 물리 상태 보존하기

경계까지 포함해 각 virtual bond에 가역 복소 G를 적용한다. 가상 행렬을 열 방향 vectorization하면 다음과 같다.

$$
A'^s=G^{-1}A^sG,\quad L'=LG,\quad R'=G^{-1}R,\quad
 S=\overline G\otimes G,\quad E'=S^{-1}ES,\quad\kappa_2(S)=\kappa_2(G)^2,\qquad
 \|E'^r-P'\|_2\le\kappa_2(S)\|E^r-P\|_2,\quad
 P'=S^{-1}PS\quad(0<p<1).
$$

E는 transfer superoperator의 행렬이고 P는0<p<1 내부의 정류 spectral projector다. Similarity는 고유값을 보존하지만 비unitary gauge는 노름·수치 prefactor를 키울 수 있다. 조건수 인자는 노름의 곱 상계로 얻으며 반드시 달성되는 값은 아니다. 물리적 gauge 불변 수축은 경계·연산자 삽입까지 일관되게 변환한다. Raw transfer 노름 증가가 곧 물리 상관 증가를 뜻하지는 않는다.

가장 간단한 비교의 두 경우는 실제 rank 사용도 다르다.

$$
p=1:\quad\psi=(|0\cdots0\rangle+|1\cdots1\rangle)/\sqrt2,\quad C_{ZZ}(r)=1;\qquad
 p=\tfrac12:\quad\psi=|+\rangle^{\otimes N},\quad C_{ZZ}(r)=0\quad(r\ge1).
$$

p=1/2에서는 곱상태에 할당한 bond2가 중복이며 최소 bond1 표현이 가능하다. p=1에서는 GHZ 상관이 감쇠하지 않고 주변 스펙트럼의 고정 방향이 여러 개다. 모든 유한-D MPS의 상관 길이가 유한하다는 일반 문장은 여기서 실패한다. 이 절은 parent Hamiltonian이나 그 excitation gap을 명시하지 않았다.

거리4 열은 N>=5를 택한다. 이 해석적 가족이 reader의 크기 제한을 늘리는 것은 아니다.


```python
from fractions import Fraction as Q
for p in [Q(1, 4), Q(1, 2), Q(3, 4), Q(1)]:
    second = 2*p-1
    print(p, second, second, second**2, second**4)
```

| 파라미터 p | lambda | CZZ(1) | CZZ(2) | CZZ(4) |
|---|---|---|---|---|
| 1/4 | -1/2 | -1/2 | 1/4 | 1/16 |
| 1/2 | 0 | 0 | 0 | 0 |
| 3/4 | 1/2 | 1/2 | 1/4 | 1/16 |
| 1 | 1 | 1 | 1 | 1 |

더 일반적인 transfer에서는 diagonalizability도 확인해야 한다. 크기2 Jordan block이 기본적인 주의점을 준다.

$$
J_\lambda=\lambda I+N,\quad N^2=0,\quad\lambda\ne0:\qquad
 J_\lambda^r=\lambda^rI+r\lambda^{r-1}N\quad(r\ge1).
$$

지수 외에 다항 prefactor가 붙는다. 조건부 선형대수 비교이며 앞의 명시적 가족에는 이 prefactor가 필요하지 않다. 또한 관측량이 특정 subleading mode와 겹치지 않을 수 있다. 따라서 decay length를 맞출 때는 한 고유값을 모든 관측량의 감쇠로 읽지 말고 스펙트럼·삽입·경계·점근 영역을 명시해야 한다.

![Bond2 가족에서 연결 ZZ 상관은 부호 교대·감쇠·장거리 값을 보일 수 있다. 할당 bond dimension을 늘리지 않아도 상관 길이는 커진다.](/assets/learning/quantum-representation-research-transfer-correlation-gauge.ko.svg)

표는 정확하다. 해석적인 transfer 가족·조건 예제이며 infinite-MPS solver나 canonicalization/truncation sweep을 구현한 것이 아니다. 기존 유한 reader는 일반 복소 유리수 tensor·환경·scale/shear 비교를 유지한다.

#### 연구 확인 1: 홀수 거리에서 부호가 남는다

p=1/4이면 lambda=−1/2·xi=1/ln2이고 거리1,2,4 상관은 −1/2,1/4,1/16이다. p=3/4에서는 크기가 같지만 홀수 거리의 부호가 반대다. 짝수 거리만 맞추면 이 정보를 잃는다.

#### 연구 확인 2: 작은 bond로 장거리 상관

p=1에서는 전부0·전부1인 문자열만 남는다. N>=2의 국소 Z 평균은0이지만 서로 다른 모든 쌍의 ZZ는1이다. Bond2로 감쇠 없는 상관을 표현한다. Transfer에는 고유값1이 하나 더 있어 고립된 leading mode의 지수 논증을 적용할 수 없다.

#### 연구 확인 3: 큰 좌표 prefactor

G=diag(g,1)·g>0이면 kappa2(G)=max(g,1/g)이며 transfer similarity의 조건수는 그 제곱일 수 있다. 두 경계를 변환하면 물리 상태·정확 ZZ 표는 같다. 가상 행렬 penalty나 중간 노름이 바뀌어도 물리 target이 바뀌지 않을 수 있다.

## 15. 여덟 문제와 해설

**1. 경로 추적.** 제시한 GHZ 텐서에서000,111,101의 원시 진폭과 전체 노름 제곱을 구하라.

**해설.** 000은 레이블0의 경로로1,111은 레이블1의 경로로1이다. 첫1이 선택한 레이블을 가운데0의 projector가 없애므로101은0이다. 다른 섞인 문자열도0이므로 N=2다. 정규화한 두 진폭은1/√2이며 확률1/2와 구별한다.

**2. 용량과 rank.** Bond1이 정규화 GHZ₃를 정확하게 표현할 수 있는가? |000〉를 bond4에 저장하면 Schmidt rank가4가 되는가?

**해설.** GHZ 계수 행렬의 rank는 양쪽 cut에서2이므로 bond1의 상한을 넘는다. 곱 상태의 계수 행렬은 비영 원소 하나로 rank1이다. 추가 가상 좌표는 용량과 중복을 제공하지만 그 용량을 물리 상태가 모두 사용하도록 강제하지 않는다.

**3. Closure 방향.** L=(1,i), B=(1,i)ᵀ에서 진폭·E·R과 두 가지 closure 후보를 계산하라.

**해설.** 진폭은1−1=0이고 두 Gram은 [[1,i],[-i,1]]이다. 같은 색인의 곱 합은1−1−1+1=0이다. Tr(ER)=4는 이 convention에서 틀렸다. 이 수축으로 정규화 상태나 국소 물리 density를 보고해서는 안 된다.

**4. 위상 부호.** |0〉⊗(|0〉+i|1〉)의 오른쪽 Bloch 벡터를 구하라.

**해설.** √2로 정규화하면 density의 오른쪽 위 원소가−i/2다. X=2Reρ₀₁=0,Y=−2Imρ₀₁=1,Z=0이다. 실수 상태가 드러내지 못하는 켤레 오류를 잡는다. 첫 qubit은 여전히 |0〉이며 오른쪽의 위상과 얽힘을 혼동하지 않는다.

**5. Gauge 삽입.** 제시한 G를 첫 GHZ bond에 적용하면 N,K,rank는 어떻게 되는가?

**해설.** G와 inverse는 물리 문자열 합을 수행하기 전에 상쇄된다. 모든 진폭·N=2·두 rank2를 보존하지만 각 site의 좌표 길이는6,3/2,2가 되어 K가6에서19/2로 바뀐다. 하나의 물리 상태를 나타내는 표현들 사이에서 좌표 norm penalty가 달라질 수 있다.

**6. 바뀐 환경 닫기.** E′=[[4,2],[2,2]], R′=[[1/2,−1/2],[−1/2,1]]을 닫아라. 각각이 identity일 필요가 없는 이유는 무엇인가?

**해설.** 같은 위치의 곱은2,−1,−1,2이며 합은2다. 두 환경에는 서로 반대의 좌표 변경이 들어 있다. Gauge 불변성은 물리 closure에 적용되며 각 가상 metric 원소의 개별 보존을 뜻하지 않는다. 복소 G이면 명시한 켤레도 필요하다.

**7. 정규화 위치.** 처음과 가운데 GHZ site가 left isometry이면 원시 상태는 정규화됐는가? 해석적인 정규화 방법을 하나 제시하라.

**해설.** 마지막 left Gram이2이므로 전체 노름 제곱도2다. 마지막 site만√2로 나누면 앞의 isometry를 유지하며 단위 노름이 된다. 유리수 계산기는 대신 원시 텐서를 유지하고 물리 출력을 정규화할 수 있다. 두 선택은 출력 상태는 같아도 원시 표현이 다르다.

**8. 전역 모호성.** One-site와 pair 평균만으로 GHZ와 같은 비율의00…0/11…1 비간섭 혼합을 구별할 수 없는 이유를 설명하라.

**해설.** 세 site에서 하나를 trace하면 GHZ의 cross term이 사라져 혼합과 같은 pair density가 된다. XXX는 두 GHZ 성분을 coherent하게 교환하여 평균1이지만 혼합에서는 각 대각 기대값이0이다. 더 큰 관측 집합이 있어야 두 전체 상태를 구별할 수 있다.

## 16. 계산한 것과 다음에 배울 것을 정리한다

유한 모형은 순서가 있는1…4 qubit·bond1…4와 일반 복소 유리수 텐서를 입력받는다. 전체 벡터·모든 cut의 환경과 closure·물리 국소 density·pair Pauli 상관·정확 계수 rank·canonicality 진단을 계산한다. 지정한 가역 scale/shear에는 정확 inverse와 covariance 검사가 있다. 영 수축은 원시 대수를 유지하지만 정규화된 물리 상태는 반환하지 않는다.

예제는 모든 가상 경로의 명시적 합과 전체 물리 행렬 계산으로 별도 검산한다. 계산기는 바닥 상태를 탐색하거나 시간 진화·자동 canonicalization·bond 절단을 수행하지 않는다. 다음 장에서는 Schmidt 가중치·entropy·최적 저랭크 오차와 버린 상태/재정규화 상태의 차이를 배운다. DMRG·MPO·긴 사슬의 성능·연구 수렴 주장은 추가 유도와 근거를 필요로 한다.

배경은 [Schollwöck의 MPS 리뷰](https://arxiv.org/html/1008.3477v2)의 gauge 설명·수축 절·정규형 구성 절을 참고하자. 여기의 색인 convention과 예제는 위에서 직접 전개했다. 그림의 색이나 움직임을 해석하기 전에 열린 다리·합산하는 다리·복소 켤레·정규화를 말로 설명해 보자. 각 그림이 검증할 식을 먼저 정하면 시각화는 결과를 꾸미는 장식에서 계산을 이해하는 도구가 된다.
