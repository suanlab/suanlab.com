## 1. 결맞은 광장의 진폭으로 데이터 표현하기

두 도파로의 복소 광장 진폭을 $x=(x_1,x_2)^T$로 표시합니다. 두 장이 섞일 때 보강되거나 상쇄되는 정도는 상대 위상에 따라 달라집니다. 공통 정규화에서 입력 파워는 $|x_1|^2+|x_2|^2$에 비례합니다. 음의 실수 진폭은 양의 진폭과 위상이 $\pi$만큼 다릅니다. 일반 광검출기는 세기 $|x_j|^2$를 읽으므로 진폭 부호 정보가 사라집니다. [Shen 외, 결맞은 나노광 회로의 딥러닝 실험](https://arxiv.org/abs/1610.02365)

## 2. 위상 조절로 두 채널을 회전시키기

이상적인 50:50 결합기와 위상 지연의 행렬을 다음 규약으로 둡니다.

$$
B=\frac{1}{\sqrt{2}}\begin{pmatrix}1&i\\i&1\end{pmatrix},
\qquad P(\theta)=\operatorname{diag}(e^{i\theta},e^{-i\theta}).
$$

결합기, 위상 지연, 역결합기를 차례로 통과하면 다음 회전을 얻습니다.

$$
B^\dagger P(\theta)B
=R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}.
$$

두 내부 경로의 위상 차이는 $2\theta$입니다. 입력 $(1,0)^T$에 대해 출력 파워는 각각 $\cos^2\theta$, $\sin^2\theta$입니다. $\theta$를 0에서 $\pi/2$로 늘리면 무손실 모형의 전체 파워가 보존되면서 파워가 한 포트에서 다른 포트로 이동합니다.

## 3. 채널별 감쇠 추가하기

실수 행렬의 특잇값 분해는 $M=U\Sigma V^T$입니다. 2차원 직교행렬은 회전과 필요할 경우 위상 반전으로 구성할 수 있습니다. 수동 감쇠기의 진폭 투과율은 최대 1이므로, 최대 특잇값이 1보다 큰 행렬은 배율을 줄여 구현하고 그 배율을 시스템에서 추적합니다. $D=\operatorname{diag}(1,1/2)$라 하면

$$
R(\pi/4)D R(-\pi/4)=\begin{pmatrix}3/4&1/4\\1/4&3/4\end{pmatrix}.
$$

진폭 투과율이 $1/2$인 채널은 파워를 $1/4$만 전달합니다.

## 4. 부호가 있는 예를 손으로 계산하기

위 행렬에 $x=(1,-1)^T$를 넣으면 $y=(1/2,-1/2)^T$를 얻습니다. 정규화된 입력 파워는 2, 출력 파워는 $1/2$입니다. 입력 전체의 부호를 바꾸어도 직접 검출 결과는 $(1/4,1/4)$로 같습니다. 결맞은 기준 광장과 섞으면 위상 차이를 세기 차이로 바꾸어 부호를 구별할 수 있습니다.

## 5. NumPy로 회로 검산하기

```python
import numpy as np

B = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)

def rotation(theta):
    phases = np.diag(np.exp(1j * np.array([theta, -theta])))
    return B.conj().T @ phases @ B

theta = np.pi / 4
R = rotation(theta)
D = np.diag([1, 0.5])
M = R @ D @ rotation(-theta)
x = np.array([1, -1])
y = M @ x
assert np.allclose(R.conj().T @ R, np.eye(2))
assert np.allclose(M, [[.75, .25], [.25, .75]])
assert np.allclose(y, [.5, -.5])
assert np.allclose(np.abs(y)**2, np.abs(M @ (-x))**2)
print(np.real_if_close(y), np.abs(y)**2)
```

## 6. 설정한 행렬을 검증할 실험 설계하기

교정한 결맞은 입력을 넣고 위상 기준을 사용해 진폭을 측정합니다. 복원한 행렬을 설정 목표와 비교합니다. 기저 입력의 파워만 재면 투과율의 크기를 확인할 수 있습니다. 입력 중첩을 추가하면 기저 입력의 파워에서 드러나지 않는 상대 위상 오차도 살필 수 있습니다. 위상 설정, 삽입 손실, 읽기 규약을 기록해 검증을 재현할 수 있게 합니다. 구현을 비교할 때는 광학 코어의 동작과 변환·제어·읽기를 포함한 시스템 비용을 기록합니다.
