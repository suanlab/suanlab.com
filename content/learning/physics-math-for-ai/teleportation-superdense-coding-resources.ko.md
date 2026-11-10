## 1. 회로를 그리기 전에 자원을 적자

양자 텔레포테이션은 공유 Bell 쌍 하나와 전송되는 고전 bit 두 개로 미지의 큐비트 상태를 옮긴다. 초밀집 부호화는 공유 Bell 쌍 하나와 전송되는 큐비트 하나로 네 가지 고전 메시지 중 하나를 보낸다. 공유 얽힘도 계산해야 할 자원이다. 어느 프로토콜도 공짜 통신을 제공하지 않는다.

이번 장은 완벽한 게이트·측정·전송을 가정한 이상적 프로토콜이다. 가지를 유도하고 복소 진폭을 검증한다. 잡음이나 구현 충실도는 추가 분석이 필요하다.

[회로 순서와 간섭](/book/online/physics-math-for-ai/finite-qubit-circuit-order-interference/), [고전 측정 기록과 조건부 제어](/book/online/physics-math-for-ai/quantum-instruments-records-feedforward/)를 복습하자.

## 2. 레지스터 순서와 초기 상태를 고정한다

텔레포테이션의 site 순서는 0,1,2이며 각각 Alice 입력, Alice의 Bell 절반, Bob의 Bell 절반이다. site0은 계산 기저 정수의 최상위 bit다. 입력은

$$
|\psi\rangle=\alpha|0\rangle+\beta|1\rangle,\qquad |\alpha|^2+|\beta|^2=1
$$

이고 전체 상태는

$$
|\psi\rangle_0\otimes|\Phi^+\rangle_{12},\qquad
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2}
$$

다. 수치 모형에는 정규화하지 않은 영이 아닌 정확 벡터도 넣을 수 있다. 원시 진폭의 크기는 유지하고 확률·밀도행렬에서 입력 norm으로 나눈다. 예를 들어 $(3,4i)$와 $(3/5,4i/5)$는 같은 정규화 상태를 나타낸다.

## 3. 네 측정 가지를 유도하자

Alice가 site0에서 site1로 CNOT을 적용한 뒤 site0에 $H$를 적용하면

$$
\frac12\sum_{a,b\in\{0,1\}}|ab\rangle_{01}\otimes X^bZ^a|\psi\rangle_2
$$

가 된다. Alice가 site0,1을 $Z$로 측정해 고전 bit $a,b$를 얻는다. Bob의 정규화 전 가지는 $(1/2)X^bZ^a|\psi\rangle$다. norm 제곱은 입력과 무관하게 $1/4$다. 계수 $1/2$는 확률이 아니라 진폭이다.

| 기록 $a,b$ | Bob의 원시 가지 | Bob이 시간순으로 실행할 게이트 |
|---|---|---|
| 0,0 | $\lvert\psi\rangle/2$ | 없음 |
| 0,1 | $X\lvert\psi\rangle/2$ | $X$ |
| 1,0 | $Z\lvert\psi\rangle/2$ | $Z$ |
| 1,1 | $XZ\lvert\psi\rangle/2$ | $X$ 다음 $Z$ |

행렬 곱은 오른쪽부터 작용한다. $X$ 다음 $Z$를 실행하면 연산자는 $ZX$이므로 마지막 가지를 정확히 상쇄한다. $ZX\,XZ=I$다.

## 4. 복소 입력으로 빠진 부호를 찾자

$\alpha=3/5$, $\beta=4i/5$를 넣으면 네 원시 벡터는

$$
\tfrac12\begin{pmatrix}\alpha\\\beta\end{pmatrix},\quad
\tfrac12\begin{pmatrix}\beta\\\alpha\end{pmatrix},\quad
\tfrac12\begin{pmatrix}\alpha\\-\beta\end{pmatrix},\quad
\tfrac12\begin{pmatrix}-\beta\\\alpha\end{pmatrix}
$$

다. 보정 후에는 모두 $(\alpha,\beta)^T/2$가 된다. 기록11에서는 $X$로 $(\alpha,-\beta)^T/2$를 얻고 $Z$로 원래 가지를 복구한다. 입력 밀도행렬의 비대각 원소는 $-12i/25$, $12i/25$이므로 이 부호도 보존되어야 한다.

$|0\rangle$만 검사하면 상대 위상을 확인하지 못한다. 정규화 밀도행렬만 검사하면 원시 전역 부호를 놓친다. 따라서 정확 계산은 원시 벡터와 밀도행렬을 모두 대조한다.

## 5. 메시지 전 Bob에게는 입력에 따른 국소 신호가 없다

Alice의 기록을 무시하면

$$
\rho_B=\frac14\sum_{a,b}X^bZ^a\rho\,Z^aX^b=I_2/2
$$

다. Bob의 국소 측정 통계는 입력 $\rho$와 무관하다. 조건부 상태는 달라도 평균에는 사용할 신호가 없다. Bob이 올바른 보정을 고르려면 Alice의 고전 bit 두 개가 도착해야 한다. 얽힘은 초광속 통신을 허용하지 않는다.

예정된 시각화에서는 가지를 조건으로 선택한 상태와 기록을 버린 혼합 상태를 분리한다. 가지 선택은 조건화이며, 기록을 모르는 Bob이 임의로 할 수 있는 행동이 아니다.

## 6. 텔레포테이션은 복제가 아니라 전달이다

측정 후 Alice의 두 큐비트에는 고전 기록이 남는다. 입력 $|\psi\rangle$의 온전한 별도 복사본이 남는 것이 아니다. Bell 쌍도 프로토콜에서 소모되므로 반복하려면 새 공유 쌍이 필요하다.

보정된 연산의 선형성 때문에 이상적 채널은 항등 채널이며 입력이 참조계와 얽혀 있어도 성립한다. 유한한 순수 입력 예제 목록은 설명일 뿐 증명은 아니다. 보정 후 가지 연산자 $K_{ab}=I/2$를 쓰면 $\sum_{ab}K_{ab}\rho K_{ab}^\dagger=\rho$다. 손대지 않는 참조 인자를 붙여도 같은 항등식이 성립한다.

## 7. 초밀집 부호화에서는 자원의 역할이 바뀐다

Alice와 Bob이 $|\Phi^+\rangle_{AB}$를 공유한다고 하자. Alice는 bit $(a,b)$를 보내려 한다. 규약을 고정해 자신의 절반에 $X^b$ 다음 $Z^a$를 실행한다. 연산자는 $Z^aX^b$다. 그 큐비트를 실제로 Bob에게 보낸다.

두 절반을 모두 갖게 된 Bob은 받은 큐비트에서 원래 자신의 절반으로 CNOT을 적용하고, 받은 큐비트에 $H$를 적용한 뒤 둘을 $Z$로 측정한다. 이 규약에서 복호 상태는 $|ab\rangle$다.

| 메시지 | Alice 절반에 적용한 연산자 | 전송 직전 Bell 상태 |
|---|---|---|
| 00 | $I$ | $\Phi^+$ |
| 01 | $X$ | $\Psi^+$ |
| 10 | $Z$ | $\Phi^-$ |
| 11 | $ZX$ | $\Psi^-$ |

$|\Psi^\pm\rangle=(|01\rangle\pm|10\rangle)/\sqrt2$다. 부호화 두 게이트의 순서를 뒤집으면 메시지11에서만 전역 부호가 달라져 복호 결과가 $-|11\rangle$가 된다. 관측되는 메시지는 같지만 원시 벡터는 다르므로 계산기는 그 부호를 유지한다.

## 8. 두 bit가 Holevo 상계와 모순되지 않는 이유

수신자의 결합계는 4차원이며 네 Bell 상태는 직교한다. 이 과제에는 기존 공유 ebit와 새로 보낸 큐비트가 있다. 고립된 보조 자원 없는 큐비트 하나로 두 bit를 보내는 과제가 아니다.

전송 전 Bob의 주변 상태는 모든 메시지에 대해 $I_2/2$다. Alice의 국소 Pauli 연산은 Bob의 축약 밀도행렬을 바꾸지 않는다. Alice의 큐비트를 받은 뒤 결합 측정을 할 수 있어야 메시지를 구별한다.

| 프로토콜 | 소모하는 공유 얽힘 | 전송하는 계 | 과제 |
|---|---:|---|---|
| 텔레포테이션 | 1 ebit | 고전 bit 2개 | 미지의 큐비트 상태 전달 |
| 초밀집 부호화 | 1 ebit | 큐비트 1개 | 선택한 고전 bit 2개 전송 |

초기 얽힘을 준비하고 나누는 비용은 이 전송 단계 계산 바깥에 있다. 자원 장부에서 이 가정을 계속 보여 줘야 한다.

## 9. 해설과 함께 푸는 문제

1. **가지 확률.** 원시 계수 $1/2$가 왜 확률 $1/4$일까? **해설:** Born 규칙은 norm 제곱을 쓴다. 입력이 비정규화이면 입력 norm 제곱으로도 나눈다.
2. **순서 점검.** 기록11에서 Bob의 가지가 $XZ|\psi\rangle/2$다. **해설:** $X$ 다음 $Z$로 전체 보정 $ZX$를 적용하면 원시 결과도 정확히 $|\psi\rangle/2$다.
3. **복소 coherence.** $(3/5,4i/5)$에서 $\rho_{01}$은? **해설:** $(3/5)(-4i/5)=-12i/25$다.
4. **메시지 없음.** bit 도착 전 $|0\rangle$와 $|+\rangle$ 전송을 Bob이 구분할 수 있을까? **해설:** 어느 축약 상태나 $I_2/2$이므로 불가능하다.
5. **전역 위상.** 초밀집 메시지11에 $ZX$ 대신 $XZ$를 쓰면? **해설:** 복호 결과는 $-|11\rangle$다. 측정 확률은 같지만 원시 벡터는 다르다.
6. **자원 계산.** 초밀집 부호화가 보조 자원 없는 큐비트 상계의 반례가 아닌 이유는? **해설:** 공유 Bell 쌍과 4차원 수신 결합계가 추가 자원이다.

## 10. 정확 회로에서 연구 문제로

잡음 있는 공유 얽힘, 불완전한 고전 제어, 손실, 참조계 보존, 유한 자원 증류는 서로 다른 문제다. 채널과 성능 기준을 명시해야 한다. 이상적인 순수 입력 시연만으로 구현의 충실도나 잡음 채널 용량을 정량화할 수 없다.

AI 기반 과학 소프트웨어에서는 레지스터 순서, 시간순 연산, 원시/정규화 표현, 자원 제약을 명시하는 습관이 중요하다. 현재 기반에는 정확 계산과 본문이 있으며 Worker·조작 화면·안내 실험·reader 통합은 남아 있다.

일차 출판 기록은 [Bennett 등의 텔레포테이션 논문(1993)](https://research.ibm.com/publications/teleporting-an-unknown-quantum-state-via-dual-classical-and-einstein-podolsky-rosen-channels), [Bennett–Wiesner의 초밀집 부호화 논문(1992)](https://research.ibm.com/publications/communication-via-one-and-two-particle-operators-on-einstein-podolsky-rosen-states)이다. 위 가지 규약과 손계산은 이 교육과정에서 명시한 것이다.
