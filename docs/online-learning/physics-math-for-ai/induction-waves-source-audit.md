# 유도·전자기파의 자료 대조와 독립 준비

2026-11-05 자료 준비 당시의 기록이다. 이후 [계산 모델·상세 한영 원고](induction-waves-foundation.md)를 작성·검증했다. 이후 [홈페이지 등록·시각화·상호작용 통합 검증](induction-waves-authoring-status.md)을 마쳤다. 아래 독립 준비 자체로 완료 수를 늘리지는 않는다.

## 확인한 1차 자료

[MIT 8.02 Chapter 10](https://ocw.mit.edu/courses/8-02-physics-ii-electricity-and-magnetism-spring-2007/ce1720fd4b21def8c2189ff4779f27f7_cha10faraday_law.pdf)의 §10.1–10.3은 유도, 운동 기전력, 비보존 전기장을 구분한다. 고정 경로의 전기장 순환과 움직이는 도선의 v×B 항을 구별하는 근거로 사용한다. 실제 전류·저항·운동 응답을 얻으려면 장뿐 아니라 장치 조건도 필요하다.

[Feynman Lectures II Chapter 20](https://www.feynmanlectures.caltech.edu/II_20.html)의 §20–1 식20.24–20.25는 양방향 진행 함수와 그에 맞는 전기·자기장 관계를 제시한다. 축을 바꿔 E x=f(z−ct)+g(z+ct), B y=[f(z−ct)−g(z+ct)]/c로 직접 유도한다. 스칼라 파동 방정식과 일차 Maxwell 관계의 일치를 별도로 검사한다.

## 후보와 교육적 초점

| 후보 ID | 중심 질문 |
| --- | --- |
| `faraday-induction-fixed-moving-loop` | 시간에 따른 B 변화와 경로의 움직임은 기전력에 어떻게 따로 기여하는가? 고정 경로 E 적분과 움직이는 직사각형의 E+v×B 적분을 비교한다. |
| `electromagnetic-wave-equation-transverse-initial-data` | 같은 초기 E라도 B와 시간 미분의 호환성에 따라 단방향 이동과 양방향 분리가 어떻게 달라지는가? Gaussian 두 진행 함수를 사용한다. |

첫 후보는 유계 영역에 균일한 B z=b(t)를 지정한다. E=(b′y/2,−b′x/2,0)이면 curl E z=−b′다. 일정 ε·μ에서 J=−ε∂tE를 함께 지정하면 B의 회전0과도 일치한다. 이는 필요한 경계와 원천을 공급하는 제조해이며 유한 솔레노이드의 정확한 전 공간 해가 아니다.

xy 직사각형을 +x 다음 +y로 돌면 법선은+z다. 폭 L(t), 높이 H인 경로의 E 순환은−b′LH, 운동 기여는−bL′H다. 합은−d(bLH)/dt와 비교한다. 경로의 운동은 지정된 조건이며 도선의 힘·되먹임·저항 전류를 푼 것이 아니다. 고정 경로 식만으로 움직이는 면의 전체 선속 미분을 설명하지 않는다.

둘째 후보는 충분히 매끄러운 두 Gaussian 진행 함수를 합한다. 초기 B=E/c인 단방향 장과 초기 B=0인 대칭 분리 장을 비교한다. E가 스칼라 파동 방정식을 만족해도 B=0을 임의로 붙인 단방향 이동은 일반적으로 Maxwell 진화 식을 위반한다. 이 평면 장은 가로 방향으로 무한하며 유한한 삼차원 안테나 복사장이 아니다. 에너지·Poynting 흐름·매질 경계는 후속 주제로 남긴다.

## 실제 독립 탐색

`.runtime/learning/prepare-induction-waves.py`는 제품 코드를 가져오지 않는다.81개 움직이는 직사각형 조건에서 실제 E·dℓ와 (v×B)·dℓ를 네 모서리에서 SciPy 적분하고, 자속의 시간5점 차분과 비교했다. 최대 관측 차이는1.36×10⁻¹² 이하다.

54개 Gaussian 양방향 조건에서는 공간·시간5점 차분으로 Faraday·Ampere 관계와 파동 방정식을 검사했다. 최대 관측 차이는1.26×10⁻⁹ 이하다. 선택한 정규화 계수와 표본에서의 관측값이며 엄밀한 오차 상한이 아니다. 결과는 `.runtime/learning/induction-waves-exploration.json`, 실제 종료는 `induction-waves-preparation-exit.json`이다.

별도 음성 대조9조건에서는 이동 Gaussian E에 B=0을 붙였다. 스칼라 파동 방정식과 두 발산은 통과하지만 일반 위치의 일차 Maxwell 관계는 실패함을 실제 차분으로 확인했다. 일정 B에서 폭이 변하는 경로1조건은 E 순환0이어도 운동 기전력이 비영임을 확인했다. 증거는 `.runtime/learning/induction-waves-negative-controls.json`, 실제 종료는 `induction-waves-negative-controls-exit.json`이다.

추가로160자리 mpmath 계산에서 작은 시간 이동의 Gaussian 차이4조건과 자기장 변화·경로 이동에 따른 선속 차이2조건을 검증했다. 직접 차감은0을 반환할 수 있지만 expm1 또는 다항식의 상수 선소거는 비영 차이를 유지한다. 일정한 B는 움직이는 경로의 기전력에 실제로 기여하므로 전위의 게이지 상수와 혼동하지 않는다. 증거는 `.runtime/learning/induction-waves-cancellation-preparation.json`이다.
