# 회로·센서와 연속장의 자료 대조와 독립 준비

2026-11-05. Gauge freedom 다음 주제의 자료·탐색 기록이다. 초기 자료 준비 단계의 기록이다. 이후 [계산 모델·상세 한영 원고 검증](circuit-sensors-foundation.md)을 마쳤다. 등록·홈페이지 통합은 다음 순서이며 자료 준비 자체로 완료 수를 올리지 않는다.

이후 [홈페이지 통합 검증](circuit-sensors-authoring-status.md)을 마쳤다. 아래 미등록·다음 단계 설명과 수치는 해당 기록 당시 기준이다.

## 확인한 1차 자료

[Feynman II §22–1](https://www.feynmanlectures.caltech.edu/II_22.html)은 단자 전압·전류로 집중 소자를 기술하는 가정과 저주파 근사를 설명한다. [Haus·Melcher §3.2](https://web.mit.edu/6.013_book/www/chapter3/3.2.html)는 EQS와 MQS가 각각 무엇을 무시하는지 구분하는 근거다. [Feynman II §§17–1–17–2](https://www.feynmanlectures.caltech.edu/II_17.html)는 고정 경로의 Faraday 순환과 움직이는 도선의 v×B 기여, 물질 경로를 바꾸는 접촉에서 단순 선속 규칙을 주의해야 하는 이유를 확인하는 자료다. 아래 두 가족은 이 관계에서 직접 구성한다.

## 후보 단원

| 후보 ID | 중심 질문 |
| --- | --- |
| `rc-circuit-field-closure-approximation` | 집중 회로의 전하·에너지 수지가 맞으면 복원한 연속장도 정확한 Maxwell 해인가? |
| `finite-area-coil-sensor-observability` | 코일 전압은 한 점의 B인가, 면적 선속의 변화인가? 다른 장 계수를 구분하려면 무엇이 달라야 하는가? |

첫 단원은 C=εA/d, I=C V′와 RC V′+V=Vs를 사용한다. 저장 에너지 CV²/2, 저항 열 RI², 공급 Vs I를 구분한다. 평판 사이의 균일 Ez=V/d와 변위 전류에서 Bφ=(μεr/2d)V′를 복원하면 Ampere는 맞지만 Faraday에는 ∂tBφ=(μεr/2d)V″가 남는다. 따라서 회로 방정식·에너지 수지가 맞는다는 사실을 정확한 연속장 인증으로 바꾸지 않는다. 가장자리·도선·분포 인덕턴스·전파·방사는 이 집중 모형에서 별도 검토가 필요하다.

두 번째 단원은 Bz=B0(t)+g(t)(x²+y²), Eφ=−B0′r/2−g′r³/4의 매끄러운 국소 가족에서 시작한다. 발산 B=0, curl E=−∂tB를 만족하며 Ampere에 필요한 분포 J를 명시해야 한다. 반지름 R, 중심 o인 평면 코일의 선속은 πR²[B0+g(|o|²+R²/2)]다. 중심의 B만 면적에 곱하면 gR²/2의 평균 효과를 잃는다.

고정 반지름으로 평행 이동하는 물질 루프에는 −dΦ/dt를 전기 순환과 v×B 순환으로 나누어 확인한다. 회전·접촉 변경·변형 전체를 해결했다고 주장하지 않는다. 서로 다른 반지름 또는 위치의 선속 관측 행렬로 B0와 g의 식별 가능성을 비교한다. 이 해석에는 선속을 관측하거나 초기 선속을 알고 적분했다는 조건이 필요하다. 고정 코일의 단자 emf만으로 직접 식별하는 것은 B0′·g′이며 정적인 B0·g 성분은 미정이다. 같은 코일의 복제 측정은 잡음 감소에 도움이 될 수 있어도 독립 공간 정보를 늘리지 않는다. 실제 센서의 부하·역작용·잡음·보정은 후속 제품 범위에서 명시한다.

## 실제 독립 계산

`.runtime/learning/prepare-circuit-sensors.py`는 제품 코드를 가져오지 않는다. RC72조건에서 SciPy 초기값 해와 해석 전압, 공급−열의 시간 구적과 저장 에너지 변화를 대조했다. 복원한 B의 시간 차분으로 준정적 Faraday 결함도 확인했다. 유한 면적 센서24조건은 이동하는 원판의 면적 구적, 실제 E와 v×B의 두 선적분, 이동 선속의5점 시간 차분을 비교했다. 두 선속 관측 설계의 NumPy rank는1/2였다. 같은 계수 행렬을 고정 코일 전압에 쓰면 식별 대상은 장 자체가 아니라 시간 미분이다.

실제 종료 코드0: `.runtime/learning/circuit-sensors-preparation-exit.json`. 결과: `.runtime/learning/circuit-sensors-preparation.json`. 선택한 수치 근거이며 전역 오차 상한·실제 장치 인증·완성된 교재로 세지 않는다.

## 집필 순서

먼저 회로의 전하·에너지·단위·RC 시간 척도를 계산하고, 같은 회로에서 복원한 장의 Maxwell 잔차를 따로 읽는다. 이어 유한 면적 코일의 공간 평균·방향·이동 기여·관측 행렬을 시각화한다. 영입력·영시간·작은 변화·미정 상태와 근사 유효 범위를 제품 계산·독립 기준·한영 원고에서 검증한 뒤 홈페이지에 연결한다.

## 코일 전압의 식별 범위 보완

별도 `.runtime/learning/prepare-circuit-sensor-observability.py`에서 관측량을 emf로 고정했다. 반복·서로 다른 반지름·거의 같은 반지름·세 번째 이동 탐침 설계의 rank는1/2/2/3이었다. 실제 E/v×B 선적분을 관측 행렬과 대조하고, 서로 다른 일정 균일 B0를 더해도 이 코일 전압들은 같음을 확인했다. 고정 코일은 B0′·g′, 이동 탐침을 포함한 지정 설계는 B0′·g′·g를 구분할 수 있지만 일정 B0는 여전히 초기 선속이나 다른 관측이 필요하다. 작은 특이값의 잡음 증폭도 후속 모형에서 다룬다. 실제 종료 코드0과 결과는 `circuit-sensors-observability-preparation-exit.json`, `circuit-sensors-observability-preparation.json`이다.

[후속 집필 설계](circuit-sensors-authoring-plan.md) · [남은 전체 순서](remaining-next-steps.md)

## 기초 집필로 확장한 검증

재현 가능한 `scripts/learning/generate-circuit-sensors-references.py`에서 RC96조건·장 차분27조건·코일 구적24조건·측정 전압 선형계2조건과180자리 정확 입력 기하4조건·작은 시간3조건 및 반올림 이득의 추정 해를 생성했다. 원시 전압·이득의 곱 차이를 유지하여 먼저 정규화할 때의 손실을 줄이되, 관측 정보 손실과 조건수는 별도로 표시한다. [기초 집필 기록](circuit-sensors-foundation.md)은 초기 탐색과 새 제품·원고 검증을 구분한다.
