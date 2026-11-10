# 정전·정자기의 자료 대조와 독립 준비

2026-11-05 자료 준비 당시의 기록이다. 이후 [순수 계산 모델·상세 한영 원고](electro-magnetostatics-foundation.md)를 작성·검증했다. 이후 [홈페이지 등록·시각화·상호작용 통합 검증](electro-magnetostatics-authoring-status.md)을 마쳤다. 아래 독립 탐색 자체를 주제 완료 근거로 삼지는 않는다.

## 확인한 1차 자료

[MIT 8.02 Gauss' Law](https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide04.pdf)의 §4.2 대칭 적용과 예제4.4의 균일한 비도체 구를 확인했다. 닫힌 면의 전기선속 관계와 구대칭을 이용한 점별 장 추론을 분리한다. 도체 평형의 표면 전하와 지정한 비도체 부피 전하는 서로 다른 준비다.

[MIT 8.02T Sources of Magnetic Fields](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/4c84733e3977ba42a930a4c0414f70d0_ch9sourc_b_field.pdf)의 §9.1 Biot–Savart, §9.3 Ampere 법칙과 예제9.3의 균일 전류 원통을 확인했다. 정상 전류의 닫힌 선적분 법칙 자체와 대칭을 통해 장을 상수로 꺼내는 계산 편의를 구분한다. 무한 원통은 유한한 열린 도선의 완전한 정상 회로가 아니다.

## 후보 두 단원

| 후보 ID | 집필의 중심 |
| --- | --- |
| `gauss-symmetry-electrostatic-sphere` | 지정한 구대칭 부피 전하, 포함 전하와 닫힌 면 선속, 내부·외부 장, 원점의 유한 벡터와 미정 방향, 경계의 연속 장과 한쪽 미분, 대칭 없는 장 추론의 한계 |
| `ampere-symmetry-steady-current-cylinder` | 무한 균일 전류 원통, 포함 전류와 닫힌 경로 순환, 방위각 방향·오른손 관례, 내부·외부 장, 축의 유한 벡터, 정상성·무한계 가정과 구성 법칙의 구분 |

닫힌 **면**의 선속과 닫힌 **선**의 순환을 다른 기하·단위로 보여 준다. 원점·축에서는 좌표 방향이 미정이어도 실제 벡터가0일 수 있다. 부피 밀도의 계단 경계에서는 장의 연속성과 점별 미분의 존재를 구분한다. 경계 미분을 임의 평균값으로 대체하지 않는다.

무한 원통의 바깥 장과 유한 장치의 총에너지·귀환 전류를 같은 것으로 취급하지 않는다. 전류 밀도를 지정한 계산이 재료의 구성 법칙이나 역학적 평형을 자동으로 푼 것은 아니다. 수치 계수 k와 μ0는 독립 준비에서 인수로 분리하며, 오래된 교재의 상수 표기를 현대 SI의 정확한 값으로 주장하지 않는다.

## 실제 독립 탐색

`.runtime/learning/prepare-electro-magnetostatics.py`는 제품 코드를 가져오지 않는다. 구63조건에서 Coulomb 구껍질의 각 평균 커널을 수치 적분한 전위를 독립5점 미분하고 Gauss 예측과 비교했다. 경계에서는 미분의 매끄러움이 달라 생기는 차이를 따로 기록한다.

원통63조건에서는 관측점 중심의 극좌표로 무한 필라멘트의 Biot–Savart 기여를 적분했다. 면적 Jacobian이 적분 가능한1/s를 상쇄한 뒤 원통 단면과 광선의 교차 길이를 사용한다. Ampere 장 공식을 적분함수로 다시 넣은 검사가 아니다. 음·영·양 전하/전류, 내부·경계·외부·축을 포함한다.

결과는 `.runtime/learning/electro-magnetostatics-exploration.json`, 실제 종료는 `electro-magnetostatics-preparation-exit.json`이다. 이 탐색은 앞으로 작성할 제품 모델의 검증을 대신하지 않는다.

## 모델·원고 작성 후 상수 확인

[NIST 2022 CODATA 상수표](https://physics.nist.gov/cuu/pdf/all.pdf)는 진공 투자율을 불확실성과 함께 제시한다. 제품의4π×10⁻⁷은 반올림한 교육용 근사이며 현대 SI에서 정확한 상수라고 주장하지 않는다. k=9×10⁹과 함께 정적 예제의 계수로 사용한다. 후속 전자기파 계산에는 ε·μ·광속의 일관성이 별도로 필요하다. 제품 모델과 독립 수치 기준의 실제 비교는 [후속 검증 기록](electro-magnetostatics-foundation.md)을 따른다.
