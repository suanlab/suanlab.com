# 전하·전류·장의 자료 대조와 독립 준비

2026-11-05 자료 준비 당시의 기록이다. 이후 파동 영역 점검을 마치고 [계산 모델·상세 한영 원고](charge-current-fields-foundation.md)를 작성·검증했다. 이어 [홈페이지 연결·통합 검증](charge-current-fields-authoring-status.md)을 마쳤다.

## 확인한 1차 자료

[MIT8.02T의 Coulomb's Law](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/c31cd03c1b74a450ec131921f5625f0a_chap2coulomb_law.pdf)에서§2.1~2.4의 전하 보존, 부호 있는 벡터 중첩, 시험 전하에 대한 장의 정의를 확인했다. 점원 특이점과 시험 전하가 원천을 교란하지 않는 조건을 구분해 집필한다. 평면 위 점전하 배치를 그리더라도 공간 차원이 자동으로 이차원 정전기학이 되는 것은 아니다.

[같은 과정의 Current and Resistance](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/6b2cf7ec2d597d7459930a7e6f700bea_chapter6current.pdf)에서§6.1~6.1.1의 방향 있는 전류, 면적 적분, 운반자 부호와 drift 방향의 관계를 확인했다. 회로의 일정 전류를 모든 공간의 전하 밀도 변화와 무조건 같다고 설명하지 않는다. 위 장 전체와 모든 예제를 검토했다는 뜻은 아니다.

## 후보 두 단원

| 후보 ID | 설명·실험의 중심 |
| --- | --- |
| `charge-superposition-test-field` | 원천 전하·시험 전하·벡터 장의 분리, 부호·거리·방향, 중첩·상쇄, 장과 힘, 스칼라 값·벡터·field line의 차이 |
| `charge-continuity-surface-current` | 이동 전하 밀도·부호 있는 전류·면 법선, 고정 제어 구간의 총전하, 양쪽 경계 유입·유출과 국소/적분 보존 |

점원 모델의 비영 특이점을 임의의 작은 거리로 덮지 않는다. 그림용 화살표 길이 조절과 원래 장 크기를 분리한다. 정적 원천의 Coulomb 장을 빠르게 움직이는 전하의 완전한 Maxwell 해로 부르지 않는다. 전하 밀도 수송 예제도 연속 방정식의 구체적 준비이며 전자기장의 자기 일관된 해까지 계산한 것은 아니다.

## 실제 독립 탐색

`.runtime/learning/prepare-charge-current-fields.py`는 제품 코드를 가져오지 않는다. 평면 배치에 삼차원 Coulomb 커널을 사용하여40개 비특이 관측에서 벡터 합과 퍼텐셜의 독립5점 미분을 대조했다. 상수 k는 인수로 분리하여 E/k와 V/k를 기록했다.

부호 있는 Gaussian 전하 밀도의 이동28조건에서는 제어 구간 전하의 CDF식과 독립 적분, 양쪽 경계 순유입과 독립 시간 미분을 대조했다. 결과는 `.runtime/learning/charge-current-fields-exploration.json`, 실제 종료는 `charge-current-fields-preparation-exit.json`에 있다. 이 계산이 미래 제품 모델의 검증을 대신하지 않는다.
