# 논리·기초수학 원주제 대조

2026-11-11에 전체 지식 지도의 앞 두 영역을 원고와 대조했다. 기존 36영역 입문 집필 목록 밖에 있는 20개 원주제도 전체 교재 검토에 포함한다. 아래 표는 한영 원고의 실제 설명 위치를 기록한다. 영역별 완전 검토와 연구 심화는 열린 상태로 유지한다.

## 논리·집합·증명

| 원주제 | 한영 원고 식별자 | 확인한 내용 |
|---|---|---|
| 명제와 술어 | propositions-implications, quantifiers-counterexamples | 명제의 진릿값, 변수와 정의역, 관계 술어 |
| 전칭·존재 양화사 | quantifiers-counterexamples | 네 양화 형태, 순서 변경, 부정, 빈 정의역 |
| 집합·함수·관계 | sets-functions, relations-orders | 집합 연산, 정의역·공역, 함수 대응과 관계 |
| 동치·순서·사상 | relations-orders, sets-functions | 동치류, 부분순서와 비교 불가능성, 단사·전사 |
| 직접·대우·모순 증명 | proof-strategies | 정수의 짝수 성질에 대한 세 논증과 잘못된 약분 |
| 수학적 귀납법 | mathematical-induction | 기초·귀납 단계, 합 공식, 강한 귀납법 |
| 반례 구성 | quantifiers-counterexamples, mathematical-induction | 공통 증인의 부재, 정의역 변경, 늦게 등장하는 반례 |
| 필요·충분조건 | propositions-implications | 금지된 진릿값, 역과 대우, 공허한 참 |
| 정의·정리·가정·결론 | proof-strategies | 정의의 사용, 가정을 만족하는 반례, 일반 증명과 유한 검사 |

짝수 합을 직접 증명하는 첫 연습에 한영 해설을 추가했다. 두 정수를 각각 2a와 2b로 표현하고 정수의 덧셈 닫힘을 사용한다. 기존 증명 전략과 문제의 의도를 유지했다.

## 기초수학

| 원주제 | 한영 원고 식별자 | 확인한 내용 |
|---|---|---|
| 정수·유리수·분수·비율·단위 변환 | numbers-ratios-units | 약분, 백분율, 150 cm 변환, 평균 속도 |
| 실수·복소수 | real-complex-numbers | 유리·무리수, 복소수 쌍, 크기와 회전 |
| 방정식·부등식 | equations-inequalities | 해집합, 0 계수, 부등호 방향, 이차식 부호 구간 |
| 다항·유리함수 | polynomial-rational-functions | 정의역의 구멍, 약분, 제거 가능 불연속과 극점 |
| 지수·로그 | exponentials-logarithms | 밑과 정의역 조건, 로그 법칙 유도, 변화율 |
| 삼각함수 | trigonometry-coordinates | 단위원, 라디안, 기준 삼각형과 주기 |
| 함수 합성·역함수 | composition-inverse-functions | 합성 순서, 아핀 역함수, 분기 제한과 역상 |
| 좌표·거리·각도 | trigonometry-coordinates | p=(r cosθ,r sinθ), q=(1,0)까지의 거리, 각도 규약 |
| 수열·급수의 도입 | sequences-series | 유한 등비합 유도, 수렴 조건, 진동과 발산 |
| 단위와 차원 | numbers-ratios-units, trigonometry-coordinates | 변환 인자, 차원 일관성, 유클리드 좌표 가정 |
| 점근 표기 | asymptotic-growth | 상수와 문턱의 증인, O와 타이트한 경계, 비율 그림 |

좌표 단원의 거리는 원점까지의 반지름과 별도로 q까지 계산된다. 경위도나 학습한 특징 공간에 유클리드 공식을 적용하려면 모형 가정이 필요하다는 설명도 확인했다.

## 실행 검토와 후속 확인

검토에 사용하는 저장소 명령은 `npm run check:logic`과 `npm run check:elementary`다. 한영·320/768/1440px·명암·안내 예제·문제·저장 복원·키보드 스크롤을 검사한다.

다시 실행하며 검사 코드의 세 가지 문제를 수정했다.

- 수화 중 아직 없는 입력 요소를 읽기 전에 실제 준비 상태를 기다린다.
- 가로 넘침이 있는 요소에서 해당 요소의 키보드 스크롤을 확인한다.
- 정수 입력에 소수를 넣었을 때 새 입력 규약에 따라 오류 표시, 이전 계산 보존, Escape 복원을 확인한다.

정적 배포 후보 `/tmp/suanlab-inverse-final.JgKYxl`을 대상으로 두 명령을 완료했다. 논리는 한영·반응형·접근성 49개 상태와 상호작용 9개 묶음, 안내 예제 18개를 통과했다. 기초수학은 70개 상태와 상호작용 10개 묶음, 안내 예제 27개를 통과했다. 검사 결과에 페이지 실행 오류와 WCAG A/AA 위반이 없었다. 실행 결과는 `.runtime/learning/logic-browser.json`과 `.runtime/learning/elementary-browser.json`에 저장했다.

이후 다른 13개 초기 영역의 127개 원주제를 같은 방식으로 대조한다. 원주제별 정리·증명·오차·AI 연구 연결의 심화 검토도 계속 진행한다.
