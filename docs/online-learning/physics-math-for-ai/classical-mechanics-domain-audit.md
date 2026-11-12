# 뉴턴 역학·강체 운동 10개 원주제 종합 대조

2026-11-12. 교육과정에 등록된 **10개 원주제·21단원·42개 한영 원고**를 현재 소스·계산 모델·최종 정적 출력에 대조했다. [단계별 보강 기록](classical-mechanics-topic-review.md)과 [원고/자산 해시·원주제 연결·재현 파일 목록](classical-mechanics-domain-audit.json)을 함께 따른다. 기존 입문 집필 완료와 구분한 원주제 대조 기록이다.

## 원주제와 단원 연결

아래 연결은 `src/data/learning-curriculum.ts`의 원래 주제 이름을 그대로 사용한다. 21단원이 중복 없이 연결되며 `mechanics-energy`의 foundation evaluator와 다른 20단원의 classical evaluator를 모두 포함한다.

| 원주제 | 등록 단원 ID | 대조한 가정과 구분 |
|---|---|---|
| 힘·질량·가속도 | `vector-force-superposition-dynamics`, `accelerating-observer-inertial-force` | 같은 양의 질량·관성계에서 벡터 합력을 쓰며 가속 좌표의 관성항을 실제 상호작용과 구분한다. |
| 운동량·충격량 | `finite-force-pulse-impulse-histories`, `two-particle-momentum-center-of-mass` | 유한 펄스의 같은 충격과 다른 변위를 비교하고, 계 경계별 내력·외력 및 질량중심 수지를 구분한다. |
| 일·에너지 | `mechanics-energy`, `quartic-potential-work-allowed-states`, `damped-oscillator-work-power-heat` | 수정 에너지/물리 에너지·안정/위상·허용 상태/도달 시간을 구분한다. 감쇠의 일·열 수지와 임계 근을 따로 다룬다. |
| 각운동량·토크 | `fixed-origin-angular-momentum-torque`, `central-harmonic-angular-momentum-area` | 고정 관성 기준점의 토크·각운동량과 중심에 대한 면적 법칙을 계산하며 임의 비교점의 변화를 확인한다. |
| 보존 법칙 | `isolated-central-pair-conservation`, `gyroscopic-force-energy-momentum` | 중심 내부 쌍의 P,L,E와 속도에 수직인 힘의 영 일률을 구분한다. 에너지 보존만으로 벡터 운동량을 추론하지 않는다. |
| 마찰·구속 | `dry-friction-stopping-release`, `circular-guide-reaction-work` | 고정 수평 법선 하중의 정지/운동 마찰과 양방향 원형 가이드를 명시한다. 마찰열·부호 있는 일·반력의 일률을 구분한다. |
| 중심력 | `inverse-square-effective-potential`, `elliptic-kepler-time-area` | 고정 원천·양의 μ·원점 특이성을 명시한다. 방사 분기와 엄격한 타원 시간 실습을 구분하며 미세 오차/포물선 극한은 별도 예제다. |
| 회전·관성텐서 | `point-mass-inertia-tensor`, `torque-free-euler-body-lab` | 순간 피벗 변경이 속도장도 바꿈을 명시한다. 실현 가능한 관성·몸체/실험실 수송·중간축·자세 위상을 검증한다. |
| 강체·접촉 | `frictionless-rod-contact-impulse`, `driven-wheel-rolling-sliding` | 단일 법선 순간 충돌과 연속 접촉을 분리한다. grazing 식별, 정지 한계, 상대 미끄럼 열과 작은 마찰의 극한 순서를 설명한다. |
| 초기 상태의 역할 | `inverted-spring-initial-state-sensitivity`, `nonlipschitz-newton-delayed-release` | 완전 상태·관측 역산·단위별 오차·비유일한 원래 법칙을 구분한다. 두 Lipschitz 정규화의 위치/속도/가속도 선택 극한을 유도한다. |

## 내용·예제·출처

- 42개 현재 원고의 핵심 방정식·명시한 질량/원천/좌표/접촉 가정과 연구 확장의 적용 범위를 대조했다. 실제 실습 범위, 별도 해석 가족, 수치 근사와 해석 증명을 구분한다.
- 각 언어의 원래 연습은 **124개**이며, 20개 단원은 6개씩, 기초 에너지 단원은 4개다. 기초 단원의 빠진 2–4번 해설을 발견해 한영으로 보완했다. 강성 증가의 엄격한 안정 간격, 연속 감쇠 수지, 고정 물리 시간 비교와 100회 고정 실습의 차이를 명시했다. 기존 질문은 보존했다.
- 한영 Python 셀 **42개**를 모두 실행했고 대응 출력이 일치했다. 기초 단원의 새 해설은 종이 유도이며 감쇠·반복 수 조작을 실제 실습에 추가했다고 표시하지 않는다. 마지막 두 단원의 관측 상계·정규화 A/B와 역시간 해석 상계를 본문에 통합했다.
- 42개 퀴즈의 정답과 한영 설명을 검토했다. 63개 안내의 실제 preset 입력, 84개 개념 선택, 42개 정답/해설 표시를 최종 브라우저에서 확인했다.
- 선수 관계가 모두 실제 단원으로 연결되고 순환하지 않음을 확인했다. 이 21단원의 선수 폐포에는 156개 등록 단원이 포함된다. 영역 선수는 `physical-modeling`, `differential-equations`다.
- 원고의 외부 참고 URL 25개는 접근을 확인했다. NASA 문서는 HEAD 500 뒤 실제 GET 200 및 원문 열람으로 확인했다. URL 응답 성공을 수학 증명의 대용으로 사용하지 않는다. 새로운 특수 예제의 논증은 각 본문에 제시한다.

## 계산과 최종 화면 검증

23개 선택 테스트 파일의 **382개 검사**가 모두 통과했다. 별도 수치 참조의 기록 수는 JSON에 보존한다. 벡터 힘·가속 관측자, 유한 충격량·질량중심, 일/열, 각운동량, 보존 법칙, 마찰/구속, 역제곱/Kepler, 관성/Euler, 접촉/굴림, 초기 상태의 전 모델군을 포함하며 foundation의 에너지 수치 해법도 검사한다. 이 수치를 전체 저장소 테스트 또는 전체 CI 완료로 바꾸어 부르지 않는다.

기초 해설 보완 뒤 최종 빌드에서 전체 21단원을 **한영 320/768/1440px·영어 다크 모드, 147상태**로 다시 검사했다. 페이지 오류와 WCAG2A/AA·2.1AA 위반은 0이었다. 목차 앵커·수식 렌더·이미지·모바일 수식/표/코드의 키보드 스크롤·그림 수치와 레이블 범위를 확인했다. 기초 에너지의 추가 해설도 별도 **7상태·3안내·소수 간격/저장/퀴즈/완료**에서 확인했다. 마지막 두 단원은 별도로 **43상태·6상호작용 그룹·6안내**에서 정수 초안/blur/Escape·저장/언어/완료·경계와 원래 가지를 검사했다.

lint·콘텐츠·빌드·타입·정적 출력 검사가 통과했다. **1,893경로·1,888 HTML**의 내부 링크/자산/앵커/사이트맵/canonical과 **42개 공개 그림**의 소스/출력 바이트를 확인했다. 21개 실제 preview HTML도 최종 출력과 일치했다. 등록 650단원·1,969안내는 유지한다.

## 재현

정적 빌드와 preview를 준비한 뒤 다음처럼 동일 테스트 목록과 전체 reader 검사를 실행한다.

```bash
python3 - <<'PYTEST'
import json, subprocess
path = 'docs/online-learning/physics-math-for-ai/classical-mechanics-domain-audit.json'
files = json.load(open(path))['testFiles']
subprocess.run(['npx', 'tsx', '--test', '--test-concurrency=2', *files], check=True)
PYTEST
SUANLAB_TEST_BASE=http://127.0.0.1:4175 node scripts/check-classical-domain.cjs
```

## 범위와 다음 순서

이 기록은 등록된 원주제와 해당 실습 가족의 대조다. 일반 다체 충돌, 임의의 접촉 구성법칙, 비선형 혼돈 전체, 실제 장치에 대한 정규화 선택이나 무제한 물리적 외삽을 검증한 것이 아니다. 분야의 모든 연구 이론이 완성되었다는 뜻도 아니다. 보존량 잔차·정적 모양·URL 응답만으로 동역학 정확성을 주장하지 않는다.

대조 기록은 **39영역·386개 원주제**가 연결되었고 후속 **12영역·123개 원주제**가 남는다. 다음은 해석역학과 구조 보존의 10개 원주제다. 전체 **51영역·509개 원주제**와 연구 심화 목표를 유지하며 전체 교육과정 완료 상태는 바꾸지 않는다. 공개 홈페이지 반영 완료는 commit/push와 별개로 배포 결과를 확인한다.
