# 유한 표본 tomography·시간 dephasing 연구 보강

2026-11-13. 기존 ensemble·spectrum/dephasing 두 단원의 한영 4원고에 표시 수식 32개, Python·4행 결과표 각 4개, 연구 해설 문제 12개와 그림 4개를 추가했다. 기존 원본문과 풀이 32개를 정확히 보존했다. [검증 근거](quantum-density-research-evidence.json)와 [자료 대조](quantum-density-research-source-audit.md)를 따른다.

## 계수 표본·물리적 추정·신뢰영역

`finite-ensemble-density-gram-tomography`는 정확 Born 복원과 유한 binomial 빈도의 차이를 명시한다. 선형 추정의 모멘트·양성 위반·보수적 confidence box를 유도한다. 방사형 Hilbert–Schmidt 거리 사영과 제약 최대우도를 구별하고, 동일 빈도·불균등 시행 데이터에서 적합점이 달라지는 예제를 푼다. 경계 rank는 원천의 순수성 증거가 아니며 점 추정과 통계적 불확실성은 별도다.

## 개별 CPTP 채널과 시간 합성의 차이

`qubit-density-spectrum-entropy-dephasing`는 η(s)≠0의 유일한 중간 map·Choi 조건과 η(s)=0의 직접 소거 논증을 구별한다. 두 X 상태의 trace distance와 CP-divisibility를 대조한다. 비음의 율의 지수 semigroup, 시행 동안 고정되는 무작위 주파수의 cos 시간 모델과 회복을 유도한다. 기존 u slider는 물리 시계로 바꾸지 않으며 일반 모든 비-Markov 개념을 동치로 취급하지 않는다.

## 검증

- CI에 선언한 버전의 Python 3.12 격리 환경에서 관련 8파일·247검사가 통과했다. 한영 수식·실행 코드·표가 같고 영어 원고에 한글 잔류가 없다. 연구 절을 제거한 원본문 hash와 해설 32개를 대조했다.
- 정확 binomial 모멘트 30준비·confidence box 27준비를 열거했다. 별도의 일반 3좌표 제약 최적화 4적합과 KKT·우도 이득을 본문 경계 호 이분법과 대조했다.
- 직접 map 합성 36쌍·중간 Choi 36쌍, 환경을 포함한 4×4 unitary의 부분추적 75준비·시간율 12준비와 trace-norm 거리를 확인했다. 정확 소거점은 cos(π/2)의 부동소수점 값으로 판단하지 않았다. 부동소수점 비교와 반올림 표는 방향성 구간 인증이 아니다.
- 빌드·타입·콘텐츠·정적 출력 검사가 실제 종료 코드 0으로 끝났다. 현재 한영 렌더링 4편과 제공된 2 HTML·4 SVG의 바이트를 대조했다.
- 한영 320·768·1440px 밝은 화면과 1440px 어두운 화면 16상태, 개념/안내/퀴즈 각 6회, 저장·완료 복원 2그룹이 통과했다. 접근성 위반·페이지 오류가 없고 브라우저 종료를 확인했다.
- 그림 원본 4개와 정상 읽기 모바일 캡처 4개를 직접 검토했다. SVG/캡션 경계·목차 앵커·긴 코드/수식/표의 키보드 스크롤도 확인했다.

## 보존과 다음 순서

기존 정확 ensemble·Gram·rank·qubit spectrum·signed dephasing 조작은 유지한다. 계수·MLE·신뢰영역·시간율·환경은 독립 연구 예제이며 새 UI 제어로 표시하지 않는다. 이전 검토 원장 915항목을 보존하고 4편을 추가했다. 기존 650단원·1,969안내·51영역·509원주제·입문 362/362 완료를 유지한다.

양자 영역 순차 연구 대조는 **10/34단원**이다. 다음은 `finite-state-covariance-uncertainty`와 `qubit-povm-instrument-backaction`이다. 전체 **45영역·447주제 대조 / 6영역·62주제 후속**은 유지한다. 영역 종합 대조·교육과정 완료는 false이며 공개 배포는 별도로 확인한다.
