# 순차 Schmidt 압축·회로 미분 연구 보강

집필 2026-11-13, 최종 검증 2026-11-21. 기존 두 단원의 한영 4원고에 표시 수식 32개, Python·4행 표 각 4개, 연구 해설 문제 12개와 그림 4개를 추가했다. 기존 본문과 풀이 22개를 정확히 보존했다. [검증 근거](quantum-compression-gradient-research-evidence.json)와 [자료·가정 대조](quantum-compression-gradient-research-source-audit.md)를 따른다.

## 순차 압축의 실제 오차

`schmidt-cuts-bond-rank-truncation`은 유한 TT-SVD의 중첩 prefix projector·직교 잔차를 유도한다. 현재 residual의 손실 합은 실제 비정규화 오차이며 원래 cut 꼬리 합은 상계라는 차이를 계산한다. 최종 정규화·관측량 오차·sqrt(n−1) 준최적 보증을 분리한다. Sweep보다 좋은 명시적 product 후보로 전역 최적성이 실패하는 반례도 제시한다.

## 생성자에 맞는 미분과 측정 비용

`finite-qubit-circuit-order-interference`는 교환자·두 고유값 parameter-shift를 유도하고 두-qubit 회로의 편미분을 계산한다. 공유 변수·비선형 손실의 잘못된 이동, controlled Ry의 세 고유값 반례와 두 주파수에 맞춘 네 평가 규칙을 설명한다. 독립 Pauli shot의 gradient 분산과 연속 budget 배분의 가정을 명시한다.

## 검증

- CI 선언 Python 3.12 격리 환경에서 관련 11파일·820검사가 통과했다. 한영 수식·코드·표, 영어 한글 잔류·KaTeX, 원본문 hash·풀이 22개를 확인했다.
- 복소 순차 SVD 60상태에서 버린 조각 240쌍의 직교, bond rank 180검사·국소 unitary spectrum 180비교를 확인했다. 해석적 sweep 4예제·축퇴 2예제·더 좋은 product 후보를 직접 대조했다.
- Dense 회로 52준비의 직접 미분·두 이동을 비교했다. 임의 입력·관측량의 두 고유값 24경우와 controlled 네 이동 24경우, 제어 간섭 17경우를 검산했다. 독립 binomial 분산 4경우를 Fraction으로 정확히 합산하고 공유 변수 반례를 SymPy로 확인했다. 부동소수점 잔차는 방향성 구간 인증이 아니다.
- 빌드·타입·콘텐츠·정적 출력 검사 실제 종료 코드는 0이다. 현재 한영 렌더링 4편과 제공된 2 HTML·4 SVG 바이트를 비교했다.
- 한영 320·768·1440px 밝은 화면·1440px 어두운 화면 16상태, 개념·예측 후 해설·안내·퀴즈 각 6회와 저장·완료 복원 2그룹을 확인했다. Schmidt의 현재 입력·cut·budget·축퇴·선택 여부, 회로의 전체 입력·단계·정확한 최종 확률을 비교했다. 접근성 위반·페이지 오류가 없고 검증 브라우저를 종료했다.
- 원본 그림 4개·정상 읽기 모바일 화면 4개를 직접 검토했다. SVG·캡션 경계·목차 앵커와 긴 표·코드·수식의 키보드 스크롤을 확인했다.

## 보존과 다음 순서

기존 Schmidt 2…4 qubit의 한 cut 분석과 회로 1…3 qubit의 정확 H/S/T 게이트 실습 범위를 유지한다. 순차 압축·연속 optimizer·장비 sampling을 새 UI로 추가한 것이 아니다. 원장 939항목을 보존하고 4편을 더했다. 650단원·1,969안내·51영역·509원주제·입문 362/362 완료를 유지한다.

양자 순차 연구 대조는 **22/34단원**이다. 다음은 `qubit-channel-kraus-choi-verification`과 `system-environment-stinespring-reference`다. 전체 **45영역·447주제 대조 / 6영역·62주제 후속**을 유지한다. 양자 영역 종합 대조·교육과정 완료는 false이며 공개 배포는 따로 확인한다.
