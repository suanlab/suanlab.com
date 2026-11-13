# 불확정성·공동 POVM 연구 보강

2026-11-13. 기존 covariance·POVM 두 단원의 한영 4원고에 표시 수식 32개, Python·4행 결과표 각 4개, 연구 해설 문제 12개와 그림 4개를 추가했다. 기존 본문과 풀이 24개를 정확히 보존했다. [검증 근거](quantum-uncertainty-research-evidence.json)와 [일차 자료·가정 대조](quantum-uncertainty-research-source-audit.md)를 따른다.

## 여러 관측량의 fluctuation과 순도

`finite-state-covariance-uncertainty`는 Hilbert–Schmidt 벡터로 여러 관측량의 복소 Gram 양성을 유도한다. 세 Pauli의 고유값·회전 공변성·순도 항등식, 일반 두 방향의 외적 인자와 거의 평행한 축의 함정을 계산한다. 순수 qubit에서 복소 Gram rank 1과 실수 covariance rank 2가 다른 이유를 설명한다. 순도 항등식은 commutator 평균을 포함한 보정된 qubit Pauli 모델이며 두 히스토그램만의 보편적 순도 검사로 확대하지 않는다.

## 두 noisy 판독의 공동 구현

`qubit-povm-instrument-backaction`은 unbiased 직교 Z/X 판독의 공동 효과를 구성하고 κ²+λ²≤1의 충분성과 필요성을 각각 증명한다. Lüders Z 뒤 sharp/unsharp X 판독이 경계·내부를 실현하는 과정과 prepare instrument의 차이를 보인다. 두 보정된 주변 판독의 오차 곱은 장치 disturbance와 다르며, 강도 0에서 보정은 정의되지 않는다. Fraction 경계 예제와 원판·오차 곡선으로 연결한다.

## 검증

- CI 선언 버전의 Python 3.12 격리 환경에서 관련 8파일·262검사가 통과했다. 한영 수식·코드·결과표가 같으며 영어 원고 한글 잔류와 KaTeX 오류가 없다. 연구 절을 제거한 기존 원본문 hash와 풀이 24개를 대조했다.
- 직접 Hilbert–Schmidt Gram 24준비·Pauli 20준비·일반 방향 쌍 80준비·proper rotation 20준비, 공동 효과 36준비·순차 Kraus 합성 20준비·보정 모멘트 10준비를 별도 행렬로 검산했다. 순수 끝점의 두 rank는 SymPy로 정확히 확인했다. 수치 잔차 비교는 방향성 구간 인증이 아니다.
- 빌드·타입·콘텐츠·정적 출력 검사가 실제 종료 코드 0으로 끝났다. 현재 한영 렌더링 4편과 제공된 2 HTML·4 SVG의 바이트를 대조했다.
- 한영 320·768·1440px 밝은 화면과 1440px 어두운 화면 16상태, 개념/안내/퀴즈 각 6회, 저장·완료 복원 2그룹이 통과했다. 접근성 위반·페이지 오류가 없고 브라우저 종료를 확인했다.
- 최초 브라우저 검사에서 SVG 제목·세로축 글자의 경계 초과를 발견해 외곽 여백을 수정하고 새 빌드에서 재검사했다.
- 그림 원본 4개와 정상 읽기 모바일 캡처 4개를 직접 검토했다. SVG/캡션 경계·목차 앵커·긴 코드/수식/표의 키보드 스크롤을 확인했다.

## 보존과 다음 순서

기존 유한 상태 두 관측량·binary POVM 실습은 유지한다. 여러 관측량·일반 네 결과 공동 POVM은 독립 연구 예제이며 새 UI 제어로 표시하지 않는다. 이전 검토 원장 919항목을 보존하고 4편을 추가했다. 기존 650단원·1,969안내·51영역·509원주제·입문 362/362 완료를 유지한다.

양자 순차 연구 대조는 **12/34단원**이다. 다음은 `bipartite-pure-schmidt-partial-trace`와 `two-qubit-ppt-correlations-chsh`다. 전체 **45영역·447주제 대조 / 6영역·62주제 후속**은 유지한다. 영역 종합 대조·교육과정 완료는 false이며 공개 배포는 별도로 확인한다.
