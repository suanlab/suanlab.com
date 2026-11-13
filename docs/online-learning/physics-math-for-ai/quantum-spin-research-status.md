# Spin 회전 정보·coherent/cat alias 연구 보강

2026-11-13. 기존 spin1/2·일반 유한 spin 두 단원의 한영 4원고에 표시 수식 32개, Python·4행 결과표 각 4개, 연구 해설 문제 12개와 그림 4개를 추가했다. 기존 원본문과 풀이 24개를 정확히 보존했다. [검증 근거](quantum-spin-research-evidence.json)와 [자료 대조](quantum-spin-research-source-audit.md)를 따른다.

## 상태의 민감도와 검출기의 정보

`spin-half-measurement-su2-rotation`는 회전 생성자의 분산·ray metric에서 순수 상태 quantum Fisher 정보를 구한다. Bloch 접선·Cauchy–Schwarz로 classical 판독 정보의 경계를 유도하고 알려진 동작점에서 경계에 도달하는 축을 구성한다. 독립 반복·국소 불편성·양의 support 아래 분산 하한을 설명한다. 영 확률 없이도 판독의 일차 민감도가 사라지는 예제와, 같은 히스토그램의 두 각도로 국소 정보·전역 식별을 구별한다. 경계의0/0을 pseudocount로 처리하지 않는다.

## 높은 spin의 곡률과 반복되는 각도

`finite-spin-ladders-casimir-metric`는 기존 polynomial metric으로 +x coherent 준비의 이항 분포를 유도한다. 두 끝점 cat과 모멘트·생존 fringe·국소 곡률을 비교한다. cat의 더 큰 quantum 정보와 더 짧은 ray alias 간격을 함께 설명하고 j=0·j=1/2 예외를 구별한다. 평균 화살표·순수성만으로 민감도가 정해지지 않으며 단일 irreducible 표현의 계산을 다입자·고정 비용 실험 우위로 확대하지 않는다.

## 검증

- CI에 선언된 패키지 버전의 Python 3.12 격리 환경에서 관련 8파일·242검사가 통과했다. 한영 수식·코드·결과표가 같고 영어 원고에 한글 잔류가 없다. 연구 절을 제거한 원본문 hash와 풀이를 대조했다.
- 일반 회전 140준비, 정칙 판독 660회·경계 40회, 최적 접선 판독 120회와 정확 유리수 판독 27준비를 확인했다. 일반 행렬 exponential·밀도행렬 미분의 순수 상태 SLD와 Born/score 계산을 사용했다.
- 9개 spin 표현의 정규직교 제곱근 ladder에서 상태 모멘트 17준비·fringe 85준비, coherent raw metric·Jx 고유상태·cat ray alias·전체2π 부호를 대조했다. 부동소수점 비교는 방향성 구간 인증이 아니다.
- 빌드·타입·콘텐츠·정적 출력 검사가 실제 종료 코드 0으로 끝났다. 현재 한영 렌더링 4편과 제공된 2 HTML·4 SVG 바이트를 대조했다.
- 한영 320·768·1440px 밝은 화면과 1440px 어두운 화면 16상태, 개념/안내/퀴즈 각 6회, 저장·완료 복원 2그룹이 통과했다. 접근성 위반·페이지 오류가 없고 브라우저 종료를 확인했다.
- 그림 원본 4개와 정상 읽기 모바일 캡처 4개를 직접 검토했다. SVG/캡션 경계·목차 앵커·긴 코드/수식/표의 키보드 스크롤도 확인했다.

## 보존과 다음 순서

기존 spin 회전·측정·ladder·가중 좌표 조작을 유지한다. Fisher 정보·표본 추정기·coherent projector 판독은 독립 연구 예제이며 새 UI 제어로 표시하지 않는다. 이전 검토 원장 911항목을 보존하고 4편을 추가했다. 기존 650단원·1,969안내·51영역·509원주제·입문 362/362 완료를 유지한다.

양자 영역 순차 연구 대조는 **8/34단원**이다. 다음은 `finite-ensemble-density-gram-tomography`와 `qubit-density-spectrum-entropy-dephasing`이다. 전체 **45영역·447주제 대조 / 6영역·62주제 후속**은 유지한다. 영역 종합 대조·교육과정 완료는 false이며 공개 배포는 별도로 확인한다.
