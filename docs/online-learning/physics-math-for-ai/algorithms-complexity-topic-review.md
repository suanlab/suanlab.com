# 알고리즘·계산복잡도·계산 구조 원주제 대조

2026-11-12 기준. `algorithms-complexity`의 10개 원주제를 한영 원고·실습·독립 예제에 대조한다. 입문 48개 및 후속 입문 362/362 완료와 전체 51영역·509개 원주제의 연구 심화 상태를 구분한다.

## 원주제와 보강 내용

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`, 기존 계산은 `src/lib/learning-algorithms.ts`다.

| 원주제 | 단원 ID | 보강한 논증·독립 실행 |
|---|---|---|
| 점근 비용과 탐색 | search-asymptotic-costs | 순서 질의의 결정 트리 하한·정렬 전처리 상각·질의 수의 비용 가정 |
| 동적 계획법 | matrix-chain-dynamic-programming | 작업량·임시 메모리 Pareto 상태·평가 순서·용량 제약의 반례 |
| 그래프와 flow | residual-flow-cut-certificates | 조기 종료의 flow/cut gap 인증·역방향 잔여 호·Edmonds–Karp 보증 |
| 무작위·근사 | random-cuts-matching-approximation | 비음수 가중 cut의 조건부 기대값 탈무작위화·절반 보증 |
| 계산 가능성과 난해성 | sat-independent-set-reduction | SAT 결정 oracle로 증인 복원·질의 수·작은 완전탐색과 복잡도 구분 |
| 비교와 하한 | comparison-sorting-lower-bounds | 접두 transcript·Kraft·평균 비교 하한·항상 정답인 무작위 알고리즘 |
| 희소·병렬 계산 | sparse-work-span-scheduling | Work-conserving 일정의 Brent 경계·float 합산 순서 반례 |
| 메모리 계층 | cache-traffic-blocked-transpose | Cold write-allocate 전송 하한·LRU stack 성질·tile 변경의 범위 |
| 분산·통신 | distributed-gossip-communication | 대칭 행렬의 spectral 수렴·메시지 비용·방향 평균화의 가중 평균 |
| 이론·실측 비용 | operation-counts-wall-time | 7회 교대 측정 원자료·입력 provenance·중앙값 비율과 짝 비율의 차이 |

언어별 비교표와 표준 Python 블록 10개, 한영 work/span·gossip SVG 네 개를 추가했다. DAG 그림은 단위 작업 tick이며 실제 CPU 시간이 아니다. Gossip 그림은 정확한 유리수 모형의 누적 방향 메시지 비용과 전역 이탈 제곱합을 비교하며 실제 네트워크 지연이 아니다.

## 실습과 근거의 범위

기존 정수 비용·잔여 그래프·캐시·일정·합의 계산 모형은 유지한다. 메모리 제약 행렬 연쇄, 강제한 비최단 augment 경로, 가중 cut, 방향 그래프는 본문의 독립 예제이며 현재 조절기에 새 모형을 구현했다고 주장하지 않는다. 완전탐색은 작은 입력 검산이지 다항 시간 SAT 해법이 아니다.

실제 브라우저 benchmark에는 측정한 N·target·batch 반복 수와 7개 시행별 시간·선행 방법을 노출했다. 두 방법의 순서는 교대하며 기존 warmup·checksum·실제 0시간을 보존한다. N·target·반복 수가 바뀌면 결과를 숨기고 진행 중 측정을 취소한다. 추상 비용 가중치는 측정한 작업을 바꾸지 않으므로 원자료를 유지한다. 7회 batch를 독립 반복 실험이나 통계적 신뢰 구간으로 해석하지 않는다.

## 출처와 보존

배경과 모형 구분은 [MIT 6.006](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/), [6.046J](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/), [6.045J](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/), [6.172](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/)와 대조했다. 추가 Python·유리수·그림은 독립 구성으로 특정 논문이나 하드웨어 성능의 재현이 아니다.

스무 원고의 기존 본문 prefix·손계산 이후 suffix를 보존했다. Gossip은 기존 메시지 회계 절을 보존 경계로 삼는다. Immutable fixture 네 개는 수정하지 않고 이전 SHA·검토 SHA·이 기록을 reviewed ledger에 연결했다. 추가 표시 수식과 Python은 한영 대응을 검사한다.

## 검증

관련 계산·원고 보존 검사 71개와 타입·lint·콘텐츠 검증이 통과했다. 마지막 문장 교정 후 원고 보존 7개를 다시 통과했다. 한영 Python 20개 블록을 실행하고 추가 표시 수식·코드의 대응과 기존 본문 prefix·손계산 suffix 보존을 확인했다.

별도 정수·유리수 참조에서 7,776개 용량 조합의 최대 flow와 원래 제약을 대조했다. 행렬 연쇄 150개는 괄호의 완전탐색 비용과, DAG 512개는 독립 생성한 의존·깊이·일정과, 캐시 216개는 별도 LRU trace 및 cold 전송 하한과 대조했다. 합의 240개의 정확한 유리수 참조에 대한 최대 scaled error는 1.3×10⁻¹⁵ 미만이다. 이 유한 검산과 본문의 일반 증명은 범위가 다르다.

1,893개 경로 정적 빌드와 1,888개 HTML의 내부 링크·자산·fragment·sitemap·canonical 검사가 통과했다. 실제 export의 101개 상태는 한영 320/768/1440px 66개·영문 desktop dark 12개·경계 20개·실측 한영 모바일 2개·영역 연결 1개다. 접근성 위반·브라우저 오류는 0이며 상호작용 13묶음·안내 30개·퀴즈 20개·이전 계산 47개·영역 연결 10개를 확인했다.

실측의 7개 원자료와 중앙값·교대 순서·checksum, 입력 metadata, 비용 가중치 변경 시 원자료 유지, N·target·반복 수 변경 및 진행 중 취소를 검사했다. 정수 초안 blur 오류·모형 보존·Escape 복원, 표·코드·실측 원자료의 키보드 스크롤, 320px 그림 네 개와 한영 밝음/다크 실측표 캡처도 확인했다. 마지막 캡처 네 개의 화면 적합성은 위 접근성 상태 수와 별도다.

소스 1,882개·학습 원고 1,300개·릴리스 파일 31개·설정 8개의 후보 byte 일치와 SVG 네 개의 export 일치를 확인했다. 전체 CI 및 공개 배포 완료는 이 로컬 검증과 구분한다.

## 후속 범위

일반 메모리 제약 DP의 Pareto 복잡도, 대규모 flow와 bit 비용, 오차 허용 무작위 하한, 통신·스케줄러 오버헤드, 실제 캐시·NUMA·네트워크 및 반복 측정의 통계 설계는 추가 심화 범위다. 원주제 대조는 분야의 모든 연구 이론 완료를 뜻하지 않는다. 이번 대조로 누적 31영역·305개, 후속 20영역·204개이며 다음은 통계학습·일반화·강건성 11개다.
