# 제어·최적제어·강화학습 원주제 대조

2026-11-12 기준. `control-decision`의 원주제 11개를 등록 단원 13개에 대조했다. 입문 48개 및 후속 입문 362/362의 완료와 전체 51영역·509개 연구 심화 상태를 구분한다.

## 원주제와 보강 범위

각 ID의 한영 원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`에 있다. 기존 설명·연습·조절기·퀴즈를 유지하면서 손계산 절 앞에 연구 확장을 추가했다.

| 원주제 | 단원 ID | 보강한 논증·실행 예제 |
|---|---|---|
| 상태 공간 | state-space-coordinate-realizations | Markov 계수·Hankel 랭크·최소 실현·숨은 차원의 비식별 |
| 제어·관측 가능성 | reachability-observability-gramians | Gramian 좌표 변환·Hankel 특잇값·약한 채널 |
| Kalman | kalman-position-velocity | Innovation 보정 가정·우도·시점별 타원과 동시 포함 |
| Lyapunov | lyapunov-nonnormal-metric | 안정 모드의 불안정 전환·공통 계량 조건 |
| LQR·MPC | finite-horizon-lqr-box-mpc | 종단 불변 집합·shift 후보·가치 감소 증명 |
| Bellman | discounted-bellman-policy-values | 가치 오차·탐욕 정책 손실의 서로 다른 인증 |
| HJB·Pontryagin | continuous-hjb-pontryagin | 절댓값 종단 가치·비매끄러운 모서리·점성해의 범위 |
| MDP·POMDP | belief-mdp-sensing-pomdp | 센서 garbling·posterior martingale·무료 정보와 유료 관측 |
| Bandit·탐색 | finite-bayes-bandit-exploration | 완전 정보 상한·사전 평균 commitment 하한·정확한 유한 DP |
| Bandit·탐색 | seeded-bandit-regret-policies | 보상 잡음 martingale·고정 시점/동시 경계·seed의 한계 |
| Policy gradient·평가 | finite-policy-gradient-baselines | Performance-difference 망원합·잘못된 방문 분포 surrogate 반례 |
| Policy gradient·평가 | sequential-off-policy-weights | Doubly robust 소거 경로·Support·nuisance 의존성 |
| 제약·강건 제어 | one-step-constrained-robust-control | Predecessor 집합·불변 kernel·한 단계 안전의 한계 |

13개 독립 표준 Python 예제를 한영에 동일하게 넣어 모두 실행했다. 정확 분수·완전 열거를 우선 사용했고 Gaussian 꼬리·전환 행렬의 지수에는 부동소수점을 사용한다. HJB 모서리·강건 불변 집합 그림은 기존 색상의 한영 SVG 네 개다. 새 보조 예제를 기존 조절기의 모형이나 보증으로 혼동하지 않도록 명시했다.

## 계산 수정

Riccati 재귀는 나눗셈을 먼저 수행해 표현 가능한 subnormal 종단 계수·gain의 중간 곱 underflow를 피한다. 이전 전체 재귀는 `expandedP`·`expandedK`로 보존한다. 실제로 표현 불가능한 값까지 복구하지 않는다.

강건 제어의 무제약 입력은 soft-threshold 후 중심을 빼는 대신 동치인 인수분해 식으로 계산한다. 작은 양의 상태 페널티에서도 입력이 사라지지 않는다. Kink·경계 projection·0 페널티를 유지하고 이전 뺄셈을 `expandedUnconstrained`로 기록한다.

IS·PDIS 분산은 pairwise 항등식으로 보상 성분의 차이를 먼저 구한다. 로그 비율의 차에는 `expm1`, 제곱 합에는 공통 scale을 쓴다. `1+gamma`가 1로 반올림돼도 표현 가능한 Bernoulli 분산 `gamma²/4`를 보존한다. 이전 중심화 분산은 `expandedFullVariance`·`expandedPdisVariance`다. 구조적 Support 부재, raw ratio overflow, 표현 불가능한 분산은 계속 구분한다. 이 방식도 binary64 계산이며 arbitrary-precision 인증이 아니다.

## 출처·보존

제어·Bellman·HJB 배경은 [MIT Dynamic Programming](https://underactuated.mit.edu/dp.html)과 [MIT LQR](https://underactuated.mit.edu/lqr.html)의 기본 유도와 대조했다. 나머지 분야의 기존 원고별 primary reference를 유지한다. 추가 유리수 모형·반례·표·그림은 독립 구성이다. 특정 논문의 전체 알고리즘이나 일반 안정성 정리를 재현했다고 주장하지 않는다.

26개 원고의 이전 prefix와 손계산 절 이후 suffix를 byte 비교하고 이전 fixture 네 개를 수정하지 않았다. 검토 SHA와 이전 SHA는 `reviewed-learning-manuscripts.json`에 연결한다. 추가 한영 수식·표의 수치·코드 대응을 검사한다.

## 검증

- 제어 관련·원고 보존 테스트 **176개 통과**, 실패 0개. 추가 수치 회귀 테스트 네 개는 subnormal·작은 페널티·작은 분산·Support/overflow 경계를 포함한다.
- `npm run typecheck`, `npm run lint`, `npm run validate:content`, `npm run build`, `npm run check:export` 통과. **1,893개 정적 경로·1,888개 HTML**의 링크·자산·앵커·사이트맵 검사를 마쳤다.
- 브라우저는 13개 단원·교재 색인의 한영 320/768/1440px, 야간 화면과 경계 입력을 포함해 **129개 상태·15개 상호작용 그룹**을 확인했다. 새 표·코드의 키보드 가로 스크롤, 그림 로드·대체 텍스트, 언어 전환·저장·초기화·정수 초안·안내 예제·퀴즈·기존 계산을 검사했으며 접근성 위반·페이지 오류는 없었다.
- 13개 Python 예제의 한영 26개 블록을 실행하고 출력·코드·표시 수식 대응 및 기존 원고 prefix/suffix 보존을 확인했다. 독립 Decimal **650자리** 계산으로 **분산 60건·강건 입력 20건·Riccati 재귀 4건**을 대조했다.
- 한영 SVG 네 개의 렌더링과 320px 실제 export 화면을 확인했다. source·원고·설정·릴리스 파일과 후보 빌드의 byte 일치를 확인했다. 검증한 출력만 이전 출력 백업을 남겨 로컬 홈페이지에 설치한다.

이 목록은 이번 변경의 선별 검증이며 전체 저장소 CI 테스트 완료를 대신하지 않는다. 공개 반영은 push 이후 GitHub Pages 배포 상태로 구분한다.

## 후속 범위

일반 비선형·확률 MPC, 강건 invariant-set 알고리즘, 다변량 balanced reduction, 미지 잡음의 Kalman 진단, 비매끄러운 비교 원리의 일반 증명, POMDP 근사 오차, 정책 occupancy bound, cross-fitted sequential DR 및 적응적 동시 추론은 추가 심화 범위다. 원주제 대조가 제어 이론 전체의 완료는 아니다. 이번 기록은 누적 34영역·338개이며 이후 17영역·171개를 이어간다. 다음은 조합 최적화와 형식적 추론 9개다.
