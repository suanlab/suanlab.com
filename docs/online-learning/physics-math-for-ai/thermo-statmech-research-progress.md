# 열역학·평형 통계역학 연구 심화 진행표

2026-11-13. 이 영역에는 이미 **11원주제·32등록 단원·64한영 원고**가 있다. [기존 입문 종합 감사](thermodynamics-domain-audit-stage12.md)는 32단원 전체의 출발 근거다. 미집필로 다시 세거나 22단원으로 범위를 줄이지 않는다. 전체 교육과정의 연구 대조는 **43영역·426주제 완료 / 8영역·83주제 후속**이며 이 영역의 종합 대조는 미완료다.

## 첫 연구 보강: 평형 점유 확률을 이용한 온도 추정

기존 `thermodynamics`의 한영 본문에 연구 절을 추가했다. 고정 스펙트럼·알려진 축퇴도·오차 없는 독립 평형 준비라는 가정 아래 열용량과 로그 온도의 Fisher 정보가 같음을 유도한다. 알려진 간격의 내부 최대우도 추정과 유한 표본 경계, 간격까지 미지수일 때 정보 행렬의 영방향을 비교한다. g=1의 상대 민감도 최대점과 두 물리 척도의 공통 배율 모호성을 그림으로 설명한다.

한영 각각 표시 수식 10개·실행 표 1개·연구 해설 3개, 공개 그림 1개를 추가했다. 기존 본문과 여섯 해설은 보존했다. 기존 실습의 새 조작으로 소개하지 않는다. 정확도 하한은 표본별 신뢰구간이나 비선형 MLE의 불편성·달성 보장이 아니다. 절대 온도의 정보 최적화와 상대 온도 민감도의 최적화도 구별한다.

관련 **4파일·50검사**, lint·콘텐츠 1,885 URL·새 빌드 1,893경로·타입·정적 1,888 HTML 검사를 통과했다. 한영 320/768/1440px 라이트와 한영 1440px 다크 **8화면**, 펼친 안내 3개·해설 퀴즈 2개·개념 4개·저장/완료 1그룹을 확인했다. 페이지 오류·Axe 위반은 0이다. 원본 크기 그림 두 개와 모바일 캡처 두 개를 직접 검토했고, 현재 한영 렌더링과 실제 제공 HTML·SVG 바이트를 대조했다. 다른 31단원·62원고와 기존 보존 장부 801항목을 유지하고 명시한 두 수정만 추가해 803항목이다.

검증 결과는 [첫 보강 근거](thermometry-research-evidence.json)에 기록한다. 이 절만으로 11원주제 또는 32단원의 연구 심화가 끝났다고 집계하지 않는다.

## 등록 단원별 후속 순서

‘후속’은 기존 입문 내용이 없다는 뜻이 아니다. 원문·출처·독립 계산·Worker·학습 화면 근거를 먼저 읽고, 필요한 연구 설명과 예제를 보강한 뒤 실제 정적 reader를 검사한다. 아래 순서는 기존 단원 연결을 따르며 현재 첫 단원만 이번 연구 보강 검증에 포함한다.

| 단원 ID | 기존 대응 | 연구 보강 상태 |
|---|---|---|
| `thermodynamics` | 기존 입문 단원과의 연결 | 첫 보강 검증 |
| `finite-body-thermal-contact-heat-entropy` | 온도·열·일 | 기존 근거 대조 후 후속 |
| `ideal-gas-two-path-heat-work` | 온도·열·일 | 기존 근거 대조 후 후속 |
| `carnot-cycle-engine-refrigerator` | 열역학 법칙 | 기존 근거 대조 후 후속 |
| `two-reservoir-cycle-entropy-admissibility` | 열역학 법칙 | 기존 근거 대조 후 후속 |
| `gapped-spectrum-low-temperature-entropy` | 열역학 법칙 | 기존 근거 대조 후 후속 |
| `finite-spectrum-helmholtz-kl-variational` | entropy·free energy | 기존 근거 대조 후 후속 |
| `ideal-gas-helmholtz-gibbs-volume-availability` | entropy·free energy | 기존 근거 대조 후 후속 |
| `binary-shell-multiplicity-block-counts` | 미시·거시 상태 | 기존 근거 대조 후 후속 |
| `finite-macro-partition-conditional-entropy` | 미시·거시 상태 | 기존 근거 대조 후 후속 |
| `finite-noisy-macro-observation-channel` | 미시·거시 상태 | 기존 근거 대조 후 후속 |
| `finite-binary-reservoir-canonical-comparison` | ensemble | 기존 근거 대조 후 후속 |
| `canonical-binary-energy-fluctuations` | ensemble | 기존 근거 대조 후 후속 |
| `grand-canonical-finite-occupancy-fluctuations` | ensemble | 기존 근거 대조 후 후속 |
| `finite-spectrum-partition-cumulant-generators` | partition function | 기존 근거 대조 후 후속 |
| `coupled-binary-sites-partition-factorization` | partition function | 기존 근거 대조 후 후속 |
| `infinite-energy-ladder-normalization-cutoff` | partition function | 기존 근거 대조 후 후속 |
| `reference-measure-boltzmann-softmax-identifiability` | Boltzmann·Gibbs 분포 | 기존 근거 대조 후 후속 |
| `finite-gibbs-conditionals-scan-support` | Boltzmann·Gibbs 분포 | 기존 근거 대조 후 후속 |
| `quadratic-gaussian-gibbs-normalizability` | Boltzmann·Gibbs 분포 | 기존 근거 대조 후 후속 |
| `finite-curie-weiss-magnetization-symmetry` | 상전이·질서변수 | 기존 근거 대조 후 후속 |
| `curie-weiss-variational-stability-limits` | 상전이·질서변수 | 기존 근거 대조 후 후속 |
| `landau-sixth-order-coexistence-stability` | 상전이·질서변수 | 기존 근거 대조 후 후속 |
| `binary-connected-correlations-field-hessian` | 상관·fluctuation | 기존 근거 대조 후 후속 |
| `periodic-ising-ring-correlation-length` | 상관·fluctuation | 기존 근거 대조 후 후속 |
| `finite-correlated-average-ar1-uncertainty` | 상관·fluctuation | 기존 근거 대조 후 후속 |
| `stationary-flux-three-state-cycle` | 상세평형 | 기존 근거 대조 후 후속 |
| `metropolis-hastings-proposal-balance` | 상세평형 | 기존 근거 대조 후 후속 |
| `reversible-two-state-periodicity-mixing` | 상세평형 | 기존 근거 대조 후 후속 |
| `boundary-growth-extensivity-additivity` | 유한계와 열역학 극한 | 기존 근거 대조 후 후속 |
| `spatial-density-self-averaging-correlations` | 유한계와 열역학 극한 | 기존 근거 대조 후 후속 |
| `two-phase-partition-zeros-rounding` | 유한계와 열역학 극한 | 기존 근거 대조 후 후속 |

## 검증과 다음 단계

본문의 실행 예제와 한영 수식·표·그림 경로·원문 보존은 `node_modules/.bin/tsx --test scripts/learning-thermometry-research.test.ts`로 확인한다. 별도 미시 상태 열거·확률 미분·횟수 우도·경계 계산은 `python3 scripts/learning/verify-thermometry-research.py`로 재현한다. 정적 출력 후 `SUANLAB_TEST_BASE=http://127.0.0.1:43309 node scripts/check-thermometry-research.cjs`를 실행한다. 브라우저 주소에는 실제 preview 포트를 사용한다.

다음은 `finite-body-thermal-contact-heat-entropy`의 기존 원문과 독립 검산을 읽고, 온도 궤적에서 식별되는 열용량 비·완화율과 절대 열용량/연결 척도를 구분하는 연구 예제를 설계하는 것이다. 아직 그 설계·집필·화면 검증을 완료하지 않았다. 이후 나머지 등록 단원을 순서대로 대조하고 마지막에 영역 전체를 같은 출력에서 종합 검증한다.
