# 프라이버시·공정성·강건성 원주제 대조

2026-11-12 기준. `statistical-privacy`의 원주제 10개와 한영 단원 16개를 대조한다. 입문 48개 및 후속 362/362의 완료는 전체 51영역·509개 원주제의 연구 심화와 구분한다.

## 원주제와 보강 범위

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`다. [기존 입문 감사](statistical-privacy-topic-audit.md)의 대응 관계와 본문·실습·연습 해설을 유지하고 연습 절 앞에 연구 확장을 추가했다.

| 원주제 | 단원 ID | 추가 논증·독립 실행 |
|---|---|---|
| DP 정의 | finite-events-differential-privacy | 무작위 검정의 네 부등식·균형 오류 하한·달성 채널 |
| epsilon·delta·구성 | fresh-copy-privacy-composition | Rényi 모멘트의 곱·복사 불변성·충분한 DP 변환과 정확 프로필의 차이 |
| 민감도·잡음 | bounded-sensitivity-laplace | 결합 민감도와 좌표 예산 배분의 엄격 볼록 최적성 |
| 민감도·잡음 | gaussian-sensitivity-calibration | 공분산 백색화·Mahalanobis 거리·Gaussian 모멘트 |
| 정보 누출·추론 | posterior-odds-privacy-inference | 후처리 사후의 조건부 평균·정보와 결정 위험의 비엄격 비교 |
| fairness·식별 | fairness-metrics-selection-identification | 포함 확률의 상대 오차·조건부 비율의 sharp 구간·역방향 변환 |
| 지표 양립 | binary-parity-compatibility | 근사 equalized odds와 결정률 차이가 제한하는 구분 능력 |
| 지표 양립 | calibration-class-balance | 정확 보정·근사 클래스 균형에서 Brier 위험의 정량적 상한 |
| 강건 통계 | replacement-contamination-location | 관측 동치인 오염 세계·모든 추정량의 절대/제곱 손실 하한 |
| 강건 통계 | influence-and-breakdown | 구간 기울기 조건·유한 편향·수치 잔차의 위치 오차 변환 |
| 적대적 섭동 | linear-norm-adversarial-margins | 타원체 위협의 달성 방향·좌표 변환 불변성 |
| 적대적 섭동 | relu-attacks-and-certificates | Lipschitz 덮개·구간 분할의 수렴 인증과 차원 비용 |
| 분포 강건성 | kl-ambiguity-risk-envelope | 작은 반경의 분산·삼차 cumulant 전개·80자리 독립 계산 |
| 분포 강건성 | finite-wasserstein-risk-certificates | 연속 도착지의 격자 반올림·비용 여유·손실 근사 경계 |
| 보장·현실 평가 | stratified-evaluation-shift-bounds | 모든 집단·표본 크기의 합산 예산·적응적 관측·정확 경로 계산 |
| 보장·현실 평가 | selection-aware-benchmark-coverage | 새 평가의 조건부 포함·무한 단계 예산·평가 입력 암기 반례 |

16개 표준 Python 예제를 한영에 동일하게 추가했다. 분수·표본 경로의 동적 계산·고정밀 Decimal을 사용한다. DP 검정 영역과 순차 영오류 상한은 한영 SVG 네 개로 비교한다. 그림의 곡선은 명시한 수학 모형이며 경험 자료나 새로운 native 조절기의 구현을 뜻하지 않는다.

## 가정과 보장 범위

- DP 검정 하한은 해당 인접 쌍·균형 사전의 결과다. Rényi 변환은 양방향 모멘트 보장이 필요하고 정확 프로필과 다르다. XOR의 분리 지지에 유한 모멘트를 적용하지 않는다.
- 벡터 Laplace의 tight 합은 동시에 달성 가능한 결합 민감도에 한정한다. Gaussian 공분산은 공개·입력 독립·양의 정부호다. 특이 공분산과 기계 sampler는 별도 문제다.
- 포함 확률의 상대 오차 구간은 외부 가정이다. 자체 신뢰수준이나 표집 불편성을 만들지 않는다. 공정성의 정량적 Brier 상한에는 정확한 보정과 내부 기저율이 필요하다.
- 오염 하한의 이진 분포 클래스는 고정 표본 교체·균등 기준과 다른 보조 실험이다. Huber 잔차를 오차로 바꾸려면 해당 구간의 기울기 하한과 근사해 위치를 함께 확인한다.
- 타원체는 좌표와 함께 변환한다. ReLU 분할은 기존 네트워크·위협을 바꾸지 않으며 고차원의 효율적 인증기로 주장하지 않는다.
- KL 전개는 고정된 양의 분산에서 국소적이다. 작은 두 반경은 native 격자 밖의 실행 예제다. Wasserstein 격자 경계는 손실의 Lipschitz 상수와 반올림 후 비용 증가를 포함한다.
- 순차 보장은 고정 예측기의 집단 내 iid 스트림과 모든 크기의 예산 합을 사용한다. 적응적 모형은 과거에 조건부로 유효한 새 평가를 요구한다. 배치 재사용·배치 내 중단을 고정 표본 구간으로 인증하지 않는다.

## 출처와 후속 연구

기존 원고의 원문 출처를 유지한다. 추가 배경은 [Dwork·Roth 교재](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf), [Mironov의 Rényi DP](https://arxiv.org/abs/1702.07476), [Kleinberg 등의 점수 양립 연구](https://arxiv.org/abs/1609.05807), [Howard 등의 confidence sequence](https://arxiv.org/abs/1810.08240)다. 본문의 상수·분포·격자·조건부 예산은 직접 유도하고 별도 코드로 검산했다. 이 예제를 각 논문의 전체 알고리즘·최적 상수·실증 재현으로 보고하지 않는다.

표본 추출을 포함한 실제 학습 accountant, 구조적·인과적 공정성, 미지 척도·고차원 강건 추정, 큰 네트워크 인증, 데이터로 정당화한 모호 집합과 의존 표본의 순차 평가는 계속할 연구 범위다. 10개 원주제 대조를 해당 분야의 모든 연구 완료로 해석하지 않는다.

## 검증과 보존

- 프라이버시·공정성·강건성·원고 보존 테스트 **197개 통과**, 실패 0개. 기존 계산 모델과 기존 개별 브라우저 스크립트는 변경하지 않았다.
- `npm run typecheck`, `npm run lint`, `npm run validate:content`, `npm run build`, `npm run check:export` 통과. **1,893개 정적 경로·1,888개 HTML**의 내부 링크·자산·앵커·사이트맵을 검사했다.
- `scripts/check-privacy-research.cjs`를 `SUANLAB_TEST_BASE`의 최종 export에서 실행했다. `SUANLAB_PRIVACY_RESEARCH_PART=0/2,1/2`의 두 분할은 16개 ID를 겹침 없이 모두 다룬다. 한영 320/768/1440px·야간·최소/최대 입력·교육과정 검색을 포함한 **152개 상태·17개 상호작용 그룹·48개 안내 예제** 통과. 새 표·코드의 키보드 스크롤, SVG 로드·대체 텍스트, 저장·언어·정수 초안·초기화·퀴즈·완료 표시를 확인했다. 접근성 위반·페이지 오류는 없었다.
- 안내 예제·경계의 UI 수치는 독립 수학 테스트를 통과한 source 모델과 대조한다. 이 브라우저 대조를 수학의 별도 독립 증명으로 세지 않는다.
- 16개 독립 Python 예제의 한영 32개 블록을 실행하고 코드·표시 수식·출력 대응과 기존 prefix/suffix 보존을 확인했다. KL 작은 반경은 Decimal 80자리, 순차 실패는 400개 크기의 정확 분수 경로 전파, 평가 재사용은 1,000개 입력 삼중항으로 확인했다. 기존 검토 해시 527개는 유지하고 32개를 추가했다.
- 한영 SVG 네 개와 320px export 화면을 확인했다. source·원고·설정·릴리스 파일 및 SVG export의 byte 일치를 검사했다. 검증 출력은 이전 출력 백업과 함께 로컬 홈페이지에 설치한다.

이번 변경의 선별 검증이며 전체 저장소 CI 완료를 뜻하지 않는다. 공개 홈페이지 반영은 push 후 GitHub Pages 배포 상태로 별도 확인한다.
