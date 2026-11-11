# 게임이론과 다중 행위자 학습 원주제 대조

2026-11-12 기준. `game-theory`의 10개 원주제와 한영 11개 단원을 대조한다. 요청 당시 입문 48개와 후속 362/362의 집필 완료는 전체 51영역·509개 원주제의 연구 심화와 별도다.

## 원주제와 보강 범위

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`다. 기존 본문·실습·연습 해설을 유지하고 연습 절 앞에 연구 확장을 추가했다.

| 원주제 | 단원 ID | 추가 논증과 독립 실행 |
|---|---|---|
| 전략형 게임·혼합 전략 | normal-form-mixed-strategies | 양의 배율·상대 행동별 상수가 Nash 집합을 보존해도 후생 순위를 보존하지 않는 예 |
| Nash·최적 반응 | nash-best-response-sets | 균일 보수 오차 δ에서 ε+2δ 인증과 sharp 반례 |
| 영합·minimax | zero-sum-minimax-bounds | 값의 δ 안정성과 쌍대 gap의 2δ 오차를 별도 증명 |
| 반복 게임 | repeated-game-regret-comparators | 공적 관측 오류가 있는 무한 trigger 전략의 Bellman 식·유인 임계값 |
| 후회 최소화 | hedge-no-regret-learning | 외부 후회와 경험 공동 법칙의 CCE, 주변분포 곱과의 차이 |
| 협력 게임·Markov 게임 | coalition-shapley-core | 균형 가중치·LP 쌍대·core 존재 조건과 비음수 least-core |
| 협력 게임·Markov 게임 | finite-markov-team-values | 전 상태 최적 반응의 비동기 엄격 개선·유한 종료와 동시 갱신 순환 |
| 평균장 게임 | mean-field-forward-backward-consistency | 독립 private draw·정확한 집단 구성에서 유한 N 기대 유인 상한과 이항 정확합 |
| 메커니즘 설계 | direct-mechanisms-truth-payments | 연속 단일 매개변수 DSIC의 단조성·지급 적분식·공공재 적자 |
| 적대적 학습 | bilinear-adversarial-gradient-dynamics | 강볼록·강오목 전역 gap과 잔차, 순수 쌍선형 비유계/제한 상자의 구분 |
| 고정점·존재 | fixed-point-existence-versus-iteration | 두 행동 Nash 사상의 잔차와 gain의 제곱근 관계·감쇠 계수 |

11개 표준 Python 예제를 한영에 동일하게 추가했다. 정확 분수·독립 최적 반응·선 교점·이항 합으로 본문의 수치 예를 확인한다. 공동 법칙과 주변 곱, 유한 인구 유인과 상한은 한영 SVG 네 개로 비교한다. 그림은 명시한 보조 모형의 계산이며 경험 데이터나 기존 실습의 새 조절기를 뜻하지 않는다.

## 가정과 해석의 경계

- 공적 관측 결과는 명시한 흡수적 처벌 전략의 perfect public equilibrium이다. 특정 전략의 실패를 모든 협력의 불가능성으로 확대하지 않는다.
- CCE는 사전에 고정한 일방 이탈을 비교한다. 조건부 추천 이탈을 비교하는 CE 및 독립 Nash와 구별한다. 예시 경로를 native Hedge 실행으로 주장하지 않는다.
- Markov 종료 증명은 유한 결정적 정상 정책·공통 보상·전 상태 최적 반응·전 지지 시작분포·엄격한 비동기 개선에 한정된다. 팀 최적성이나 동시 갱신 수렴은 보장하지 않는다.
- 평균장 상한은 타인의 실현 행동을 보기 전 이탈하는 기대 유인이다. 표본 경로 보장·일반 PDE의 수렴률이 아니다.
- 지급식은 연속 보고형의 기대 효용 DSIC다. 유한 보고 격자만으로 필요성을 주장하거나 비용 회수를 보장하지 않는다.
- μ=0에서 전역 gap의 무한대는 결합 a≠0·비영점 상태에 한정한다. μ=a=0인 평탄한 게임을 제외한다. 화면의 축 범위를 행동 제약으로 간주하지 않는다.
- 고정점 근사 인증은 명시한 보수 차이 상한을 사용한다. 감쇠로 잔차만 줄여 현재 전략의 유인이 개선됐다고 주장하지 않는다.

## 출처와 후속 범위

기존 원고의 원문 출처를 유지한다. 추가 배경은 [Cornell의 CCE·후회 강의](https://www.cs.cornell.edu/courses/cs6840/2020sp/note/lec16.pdf), [Roughgarden의 지급식 강의](https://timroughgarden.org/f13/l/l3.pdf), [Serrano의 core 해설](https://economics.brown.edu/sites/g/files/dprerj726/files/papers/2007-11_paper.pdf), [Cardaliaguet의 평균장 게임 노트](https://www.ceremade.dauphine.fr/~cardaliaguet/MFG20130420.pdf)다. 본문의 유한 예·상수 계산은 직접 유도하고 독립 코드로 검산했다. 일반 신호 구조의 반복 게임·연속 상태 확률 게임·일반 평균장 PDE·다차원 유형 설계는 후속 범위다.

## 검증과 보존

- 게임이론·원고 보존 테스트 **223개 통과**, 실패 0개. 기존 계산 모형은 변경하지 않았다.
- `npm run typecheck`, `npm run lint`, `npm run validate:content`, `npm run build`, `npm run check:export` 통과. **1,893개 정적 경로·1,888개 HTML**의 링크·자산·앵커·사이트맵을 검사했다.
- 한영 320/768/1440px·야간·경계·설정·퀴즈를 포함한 브라우저 **173개 상태·13개 상호작용 그룹·33개 안내 예제** 통과. 새 표·코드 키보드 스크롤과 SVG 로드·대체 텍스트, 언어 전환·정수 초안·저장·초기화·완료 표시를 확인했다. 접근성 위반·페이지 오류는 없었다.
- 11개 독립 Python 예제의 한영 22개 블록을 실행하고 코드·표시 수식·출력 대응과 기존 prefix/suffix 보존을 확인했다. 기존 검토 해시 505개를 유지하고 새 원고 22개를 등록했다.
- 한영 SVG 네 개와 320px export 화면을 확인했다. source·원고·설정·릴리스 파일·SVG export의 byte 일치를 검사했다. 검증 출력은 기존 출력 백업과 함께 로컬 홈페이지에 설치한다.

이는 이번 변경의 선별 검증이며 전체 저장소 CI 완료를 뜻하지 않는다. 공개 홈페이지 반영은 push 후 GitHub Pages 배포 결과로 별도 확인한다.
