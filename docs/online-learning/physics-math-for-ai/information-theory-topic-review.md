# 정보이론 원주제 대조와 보존·부호·유한 블록·가중치 심화

2026-11-11에 정보이론과 압축의 10개 원주제를 한영 11개 reader, `learning-information.ts`·`learning-foundations.ts`와 대조했다. 데이터 처리의 등호, 접두 부호 구성, 유한 블록 converse·복원점 열거, Rényi와 가중치 모멘트를 보강했다. 작은 양의 상호정보량이 엔트로피 뺄셈에서 사라지는 실제 수치 문제도 수정했다. 전체 51영역의 일반 이론과 연구 심화 완료를 뜻하지 않는다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/`의 각 ID에 해당하는 `.ko.md`와 `.en.md`다.

| 원주제 | 원고 ID | 확인한 정의·가정·논증 |
|---|---|---|
| 자기정보·entropy | self-information-entropy | 실현 놀람과 평균, 영 질량의 무한 놀람·0 기여, 유한 정보원 |
| 조건부 entropy·상호정보량 | conditional-entropy-mutual-information | 결합·조건부 법칙, 반전·독립·null 열, 작은 양의 값의 KL 평균 계산 |
| cross entropy·KL | information-theory, cross-entropy-kl | bits·nats, 지지와 clipping, 모집단·경험 점수, 유한 그림과 전체 값 |
| Jensen 부등식 | jensen-information-inequalities | 엄밀 볼록성·등호, P 지지 밖 Q 질량을 포함한 KL 비음수성 |
| 데이터 처리 | data-processing-information | Markov·연쇄법칙, 사후 KL 간격과 등호, 비가역 보존·손실 병합 |
| 부호화·압축·MDL | coding-compression-mdl | prefix·모호성·헤더, Kraft·KL 하한, Shannon 구성과 블록 정수화 비용 |
| rate-distortion | rate-distortion-binary | 평균 Hamming·시험 채널, 유한 메시지 converse, 2·3·4비트 완전 열거 |
| Fano 경계 | fano-error-bounds | 오류 사건의 연쇄법칙, 날카로운·느슨한 경계, 지정·MAP 디코더 차이 |
| Rényi 정보량 | renyi-information | 차수·지지·극한, D₂와 가중치 두 번째 모멘트, 곱 법칙·유한 ESS |
| bits·nats와 정규화 | information-units-normalization | softmax·공통 이동, 밀도 면적·좌표 척도, 균등 빈의 정확 항등식 |

## 보강한 논증과 실행 예제

유한 Markov 사슬에서 정보 손실을 사후 KL 평균으로 나타냈다. 활성 (y,z)마다 P(X|y)=P(X|z)일 때만 등호이며 원시 Y의 역함수는 필요 없다. Y 질량 (1/4,1/4,1/2)와 성공 사후 (1/5,1/5,4/5)의 보존·손실 병합을 비교했다. retained information은 약 0.278072와 0.091305 bits이며 손실 병합의 간격은 약 0.186767이다. 고정 결합분포의 등호를 모든 모수·사전에 대한 충분성으로 바꾸지 않았다. 한영 벡터 그림은 두 병합을 같은 사후 수치로 보여준다.

Kraft 합 K와 정규화한 r_i=2^(−ℓ_i)/K에서 L−H=KL(P||r)−log₂K를 유도했다. 정수 Shannon 길이를 canonical prefix 단어로 실제 구성하고, 알려진 iid 분포의 m블록에서 H≤L_m/m<H+1/m을 증명했다. p=(1/2,1/3,1/6), m=1,2,4에서 기호당 길이는 5/3,5/3,305/192다. 모든 두 블록 메시지의 복호화를 검사한다. 모형·길이의 전달 비용과 블록 알파벳의 증가를 숨기지 않는다.

공정 iid 비트·고정 b bit 메시지의 Markov 조건 아래 오류 벡터 E^n을 사용해 b/n≥1−h₂(D)를 유도했다. 평균·최악 왜곡을 구분하고 하한과 달성 정리를 분리했다. 두 복원점의 n=2,3,4 후보 6·28·120개와 모든 원본을 열거하여 최적 평균 왜곡 1/4,1/4,5/16을 구했다. 작은 부호의 양의 경계 간격을 점근 경계와 혼동하지 않는다.

정규화 P,Q와 w=p/q에서 E_Q[w²]=2^D₂, 정규화 상수 추정 분산 (2^D₂−1)/N을 유도했다. 고정 분포에서 표본 ESS/N→2^(−D₂)이고 이를 유한 N의 기대 ESS로 읽지 않도록 했다. 독립 d좌표에서 같은 불일치를 반복하는 예제는 E[w²]=(34/25)^d다. d=1,N=2의 정확 기대 ESS는 59/34로, 모집단 계수에 N을 곱한 25/17과 다르다. 일반 함수 추정 오차와 가중치 모멘트, 차원 자체와 반복 불일치를 구분했다.

## 수치 수정

기존 `bernoulliChannelInformation`은 H(Y)−h₂(ε)를 빼서 p=10⁻³⁰⁰,ε=.1과 p=.5,ε=.5+2⁻⁴⁰에서 양의 MI를 0으로 반환했다. 조건부 채널과 출력 법칙의 KL 평균으로 바꾸었다. 가까운 확률의 KL에는 (1+x)log1p(x)−x의 이차항부터 계산하는 급수를 사용한다. 조건부·주변 확률의 차이는 −p(1−2ε), (1−p)(1−2ε)로 먼저 계산해 주변 확률 반올림에서도 작은 차이를 잃지 않게 했다.

입력·출력 비트 재명명의 정보 불변성으로 p와 ε를 각각 0.5 이하 방향으로 계산해 1에 가까운 주변 질량의 여사건 반올림도 줄인다. 독립 ε=.5·결정적 p·완전 반전의 기존 정확 경계는 유지한다. 희귀 입력의 I/p 극한과 균형·불균형 입력의 거의 공정한 잡음에서의 이차 계수를 새 회귀 검사로 확인한다. bits 변환은 희귀 가중치 곱보다 먼저 적용해 표현 가능한 subnormal 결과의 중간 underflow를 줄였다. 합성 채널에서는 대비 (1−2ε₁)(1−2ε₂)를 유지해 유효 반전의 반올림과 분리했다. 수학적으로 양수인 선택 정보가 underflow하면 —·수치 범위 안내를 표시하고 선택 점을 보류한다. 정보 손실의 뺄셈과 다른 엔트로피 값은 여전히 유한 정밀도 계산이며 모든 작은 차이의 정량 인증이나 임의 정밀도를 보장하는 구현은 아니다. 이 수치 변경을 원고에도 설명했다.

## 출처와 후속 범위

- [MIT 6.441 Chapter 3 §3.1–3.2](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/486b7d83cc428acf75b74323ecafcc11_MIT6_441S16_chapter_3.pdf): 충분성·데이터 처리·조건부 KL 표현의 배경. 세 상태 수치는 직접 구성했다.
- [MIT Chapter 6, Theorems 6.5–6.6](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/67d8e68cd8fda55366e3f9f0a9465c17_MIT6_441S16_chapter_6.pdf): Kraft·Shannon 평균 길이 배경. 세 질량의 블록 코드는 직접 구성했다.
- [MIT Chapter 23, Theorem 23.2](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/880c39878ba7dc35a5fbb35758d42a69_MIT6_441S16_chapter_23.pdf): 고정 길이·평균 왜곡 converse 배경. 오류 벡터 증명과 작은 부호 열거는 직접 전개했다.
- [van Erven–Harremoës, §I-B·III-G](https://arxiv.org/html/1206.2459v2): Rényi의 χ² 관계와 곱 분포의 배경. 가중치·ESS 분수는 직접 계산했다.
- [Shannon (1948)](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf): 원래 정보·부호화 문제의 배경.

일반 알파벳의 측도론적 정보량, 연속·비정상 정보원의 부호화, 보편·정교한 MDL, 유한 블록의 정량 달성·분산 경계, 일반 Rényi 상호정보량과 실제 표현 압축·의미 왜곡 벤치마크는 후속 연구 심화다. 10개 원주제 대조를 전체 이론의 완료로 계산하지 않는다.

## 검증 기록

- `learning-information.test.ts` 36개와 `learning-manuscript-preservation.test.ts` 7개, 기존 기초 정보량 계산 1개, 연결된 11개 reader의 한영 원고·수식 검사 11개가 통과했다. 합계 55개이며 선택 실행에서 제외한 검사는 통과 수에 넣지 않았다. 모바일 표 제목과 수치 설명을 정리한 뒤 영향받은 원고 렌더링·보존도 재확인했다.
- 네 예제의 한영 Python 블록 8개를 실행했다. 최종 원고의 코드 해시와 실행 기록이 일치한다. 병합 사후 KL, canonical 부호와 모든 두 블록 메시지 복호화, 모든 복원점 쌍·원본, 유한 ESS의 분수 계산을 확인했다.
- 숫자 입력의 정확한 binary64 값을 `Decimal.from_float`로 읽고 750자리 결합분포 MI를 독립 계산해 81개 입력을 대조했다. 기준과의 차이는 상대 10⁻¹²+8ulp 허용 안이며 관측한 최대 상대차는 약 2.63×10⁻¹⁶이다. 이 사례 검사를 모든 입력의 오차 인증으로 일반화하지 않는다. 큰 p의 여사건과 subnormal 경계에서 발견한 차이는 비트 재명명과 bits 변환 순서로 수정하고 회귀 검사에 추가했다.
- 한영 320px 원고의 새 표·병합 그림을 확인했고, 최종 정적 페이지의 한영 그림도 확인했다. 두 SVG의 빌드된 바이트는 원본과 일치한다.
- Node 20에서 `npm run build`, `npm run typecheck`, `npm run validate:content`, `npm run check:export`가 통과했다. 1,893개 경로를 생성했고 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 검사했다. 1,300개 한영 원고와 1,882개 `src/` 파일·생산 설정이 빌드 사본과 일치한다.
- 수정한 수치 경계·기존 정보량·정수 입력의 실제 페이지 기능을 먼저 통과한 뒤, `SUANLAB_TEST_BASE=http://127.0.0.1:43299 npm run check:information` 전체를 통과했다. 11개 reader와 교재 화면의 한영 320/768/1440px·영어 다크 모드·특수 경계 등 89개 상태, 13개 상호작용 묶음이다. 33개 안내 예제·22개 확인 문제, 입력 초안·Escape·초기화, 저장·언어 전환·완료 상태, 목차·검색·그림 로딩과 필요한 키보드 스크롤을 확인했다. 희귀 입력의 양의 정보, 합성 반전 0.5 표시와 양의 정보의 공존, 표현 범위 밖의 —·안내도 확인했다. 자동 접근성 위반과 JavaScript 오류는 없었다.

원시 기록은 git에서 제외한 `.runtime/learning/information-*.log`, `information-browser.json`, `information-research-python.json`, `information-decimal-reference.json`에 있다. 고정밀 대조 스크립트는 `.runtime/learning/check-information-decimal.py`다. 전체 테스트 모음과 공개 배포 성공 여부는 GitHub Actions 결과로 별도 확인한다. 이번 선택 검사를 전체 모음의 성공으로 보고하지 않는다.
