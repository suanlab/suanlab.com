# 부호·계산과 알고리즘 정보 원주제 대조

2026-11-11 기준. `coding-algorithmic-information`의 9개 원주제를 한영 원고와 `learning-coding.ts`에 대조했다. 기존 입문을 보존하면서 오류 보장·부호 비용·계산 가능성·난수의 양화 순서를 보강했다. 기록 범위는 17영역·166개 원주제이며 전체 51영역·509개 일반 이론과 연구 심화의 완료를 뜻하지 않는다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{ko,en}.md`이다.

| 원주제 | ID | 정의·가정·보강 근거 |
|---|---|---|
| 채널·용량 | binary-channel-capacity | BSC/BEC의 공정 입력 최적성, 무피드백 DMC의 Fano 약한 역정리, 작은 정보량 |
| 오류정정부호 | repetition-error-correction | 거리 공의 분리, 독립 Bernoulli의 최적 Chernoff 상계와 공통 반전 반례 |
| 유한체·선형 부호 | hamming-linear-codes | H의 rank·coset·완전 덮기, 오류 가중치별 성공, 동일 payload의 블록 비교 |
| 압축·복원 | lossless-compression-reconstruction | 외부 N의 prefix 해독, run 조합 수·Kraft, 균등 원본의 절약 확률·전수 복원 |
| algorithmic complexity | finite-description-length | 보편 모사의 가산 상수·계산 가능한 전단사, 유한 프로그램 질량과 최단 설명 |
| Kolmogorov 계산 불가능성 | kolmogorov-uncomputability | gamma 정수 부호의 실제 길이·대각 논증, 수렴 상한과 종료 인증, plain/prefix 구분 |
| MDL·모형 선택 | two-part-mdl-selection | 정수 prefix 길이와 이상 길이, 두 부분·mixture, 유한 Bernoulli NML의 minimax regret |
| 계산 제한·근사 | budgeted-description-search | 더 짧은 후보의 완료 인증, 길이·시간 동시 제한과 유한 oracle의 범위 |
| 통신 복잡도·분산 표현 | communication-fingerprints | GF(2) kernel·rank, 고정 쌍과 동시 보장의 다른 양화, 2×2 전수조사·full rank |

## 독립 예제와 그림

Python 8종을 한영에서 같은 코드로 실행한다. 반복 부호는 ε=.1,n=3·5·9의 정확 꼬리와 Chernoff 상계를 비교한다. Hamming은 16개의 반경 1 공이 128개 수신열을 중복 없이 덮는지 검사한다. 네 비트 payload 기준 Hamming 7회 사용과 반복 12회 사용의 오류를 비교하며 전송 비용을 같다고 부르지 않는다.

Raw/RLE는 N=8·16의 모든 256·65,536개 입력을 정확히 복원하고 길이순 prefix 충돌·Kraft 합을 검사한다. 유한 D는 278개 명령의 질량 15/16을 출력별로 합한다. Gamma는 양의 정수 1~4095의 복호와 정확 Kraft 합 4095/4096을 확인한다. 이 실험들이 보편 K를 계산한다고 주장하지 않는다.

MDL은 N=16,K=4의 두 부분 14.980450과 mixture 14.592679 ideal bits를 비교한다. Bernoulli NML은 N=4·8·16에서 정확분수 정규화와 모든 count의 같은 regret 비율을 확인한다. NML은 화면의 네 모형 선택기에 추가 구현한 제어가 아니라 별도 실행 예제다.

탐색 예제는 어떤 후보를 방문했는지로 충분한 최적성 인증을 구분한다. 지문은 균등 2×2 행렬 16개의 rank 개수 1·9·6과 고정 비영 차이 충돌 4/16, 어떤 비영 차이의 충돌 10/16을 구분한다. 8×8의 full-rank 확률은 약 .289919다. 한영 SVG 두 종류가 Hamming의 덮기와 지문의 양화 차이를 설명한다.

## 수치 수정

채널의 엔트로피 차는 거의 독립인 ε=.5+2⁻⁴⁰에서 양의 용량을 0으로 잃었고 p=10⁻³⁰⁰,ε=.1의 작은 정보량도 잃었다. `learning-information.ts`의 안정적인 Bernoulli 채널 KL과 log1p 엔트로피를 재사용한다. 수학적으로 양수지만 표현 범위 아래인 선택 정보량은 —·안내와 점 보류로 처리한다. BSC의 독립·상수 입력, BEC의 완전 소거에서의 정확한 0과 구분한다. 표시 막대의 반올림은 그 모형의 정확 독립성을 증명하지 않는다.

반복 n=9,ε=10⁻¹⁰⁰의 양의 실패·선택 수신 확률이 0으로 보이던 경우도 —·안내로 구분했다. ε=0의 실제 영 확률은 유지한다.

## 출처와 후속 범위

- [MIT 6.441 Lecture 8](https://ocw.mit.edu/courses/6-441-information-theory-spring-2010/8a41ad57fef4487df71ea164635e1248_MIT6_441S10_lec08.pdf): DMC의 n사용 정보 상계·BSC/BEC. 기존 Fano 단원과 연결해 약한 역정리 식을 직접 전개했다.
- [MIT MAS.160 Error-Correcting Codes](https://ocw.mit.edu/courses/mas-160-signals-systems-and-information-for-media-technology-fall-2007/94144b6cc83df4bea4f9cef22c91baf4_1017_error.pdf): 거리·Hamming 배경. 완전 덮기와 같은 payload 비교는 직접 계산했다.
- [Grünwald–Vitányi, §§2·4](https://arxiv.org/html/0809.2754): prefix 설명·모사 배경. 유한 D의 질량 계산은 보편 coding theorem의 구현과 구분한다.
- [Vitányi (2020), §§1·2](https://homepages.cwi.nl/~paulv/papers/incomp.pdf): plain C의 비계산성·상한 근사 배경. prefix K에는 gamma·literal의 자기 구분 비용을 포함해 적용했다.
- [Grünwald–Roos, §2](https://arxiv.org/html/1908.08484): mixture·NML·regret 배경. 유한 Bernoulli 정규화와 하한을 직접 전개했다.
- [Roughgarden, Lecture 3 §1](https://theory.stanford.edu/~tim/w15/l/l3.pdf): 고정 쌍의 public-coin equality. rank 분포·동시 보장 격자는 직접 열거했다.

일반 채널의 달성 가능성·유한 블록 최적 경계, 큰 부호의 복호·LDPC/polar 설계, 보편 알고리즘 확률의 coding theorem·무작위성, 특이/연속 모형의 refined MDL, 일반 private-coin·상호작용 통신 하한은 후속 심화다. 입문 완료와 원주제 대조를 이 전체 연구 범위의 완료로 바꾸어 계산하지 않는다.

## 검증 기록

- 계산 검사 26개와 기존 원고 보존 검사 7개, 선택한 한영 reader 렌더링 검사 9개를 통과했다. 전체 테스트 모음의 통과로 확대하지 않는다.
- Python 코드 16개를 재실행하고 최종 코드 SHA·출력을 대조했다. 9개 단원의 한영 절 수·표시 수식·실행 코드가 일치한다. 320px에서 새 표 16개와 최종 SVG 4개의 배치를 확인했다.
- 채널 15개 입력의 정보량·용량을 정확분수 결합과 450자리 Decimal 로그로 독립 검산했다. 극소 반복 오류의 정확분수 꼬리도 양수이며 binary64 최소 양수의 반 미만임을 확인했다.
- 실제 정적 빌드의 경계 선행 검사는 11개 상태·4개 동작 그룹이 통과했다. `npm run check:coding` 최종 실행은 한영·320/768/1440px·다크 모드의 84개 접근성·배치 상태, 11개 동작 그룹, 안내 예제 27개와 퀴즈 18개를 통과했다. 표·코드의 키보드 스크롤, 잘못된 정수 draft·Escape 복구, 저장·언어 전환·초기화와 기존 계산 회귀를 포함한다.
- `npm run build`, `npm run typecheck`, `npm run validate:content`, `npm run check:export`가 통과했다. 1,893개 빌드 경로와 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 확인했다.
- 소스 1,882개와 한영 원고 1,300개, 설정·릴리스 파일·내보낸 SVG 4개의 바이트를 빌드 후보와 대조했다. 공개 반영 완료는 commit/push 이후 GitHub Pages 배포 결과로 별도 확인한다.
