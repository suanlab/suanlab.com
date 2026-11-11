# 조화해석·신호 처리 원주제 대조

2026-11-11 기준. `harmonic-analysis`의 10개 원주제를 기존 한영 원고와 `learning-harmonic.ts`에 대조했다. 기존 입문·실습·연습을 보존하며 수렴·경계·함수 공간·역복원·단위의 조건을 증명과 독립 예제로 보강했다. 이 기록을 포함한 범위는 20영역·195개 원주제이며 전체 51영역·509개 일반 이론과 연구 심화의 완료를 뜻하지 않는다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{ko,en}.md`이다.

| 원주제 | ID | 정의·가정·보강 근거 |
|---|---|---|
| Fourier series·transform | fourier-series-transform | Fejér 양의 근사항등원·균등 수렴, Gibbs의 움직이는 최대와 점프 대비 비율 |
| Convolution 정리 | convolution-boundary-theorem | L¹ Fubini·L¹/L² Young 경계, crop Toeplitz와 circulant의 commutator |
| Laplace·Z | laplace-z-convergence | 두 극점의 세 ROC·양방향 안정 신호, 절대 합과 BIBO 필요충분 논증 |
| Sampling·aliasing | sampling-aliasing | L² 대역 제한의 연속 대표와 sinc 유도, 임계 표본률과 무한 에너지 tone 구별 |
| DFT·FFT | dft-fft-algorithms | 유한 기하합의 leakage, padding 격자와 실제 관측 폭, butterfly 점화식 |
| STFT·wavelet | stft-haar-localization | 분석/합성 창·dual·overlap 합, symmetric/periodic Hann과 Haar 완전성 |
| 필터·응답 | fir-frequency-response | FIR BIBO와 인과/비인과 안정 역, group delay·영점과 작은 실수 성분 |
| Parseval | parseval-normalization | Schwartz에서 L² 완비화로의 Plancherel, real spectrum 켤레 가중치·MSE 배율 |
| 시간·주파수 localization | gaussian-time-frequency-localization | 중심화·부분적분의 불확정성 증명, chirp·대칭화 공분산과 폭의 곱 |
| Inverse filtering | inverse-filtering-regularization | Tikhonov 최소해·gain 경계, 두 모드의 정확 MSE·잔차와 nullspace |

## 독립 예제와 그림

Python 10종을 한영에서 같은 코드로 실행한다. 코드·그림의 별도 확장 예제를 기존 실습의 새 제어라고 부르지 않는다.

Gibbs의 첫 최대를 t=π/(2M)에서 계산하면 M=1·4·16·64의 높이는 1.273240·1.184225·1.179305·1.179000이다. 극한 1.1789797445의 초과량은 점프 크기 2의 약 8.95%다. Fejér의 다른 양의 평균 절차와 원래 부분합을 구별한다. 한영 그림은 시간에서 줄어드는 폭과 y=2Mt에서 유지되는 초과량을 함께 보여준다.

Crop Toeplitz와 circulant는 정확분수 행렬로 구성해 순환 이동 commutator를 비교한다. 두 극점 .5와 2의 가운데 ROC를 택하면 DC 합은 1, 절댓값 합은 3으로 간다. “모든 극점이 단위원 안”이라는 안정 규칙의 인과성 전제를 드러낸다.

임계 표본률의 L² sinc는 n=−N,…,N의 실제 복원과 직사각형 spectrum의 독립 적분을 비교한다. 무한 지속 Nyquist sine의 영 표본은 이 L² 정리의 반례가 아니다. 같은 8개 복소 관측을 8·32·128점으로 padding해도 첫 영점은 바뀌지 않는다.

Periodic 16점의 WOLA 예제는 symmetric/periodic Hann·root Hann·겹침 없는 창의 Σw²를 계산한다. 양의 겹침 합으로 나눈 복원과 정보가 없는 D=0 좌표를 구분한다. Native 단일 frame은 0 연장이고 이 예제는 주기 경계다.

Chirp 예제는 시간·도함수 에너지를 독립 적분해 폭과 대칭화 공분산을 확인한다. 한영 그림의 두 패널은 시간과 각주파수라는 다른 좌표다. 양의 공동 시간·주파수 밀도를 가정하지 않는다. Inverse 예제는 80자리 Decimal로 8차원 시간 normal equation을 풀어 모드 MSE와 대조한다.

## 수치 수정과 해석 범위

기존 offset modulo는 f=10⁻²⁰,fs=8의 alias를 0으로 잃었다. 나머지를 먼저 구해 부호 대표를 정하고, quarter-turn 위상은 덧셈 대신 회전으로 적용한다. 같은 회전은 DFT 실습에도 적용한다. Nyquist의 실제 영 표본은 유지한다.

사각파의 0·±π에서 부분합을 정확히 0으로 유지한다. FIR은 π 근처에서 (1−2r)+2r cos²(ω/2)로 작은 실수부를 계산한다. ω=π−2⁻⁵¹에서 약 8.02×10⁻³²인 성분을 잃지 않고, 지정한 r=.5,ω=π의 정확한 영점은 유지한다.

복원·관측 잔차 MSE는 이 합성 실습의 해석적인 두 모드 진폭으로 계산한다. r=0,η=10⁻¹⁰⁰의 직접 MSE는 10⁻²⁰⁰이며 약 10⁻³²의 FFT 반올림 차와 다른 양이다. Native 안내와 본문에 수치의 의미를 명시했다. 곡선·표는 기존 FFT 기반 유한 정밀도 복원을 유지한다. 양의 모형 오차가 표현 범위 아래이면 —·안내로 표시하고, 정확한 무잡음 직접 역의 0·undefined 역·nullspace의 ∞를 구별한다.

## 출처와 후속 범위

- [Jerison, Fourier Series Part 3](https://ocw.mit.edu/courses/18-103-fourier-analysis-fall-2013/63577c6dfc0af665376b373b297a4185_MIT18_103F13_fseries3.pdf): convolution과 양의 평균 배경. Gibbs 재척도와 유한 commutator 예제는 본문에서 전개했다.
- [Stroock, Topics in Fourier Analysis](https://ocw.mit.edu/courses/res-18-015-topics-in-fourier-analysis-spring-2024/mitres_18_015_s24_full_lec.pdf): Fourier 급수·변환·L² 확장의 배경. 주기 1과 각주파수의 정규화를 혼용하지 않는다.
- [MIT 6.003 Z Transform](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/64490a008c1c5c25c86044351465abf7_MIT6_003F11_lec05.pdf), [Sampling](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/12e6e5d7567fca2e993ef8563fef5a60_MIT6_003F11_lec21.pdf): ROC·표본화 배경. 두 극점 안정 신호와 임계 표본률 함수 공간은 별도 명시한다.
- [Lerman, Shannon Sampling Theorem](https://www-users.cse.umn.edu/~lerman/math5467/shannon_aliasing.pdf), [Müller, FMP Sampling](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C2/C2S2_DigitalSignalSampling.html): spectrum의 급수 전개와 sinc의 근거. 본문은 L² 대역 제한의 연속 대표·유한 절단을 구분한다.
- [Smith, Weighted Overlap Add](https://www.dsprelated.com/freebooks/sasp/Weighted_Overlap_Add.html): 분석/합성 창의 곱과 hop 조건. 주기 16점 전수 복원 예제는 별도로 계산한다.
- [Hill, The Uncertainty Principle, §§4·5](https://math.uchicago.edu/~may/REU2013/REUPapers/Hill.pdf): Schwartz·부분적분의 배경. 본문은 각주파수로 상수를 맞추고 chirp의 공분산을 직접 계산한다.

일반 Lᵖ 승수·특이 적분, 비균일·잡음 표본화와 프레임, 일반 wavelet의 정칙성·admissibility, 연속 STFT 프레임의 완전한 이론, 일반 비주기·고차원 inverse 문제와 자료 기반 정규화 선택은 후속 심화다. 유한 모델 검증과 원주제 대조를 이 전체 연구 범위의 완료로 확대하지 않는다.

## 검증 기록

- Harmonic 계산·guide 검사 31개·한영 원고/reader 검사 20개·기존 원고 보존 검사 7개를 통과했다. 이 58개 검사를 전체 테스트 모음의 통과로 확대하지 않는다.
- Python 코드 20개를 최종 SHA·출력과 대조해 재실행했다. 10개 단원의 한영 절 수·표시 수식·실행 코드가 일치하며 영어 원고의 한글과 두 언어 원고의 일본어 가나가 없다. 320px에서 표 20개와 최종 SVG 4개의 배치를 확인했다.
- Alias 7개·FIR 21개·inverse 15개 입력을 정확분수 또는 750자리 기준 계산과 독립 대조했다. 작은 양수·subnormal 벌점·표현 범위 손실·구조적인 실제 0·undefined 직접 역을 구별했다.
- 실제 정적 빌드의 경계 선행 검사는 23개 상태·10개 동작 그룹이 통과했다. `npm run check:harmonic`의 한영·320/768/1440px·다크 모드 101개 접근성·배치 상태, 12개 동작 그룹, 안내 예제 30개·퀴즈 20개를 통과했다. 표·코드의 키보드 스크롤, 정수 draft·Escape 복구, 저장·언어 전환·초기화, 이전 계산 회귀도 확인했다.
- `npm run lint`, `npm run build`, `npm run typecheck`, `npm run validate:content`, `npm run check:export`가 통과했다. 빌드 1,893개 경로와 내보낸 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 확인했다.
- 소스 1,882개·한영 원고 1,300개·설정 8개·릴리스 파일 31개를 빌드 후보와 바이트 대조했고, 내보낸 SVG 4개도 원본과 같다. 공개 반영 완료는 commit/push 이후 GitHub Pages 배포 결과로 별도 확인한다.
