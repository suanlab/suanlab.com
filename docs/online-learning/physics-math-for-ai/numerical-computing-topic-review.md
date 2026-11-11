# 수치계산과 자동 미분 원주제 대조

2026-11-12 기준. `numerical-computing`의 10개 원주제를 한영 원고·기존 실습·실행 예제에 대조했다. 누적 27영역·263/509개 원주제의 대조 기록을 연결한다. 입문 48개 및 후속 입문 362/362 완료와 전체 연구 심화는 구분한다.

## 원주제와 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`, 계산은 `src/lib/learning-numerical.ts`다.

| 원주제 | 단원 ID | 보강한 논증·실행 |
|---|---|---|
| 반올림 | binary-floating-rounding | γₙ 합 오차·합의 조건수·상쇄; 로그 없는 정확 유리수 반올림과 absorption |
| 안정 지수 | stable-exponentials-logsumexp | LSE Hessian의 분산·볼록성·이동과 섭동 상계; 작은 leading-class 손실 |
| 조건·backward error | conditioning-backward-error | 행렬·우변 섭동의 Neumann 분모; 조건에 따른 forward 상계와 중심화한 작은 데이터 변화 |
| 선형·비선형 해법 | linear-nonlinear-solvers | 양의 제곱근 Newton의 전역 수렴·이차 오차; 정확한 유리수 enclosure와 기계 산술의 구별 |
| Sparse·반복법 | sparse-iterative-residuals | Krylov 최적성·Chebyshev CG 상계; 실제 유리수 CG와 A-노름 검산 |
| 보간·구적 | interpolation-quadrature-errors | Richardson 선도 오차 제거·추정과 인증의 차이; x⁴의 정확한 격자별 오차 |
| 자동 미분 | autodiff-jvp-vjp | Dual 값·adjoint의 forward-over-reverse Hv; 공유 부모의 기여와 해석 Hessian 비교 |
| Gradient check | finite-difference-gradient-check | 함수 평가·절단 오차 예산과 보폭; 수직 gradient 오차의 방향별 거짓 통과 |
| 혼합 정밀도 | mixed-precision-refinement | 고정 인수의 수축·잡음 바닥과 가역인데 발산하는 반례; 실제 toy 반올림과 적용 범위 구분 |
| Checkpoint | computation-memory-checkpointing | 제곱근 저장 일정·정수 최적화·재계산 비용; mask 재생 실패와 같은 함수 보존 |

기존 본문·연습·실습의 tape·기저·조절기·안내·퀴즈를 보존했다. 한영 Python 10종은 표준 라이브러리만 사용한다. 차분 오차 예산과 깊이별 checkpoint 저장의 한영 SVG 네 개는 가정한 산술·일정의 별도 분석이다. 실제 binary64 오차 측정이나 GPU 속도 측정을 그렸다고 주장하지 않는다.

## 수치 계산 보완

반올림 helper는 log2가 바로 아래 binade의 값을 경계로 반올림했는지 비교해 지수를 보정한다. Helper의 53-bit 모드는 이미 표현된 binary64 값을 유지하며, UI의 3–24-bit 범위는 그대로다.

안정 확률은 요청한 상대 간격 (0,−g/T,−2g/T)에서 평가한다. Leading-class cross entropy는 log1p 꼬리로 직접 계산하여 offset=1000·g=50의 약 1.93×10⁻²²을 보존한다. 절대 LSE와 최대 logit은 같은 값으로 반올림될 수 있다. 요청 간격을 쓰는 것과 이미 반올림된 같은 logit에서 정보를 복구하는 일을 구별한다. 개별 exp가 유한해도 합이 overflow이면 naive 확률은 부재로 표시하며 0 벡터로 보이지 않는다.

대각 조건 실험은 η/Aᵢ의 데이터 변화와 산술 오차·잔차를 중심화한 좌표에서 평가한다. 목표 좌표는 별도로 반올림하며, 좌표에서 1을 빼는 방식의 상쇄를 민감도 정의로 삼지 않는다. 계산은 여전히 binary64이므로 잔차의 수학적 정리와 기계에서의 엄밀한 interval 인증은 구분한다.

## 출처·보존

부동소수점의 표현은 [Python 공식 설명](https://docs.python.org/3/tutorial/floatingpoint.html), 오차 정의는 [LAPACK](https://netlib.org/lapack/lug/node75.html), 안정 지수·CG·구적의 사용 조건은 SciPy 공식 문서와 대조했다. CG의 polynomial·Chebyshev 논증은 [Shewchuk 저자 강의](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf), 미분·재계산 배경은 [JAX autodiff cookbook](https://docs.jax.dev/en/latest/301/cookbook.html)과 [checkpoint 문서](https://docs.jax.dev/en/latest/gradient-checkpointing.html)를 확인했다. 특화된 유리수·dual tape·반례·그림은 독립 구성했으며 라이브러리 벤치마크나 원 논문 실험 재현이 아니다.

스무 원고의 기존 본문·연습을 그대로 보존하고 과거 fixture SHA·현재 SHA·이 기록을 reviewed ledger에 연결했다. Immutable fixture 네 개는 바꾸지 않는다.

## 검증

관련 계산·원고 보존 검사 65개와 타입·lint·콘텐츠 검증이 통과했다. 한영 Python 20개 블록의 코드 SHA·실행 출력·표시 수식이 일치한다. 정확한 유리수 기준의 이진 반올림 589건이 모두 일치했다. 별도 600자리 계산에서 안정 지수 48건·조건 실험 24건을 대조했으며, 최대 class-loss 상대 오차는 2.8×10⁻¹⁴, 확률 절대 오차 1.05×10⁻¹⁶, LSE scaled error 1.20×10⁻¹⁶, 데이터 민감도 상대 오차 1.32×10⁻¹⁶ 이하다.

1,893개 경로 정적 빌드와 1,888개 HTML의 내부 링크·자산·fragment·sitemap·canonical 검사가 통과했다. 실제 export의 한영·320/768/1440px·명암 101개 상태에서 접근성 위반·브라우저 오류가 0이고 상호작용 12묶음·안내 30개·퀴즈 20개·이전 계산 31개·원주제 연결 10개를 확인했다. 정수 초안의 blur 오류·모형 보존·Escape 복원, 표·코드의 키보드 스크롤과 최종 320px 과학 그림 네 개도 확인했다.

새 class 손실 항목은 기존 수치 목록 끝에 추가해 index로 연결된 네 개념 단계의 수치 순서를 보존했고 회귀 검사도 추가했다. 작업 소스 1,882개·학습 원고 1,300개·릴리스 파일 31개·설정 8개의 후보 byte 일치 및 SVG 네 개의 export 일치를 확인했다. 전체 CI 및 공개 배포 완료는 이 로컬 검증과 구별한다.

## 후속 범위

전체 IEEE 예외·초정밀 산술·interval 증명, 임의 희소 시스템의 전처리·재시작, 비매끄러운 AD의 미분 개념, 적응 구적의 엄밀한 정지 조건, 실제 하드웨어 혼합 정밀도 수렴·일반 분기 그래프의 최적 checkpoint 일정은 후속 심화 범위다. 이번 원주제 대조가 해당 분야의 모든 연구 이론 완료를 뜻하지 않는다. 다음은 상미분·편미분방정식의 10개 원주제다.
