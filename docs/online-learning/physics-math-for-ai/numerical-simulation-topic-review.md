# 과학계산·수치 시뮬레이션 원주제 대조

2026-11-12 기준. `numerical-simulation`의 11개 원주제를 한영 원고·실습·독립 예제에 대조한다. 입문 48개 및 후속 입문 362/362 완료와 전체 51영역·509개 원주제의 연구 심화 상태를 구분한다.

## 원주제와 보강 내용

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`, 계산은 `src/lib/learning-scientific.ts`다.

| 원주제 | 단원 ID | 보강한 논증·독립 실행 |
|---|---|---|
| 시간 적분 | explicit-implicit-runge-kutta | 부정확 implicit solve의 잔차 전파·tolerance와 차수·작은 해석적 변화 |
| 국소·전역 오차 | local-global-discretization-error | 이산 Gronwall·전파·step별 산술 예산과 균형 보폭 |
| 강성 | stiff-fast-slow-modes | 복소 A-stability 증명·L-stable 감쇠·trapezoidal ringing·명시 RK 한계 |
| Symplectic | symplectic-oscillator-integrators | 위상률 편향·장시간 오차·각각 안정해도 교대하면 불안정한 유리수 반례 |
| FD·FV·FEM | poisson-difference-volume-element | 정확 weak load의 절점 보간 증명·함수 L2/H1 오차·구적 가정 |
| Spectral | fourier-spectral-differentiation | Skew-adjoint·비선형 곱 규칙 실패·split 에너지 항등식 |
| Mesh·경계 | nonuniform-mesh-boundary-data | Neumann range·pinning의 숨은 잔차·물리 source와 load 사영의 차이 |
| CFL | cfl-heat-upwind-stability | Summation by parts·물리 공간 에너지 증명·정확 셀 순열 |
| Multigrid | two-grid-smoothing-correction | A-에너지 사영·피타고라스·고주파 minimax smoothing |
| Adjoint | discrete-adjoint-sensitivity | Implicit root와 고정 Newton 프로그램의 미분·잔차값과 잔차 도함수 |
| 검증 | manufactured-convergence-verification | M-matrix barrier·잔차/경계/이산화 예산·Jacobi의 정련별 작업량 |

언어별 비교표와 표준 Python 블록 11개, 한영 위상·smoothing SVG 네 개를 추가했다. 위상 그림은 분석한 주파수 편향이며 위치 오차나 실제 adaptive trajectory가 아니다. Smoothing 그림은 특정 uniform Poisson 대역의 이론 인자다. 기존 본문·연습·개념·안내·퀴즈·조절기 범위는 보존한다.

## 계산 보완

작은 감쇠에서는 저장 상태의 변화와 expm1의 해석적 변화를 비교해 같은 표시 숫자 뒤의 비영 오차를 유지한다. 큰 시간에서는 직접 exponential 기준을 유지해 작은 끝 꼬리를 지우지 않는다. 끝 진단과 시간별 표가 같은 비교를 쓴다. 이는 실제 실행 상태에 대한 오차이며 이상적인 truncation error로 대체한 값이 아니다.

Pure Neumann 존재는 알려진 source bias의 해석적 합 δ=0으로 판정한다. 작은 비영 bias를 tolerance로 허용하지 않으며, 조립된 부동소수점 합은 `assembledCompatibility`로 별도 보존한다. Upwind는 combination 형태로 계산해 CFL=1에서 작은 표본도 bit 단위의 정확한 순열을 유지한다.

연속 logistic objective gradient는 초기값과 목표 차이·expm1 변화를 중심화해 계산한다. 기존 이산 tangent·adjoint·유한차분을 바꾸지 않으며, 직접 뺀 연속 진단은 `expandedContinuousGradient`로 남긴다. 이 계산은 binary64 진단이며 임의 정밀도 또는 interval 인증이 아니다. 기존 stats 순서를 보존한다.

## 출처와 보존

배경과 해법 범위는 [MIT 수치 PDE 강의](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/), [SciPy IVP 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)와 대조했다. 새 반례·유리수·Decimal·에너지 예제와 그림은 독립 구성으로, 패키지의 adaptive solver나 원 논문 성능 재현이 아니다.

스물두 원고의 손계산 절 앞에 내용을 추가하고 기존 prefix·손계산 이후 suffix를 보존한다. Immutable fixture 네 개는 수정하지 않으며 이전 SHA·검토 SHA·이 기록을 reviewed ledger에 연결한다. 기존 한영 식의 표기 차이는 원문을 보존하고, 추가한 표시 수식·코드의 언어별 일치를 검증한다.

## 검증

관련 계산·원고 보존 검사 64개와 타입·lint·콘텐츠 검증이 통과했다. 한영 Python 22개 블록을 실행하고 추가 표시 수식·코드의 대응을 확인했다. 별도 650자리 Decimal 참조에서 감쇠 오차 1,296건의 최대 scaled error는 1.3×10⁻¹⁶ 미만, gradient 108건은 6.1×10⁻¹⁷ 미만이다. 극소 변화의 상대 오차는 각각 1.3×10⁻¹⁶·1.6×10⁻¹⁶ 미만이다. 경계 문제 405건의 호환성과 해를 대조해 최대 절점 오차 6.6×10⁻¹⁴ 미만을 확인했고, 셀 순열 18건은 bit 단위로 일치했다.

1,893개 경로 정적 빌드와 1,888개 HTML의 내부 링크·자산·fragment·sitemap·canonical 검사가 통과했다. 실제 export의 107개 상태는 한영 320/768/1440px 72개·영문 desktop dark 12개·경계/영역 연결 23개다. 접근성 위반·브라우저 오류가 0이며, 상호작용 13묶음·안내 33개·퀴즈 22개·이전 계산 43개·영역 연결 11개를 확인했다. 정수 초안의 blur 오류·모형 보존·Escape 복원, 표·코드 키보드 스크롤과 320px의 과학 그림 네 개도 확인했다.

원본 스물두 원고의 본문 prefix·손계산 suffix를 다시 비교했다. 소스 1,882개·학습 원고 1,300개·릴리스 파일 33개·설정 8개의 후보 byte 일치와 SVG 네 개의 export 일치를 확인했다. 전체 CI 및 공개 배포 완료는 이 로컬 검증과 구분한다.

## 후속 범위

일반 adaptive RK·강성 비선형 해법, 가변·비등방 계수의 FEM 및 multigrid, 비선형 보존형 spectral 해법의 수렴·entropy, adaptive mesh와 제약, adaptive/event/근사 implicit adjoint, 실제 하드웨어 오차 인증은 추가 심화 범위다. 원주제 대조가 해당 분야의 모든 연구 이론 완료를 뜻하지 않는다. 이번 검증으로 누적 30영역·295개, 후속 21영역·214개이며 다음은 알고리즘·계산복잡도·계산 구조 10개다.
