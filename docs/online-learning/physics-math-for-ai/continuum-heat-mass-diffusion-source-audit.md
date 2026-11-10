# 열·물질 확산: 다음 자료 검토

이후 [계산 모델과 상세 한영 원고](continuum-diffusion-foundation.md)를 작성·검증했다. 홈페이지 통합은 미완료이며 아래 준비 단계 설명은 당시 기록이다.

2026-11-06. Euler·Navier–Stokes 화면 검증과 병행한 다음 주제의 준비다. 새 모델·원고·등록·주제 완료를 뜻하지 않는다.

## 확인한 1차 자료

- Gilbert Strang, MIT OCW, [5.4 The Heat Equation and Convection-Diffusion](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/5db29e69494eb09a26f7224d43adc6f6_am54.pdf), 1–4쪽. Fourier 모드의 감쇠, Gaussian 기본해와 폭 증가, 적분 보존과 제곱 노름 감소의 구별을 확인했다. 노트의 수학적 L² 에너지를 열역학적 내부 에너지J라고 바꾸어 표시하지 않는다. 이류 항의 부호 관례는 직접 확인해 우리 좌표계에 대입한다.
- Karl P. Burr, MIT Fluids Modules, [Illustration of the Diffusion Equation](https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec09/heat/vorticity.html). Gaussian 초기 온도와 확산계수의 역할, 작은 계수를 위한 로그 조절 설명을 확인했다. 페이지 초기식의 sigma는 지수에 곱해지는 역폭 매개변수다. 새 모델의 표준편차와 같은 기호로 무비판적으로 복사하지 않는다. 오래된 Java applet은 실행·검증한 것으로 주장하지 않는다.

## 집필 방향

### 화면 통합의 CDF 대조

2026-11-06에 [NIST DLMF7.9의 erfc 연분수](https://dlmf.nist.gov/7.9)를 확인했다. 새 실습의 반올림 CDF 대조는 표준정규 꼬리의 연분수와 원점 근처 급수를 사용하며 mpmath70자리 erfc 값13개로 대조한다. 이 반올림 뺄셈은 실제 질량 적분에 사용하지 않는다. 실제 유한 창 질량은 물리적 폭을 보존한 로그 공간 구적을 유지하며 CDF가 모두1이 되는 경우도 구별한다.

첫 단원은 양의 절대 온도 위에 놓인 주기 Fourier 모드로 Fourier 열유속, 유한 창의 열 수지와 온도 불균일성 감소를 구분한다. 둘째 단원은 일정 단면적의 무한 직선에서 유한 질량 Gaussian 농도의 Fick 확산을 다룬다. 평균 이동과 분산 증가, 총질량과 유한 창의 질량을 분리한다. 공통 모양의 PDE가 물리량·구성식·경계 조건까지 같다는 뜻은 아니다.

선택한 정확 해는 일반적인 변계수·벽·반응·다중물리 솔버의 인증이 아니다. 그 범위는 뒤의 별도 주제에서 확장한다. [집필 설계](continuum-heat-mass-diffusion-authoring-plan.md)를 따른다.
