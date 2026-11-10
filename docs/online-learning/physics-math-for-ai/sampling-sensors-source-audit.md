# 관측·샘플링·센서: 자료 대조와 집필 준비

2026-11-05. Fourier optics 다음 원래 주제의 준비다. 제품 모델·새 원고·홈페이지 등록·완료 주제 수에는 포함하지 않는다.

## 실제 확인한 1차 자료

- [MIT Signals and Systems, Lecture16: Sampling](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/8708ec068ebdea2c4ee2f38fad39fb83_MITRES_6_007S11_lec16.pdf)의16-1~16-2 설명을 확인했다. 대역 제한 가정, 최고 주파수의 두 배보다 높은 표본률, 서로 다른 연속 신호가 같은 표본을 만드는 aliasing을 다룬다. 유한한 표본 창의 sinc 합을 무한 표본의 정확한 복원과 같다고 설명하지 않는다. Nyquist 경계의 위상 의존성은 별도 예제로 준비한다.
- [EMVA 공식 다운로드](https://www.emva.org/standards-technology/emva-1288/emva-standard-1288-downloads-2/)에서 Linear/General 구분을 확인했다. [Release4.0 Linear](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf)의§2.1,2.4~2.7(pp.7~11)를 읽었다. 픽셀 면적·노출·광자 에너지, 유효 QE, DN/e− 이득, shot/read/양자화 잡음과 포화의 구분을 참고한다. 전체 표준의 계측 절차를 구현하거나 적합성을 검증한 것은 아니다. QE가 전체 픽셀 면적 기준이면 fill factor를 중복 적용하지 않는다.

## 두 단원의 설계

| 후보 ID | 핵심 질문과 그림 |
| --- | --- |
| `pixel-aperture-sampling-aliasing` | 연속 강도에서 픽셀의 면적 평균·적분, 점 표본과 유한 aperture, sinc 감쇠와 부호, 같은 표본을 만드는 서로 다른 주파수, Nyquist 경계와 유한 복원 창 |
| `photon-counting-exposure-sensor-noise` | 교정된 W/m²에서 노출 중 광자·전자·DN으로, Poisson 분포·평균/분산·SNR, 노출과 이득의 차이, 암전류·읽기 잡음·양자화·포화 |

앞 단원의 상대 강도·1차원 커널 적분을 광자 수로 바로 바꾸지 않는다. 물리 단위를 사용할 때에는 파장·면적·시간·복사조도의 별도 교정 입력을 둔다. 시간 변화는 노출 적분으로, 공간 변화는 픽셀 영역 적분으로 구분한다. 단색·독립 Poisson 도착·독립 Gaussian 읽기 잡음은 명시한 교육용 가정이다. 확률이 작은 한 번의 표본을 이론 곡선의 오류로 해석하지 않는다.

## 실제 독립 탐색

`.runtime/learning/prepare-sampling-sensors.py`는 제품 코드를 가져오지 않고 SciPy 적분·분포 합으로 비교한다. 픽셀48조건·432관측, 점 표본 alias6쌍, 포아송/정규 합성의 양자화 전 모멘트45조건을 계산했다. 값과 오차는 `.runtime/learning/sampling-sensors-exploration.json`, 실제 종료는 `sampling-sensors-preparation-exit.json`에 있다. 이는 미래 모델의 검증 결과가 아니다.

다음 구현에서는 닫힌식과 독립 적분을 대조하고, 영노출·영 QE·영잡음·위상 경계·고주파·잘못된 단위·입력 배열 누락을 검사한다. 양자화의1/12 DN² 근사는 충분한 변동이 있는 조건으로 한정한다. 정확한 반올림·ADC clipping·full-well은 각 변환을 명시하고, clipping 뒤 평균/분산을 양자화 전 식으로 보고하지 않는다. 일반 대학원생을 위한 단계별 계산 예제·직관·해설 연습과 한영 UI는 다음 집필 단계다.

## 후속 계산 모델·원고 작성

두 제품 계산 모델과 한영 상세 원고 네 편을 작성하고 독립 기준과 대조했다. 전용30개·전체3,793개 테스트와 TypeScript·lint·기존 콘텐츠 검사가 통과했다. 현재 단계는 [기반 집필](sampling-sensors-foundation.md)이며 홈페이지 등록·시각화·브라우저 검증은 다음이다.

## 후속 홈페이지 연결

위 기반 집필 이후 두 단원의 실제 홈페이지 연결·통합 검증을 마쳤다. 최신 상태는 [홈페이지 집필·검증 기록](sampling-sensors-authoring-status.md)을 따른다.460개 한영 단원이며 다음은 [파동 영역 종합 점검](waves-optics-domain-audit-plan.md)이다.
