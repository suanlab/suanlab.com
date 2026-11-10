# Fourier optics: 다음 주제의 자료·독립 탐색

> 후속 갱신(2026-11-05): 아래는 준비 단계의 기록이다. 이후 [계산 모델과 한영 원고](fourier-optics-foundation.md)를 작성하고 독립 연속 적분 기준값·전용30개·전체3,749개 테스트·타입·lint·기존 콘텐츠 검증을 마쳤다. 홈페이지 연결은 다음 단계이며 등록·입문 완료 수는 유지한다.


2026-11-05. 기하·파동 광학 다음 원래 주제의 준비다. 아직 제품 모델·한영 원고·홈페이지 등록은 아니며 준비만으로 입문 완료 수를 늘리지 않는다.

## 실제 대조한 일차 자료

[TU Delft Scalar Diffraction Optics](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap6_Diffraction/DiffractiveOptics_2022_01Clean.html)의 Fourier Optics와 Spatial Light Modulators and Optical Fourier Filtering 부분을 읽었다. 렌즈로 만든 Fourier 평면의 마스크와 뒤 렌즈에 의한 재결상, 실제 좌표 반전을 구별한다. 전체 장이나 연결된 다른 교재를 읽었다고 주장하지 않는다.

[MIT2.71 강의22 슬라이드](https://ocw.mit.edu/courses/2-71-optics-spring-2009/bebe78c6a2ae2603a0d1a3ea85e0b4ca_MIT2_71S09_lec22.pdf)의4F 결맞은/비결맞은 결상, ATF·PSF·OTF 유도, MTF의 격자 대비 해석 부분을 읽었다. 실제 출력 좌표와 축척·반전을 흡수한 reduced coordinates가 다르다. 자료의 제3자 그림은 재사용하지 않는다. 결맞은 장에는 진폭 전달을, 비결맞은 강도에는 별도의 강도 전달을 적용한다.

## 새 단원의 초점

후보는 `coherent-fourier-pupil-spatial-filtering`, `incoherent-psf-otf-image-formation`이다. 앞서 작성한 유한 개구의 Fresnel/Fraunhofer 전파와 Gaussian 집속을 반복하는 대신 결상계·공간주파수 선택·관측의 결맞음 조건을 전개한다. 센서 픽셀·샘플링·잡음은 그다음 원래 주제이며 몰래 포함한 것으로 세지 않는다.

공간주파수는 rad/m가 아닌 cycles/m의 Fourier 관례를 명시한다. 물리 Fourier 평면 좌표는 λ f ν와 연결되며,4F 실제 출력에는 −f2/f1의 배율이 있다. 양의 축척만 사용하거나 좌표 반전을 생략하면 위상 마스크·이동·비대칭 물체에서 잘못된 결과가 될 수 있다.

결맞은 전파는 장을 선형으로 거른 뒤 제곱한다. 비결맞은 점원 집단은 강도를 더한다. 같은 장의 제곱을 강도 전달함수로 간주하지 않는다. 완전 차단 동공이나 영입력에는 원래 광량과 조건부 정규화를 구별하고 정의되지 않은 위상·정규화 PSF·OTF를 임의로 만들지 않는다.

## 일차원 직사각 동공의 직접 유도

평탄한 진폭 전달 Hc(ν)=1(|ν|<fc)에서 일차원 진폭 충격응답은 hc(x)=2fc sinc(2fc x)다. sinc는 sin(πt)/(πt)로 정의한다. 정규화한 비결맞음 PSF는 |hc|²/(2fc)이고 OTF는 max(0,1−|ν|/(2fc))다. 따라서 결맞은 장의 차단 주파수fc와 비결맞은 강도 전달의 지지2fc는 다르다. 이것만으로 어떤 물체·조명에서도 한쪽의 해상도가 무조건 우수하다고 결론 내리지 않는다.

이 삼각 OTF는 일차원 직사각 동공에 대한 결과다. 이차원 원형 동공의 Airy 함수나 원판 겹침 OTF라고 표시하지 않는다. 유한 창에서 sinc²를 잘라 적분한 값과 무한 공간의 정규화를 혼동하지 않는다. 모서리에서의 포함 관례는 연속 적분에는 측도0이지만 유한 DFT의 픽셀 수에는 영향을 준다.

## 실행한 독립 탐색과 남은 검증

`.runtime/learning/explore-fourier-optics.py`는 제품을 불러오지 않는다. 연속 직사각 동공의 복소장51개를 실제 적분과 sinc 식으로 비교하고, 이동한 동공의 겹침93개를 삼각 OTF와 대조했다. 별도의256점 유한 DFT에서는 복소 위상 동공32개를 만들고 직접 합96개·동공 자기상관·양의 점원6개로 만든 비결맞음 영상32개를 비교했다.

DFT 배열은 주기적이며 동공은 영으로 채운 여백을 갖는다. 이 주기 모델의 자기상관·영상 일치를 연속 무한 공간의 정확도나 실제 카메라 해상도 검증으로 부르지 않는다. 선형 위상·이차 위상·물체 위치에 따른 복소 OTF와 비음수 PSF를 보존한다. 현재 관측 오차와 실제 종료는 `.runtime/learning/fourier-optics-exploration.json`, `fourier-optics-preparation-exit.json`에 있다.

다음은 물리 좌표·입력 범위·광량과 정규화·차단점·결맞음 가정을 확정한 순수 모델, 이식 가능한 독립 기준값, 한영 상세 원고를 작성하는 단계다. 위 탐색은 그 산출물을 대신하지 않는다.

## 위상 동공에서 OTF의 부호와 위상

추가로 `.runtime/learning/explore-fourier-phase-otf.py`에서 일차원 동공 P(ν)=exp[i a(ν/fc)²+i bν/fc]의 이동 겹침 적분495개를 실행했다. t=ν/fc, L=max(0,2−|t|)이면 정규화 OTF는 (L/2)sinc(a t L/π)exp(i b t)다. 독립 수치 적분과의 최대 차이는약1.25×10⁻¹⁶이었다. 선형 위상이0인 경우에도 음의 실수 OTF 관측40개가 있었다.

수동 위상 동공의 비음수 PSF와 음·복소 OTF는 모순이 아니다. MTF의 크기만 보여주면 대비 반전·이동 위상이 사라진다. 새 단원은 원래 복소 전달과 크기를 구분하고, 선형 위상에 따른 좌표 이동의 부호를 Fourier 관례와 함께 검증해야 한다. 추가 탐색도 제품 모델·단원 검증으로 세지 않는다.

## 후속 홈페이지 연결

이 문서의 기반 집필 시점 이후 두 단원의 홈페이지 연결과 전체 검증을 마쳤다. 현재458개 한영 단원이며 [최신 집필·검증 기록](fourier-optics-authoring-status.md)을 기준으로 진행 상태를 읽는다. 다음은 [관측·샘플링·센서](sampling-sensors-source-audit.md)의 모델·원고 집필이다.
