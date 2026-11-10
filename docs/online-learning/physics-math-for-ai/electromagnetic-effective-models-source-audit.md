# 전자기 유효 모델의 자료 대조와 독립 준비

2026-11-05. 회로·센서 다음의 자료 준비다. 초기 자료 준비 단계의 기록이다. 이후 [계산 모델·상세 한영 원고 검증](electromagnetic-effective-models-foundation.md)을 마쳤다. 홈페이지 등록·연결은 다음 순서이며 준비 계산으로 입문 완료 수를 올리지 않는다.

이후 [홈페이지 통합 검증](electromagnetic-effective-models-authoring-status.md)을 마쳤다. 아래 미등록·다음 단계 설명과 수치는 해당 기록 당시 기준이다.

## 확인한 1차 자료와 범위

[Haus·Melcher §14.5](https://web.mit.edu/6.013_book/www/chapter14/14.5.html)의 정현파 전송선·종단 조건·저주파 집중 근사·사분파장 거동을 대조했다. [MIT OCW 3.024 Lecture 23](https://ocw.mit.edu/courses/3-024-electronic-optical-and-magnetic-properties-of-materials-spring-2013/resources/mit3_024s13_2012lec23/)의 전달행렬·경계 연결·Bloch 조건을 확인했다. 아래 식과 계산은 명시한 단순 가족을 직접 구성한 것이며 실제 복합 재료 전체의 유효 물성을 주장하지 않는다.

## 후보 단원

| 후보 ID | 중심 질문 |
| --- | --- |
| `transmission-line-lumped-limit-energy` | 한 주파수에서 맞춘 단자 응답이 저장 에너지까지 맞추는가? |
| `layered-medium-bloch-effective-limit` | 장파장 평균 물성으로 주기 구조의 모든 전파·금지대를 설명할 수 있는가? |

첫 후보는 양의 단위 길이 L,C와 길이 ℓ인 무손실 개방 전송선이다. exp(+iωt) 규약에서 c=1/√(LC), Z=√(L/C), θ=ωℓ/c이며 입력 어드미턴스는 i tanθ/Z다. 저주파 근사는 iωCℓ다. 입력 전압 진폭1에서 주기 평균 에너지는 Cℓ/(4cos²θ)다. 한 주파수의 응답을 맞춘 Cfit=tanθ/(ωZ)를 비분산 커패시터처럼 쓰면 에너지가 어긋난다. 반면 입력 susceptance의 주파수 미분/4는 이 가족의 에너지를 복원한다. 정확한 사분파장 극점에서는 유한 입력 구동과 개방 경계의 호환성을 별도 처리해야 한다. ω=0의 정적 에너지와 ω→0의 주기 평균 극한도 구분한다.

둘째 후보는 μ=1인 양의 유전율 두 층, 수직 입사·접선 E의 한정된 가족이다. exp(−iωt), 상태(Ey,Hz), M=M2M1 규약을 고정한다. 장파장 유효 ε는 두 층의 두께 가중 산술 평균이다. 이를 모든 방향의 등방 평균으로 확장하지 않는다. 정확한 반 trace D의 |D|>1인 Bloch 금지대는 이 상수 양의 평균 모델이 놓친다. 유한 무손실 적층의 반사·투과는 R+T=1이며 금지대를 흡수라고 부르지 않는다.

## 실행한 독립 탐색

제품 코드를 가져오지 않는 `.runtime/learning/prepare-electromagnetic-effective-models.py`에서 전송선72조건의 역방향 SciPy ODE·에너지 구적·입력 응답 미분을 대조했다. 극점 근처3조건은 실제 이진 입력을180자리 기준으로 확인했다. 두 층162조건은 행렬 지수·고유값·경계 선형계로 계산했고14조건에서 금지대를 확인했다. 각 조건의3주기 무손실 적층은 R+T를 검증했다.

실제 종료 코드0과 선택한 수치 결과는 `.runtime/learning/electromagnetic-effective-models-preparation-exit.json`, `electromagnetic-effective-models-preparation.json`에 있다. 이는 전역 오차 상한·완성된 원고·장치 인증이 아니다. 다음 등록 전 기준은478단원·42실험 그룹469항목의 `.runtime/learning/pre-electromagnetic-effective-models-registry-hashes.json`이다.

[집필 설계](electromagnetic-effective-models-authoring-plan.md) · [남은 순서](remaining-next-steps.md)

## 작은 차이와 위상 해석의 추가 준비

별도 `.runtime/learning/prepare-electromagnetic-effective-precision.py`는240자리 계산으로 전송선7조건·장파장 층상 매질6조건·균일 매질 위상4조건을 확인했다. tanθ/θ가 부동소수점에서1로 보이거나 반 trace가1로 반올림되어도 정확한 입력에서 작은 보정·전파 위상이 남는다. 실제 이진 입력이 π/2에 가까운 경우와 수학적으로 정확한 극점도 구분해야 한다. 균일 매질조차 arccos(cosθ)의 주값은 접힌 Bloch 위상이며 실제 누적 위상 전체를 복원하지 않는다. 실제 종료 코드0과 결과는 `electromagnetic-effective-models-precision-preparation-exit.json`, `electromagnetic-effective-models-precision-preparation.json`에 있다. 제품 구현·완료 단원으로 세지 않는다.

## 기초 집필로 확장한 독립 기준

재현 가능한 `scripts/learning/generate-electromagnetic-effective-models-references.py`로 전송선108조건·층상 매질162조건과 내부장972표본,240자리 전송선7조건·층상 매질9조건을 대조했다. [기초 집필 기록](electromagnetic-effective-models-foundation.md)은 자료 탐색과 새 제품·원고 검증을 구분한다.
