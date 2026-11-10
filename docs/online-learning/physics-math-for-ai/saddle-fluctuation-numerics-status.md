# 안장점·요동·비가우스 근사: 생산 수치와 독립 기준

2026-11-10. `saddle-fluctuation-and-nongaussian-approximation`의 소유 입력·비선형 정상점/잔차 인증·전체 Gaussian 행렬·첫 보정 구간·단일 좌표 우물의 세 경우를 구현했다. **새 수치 91개와 원고 2개, 관련 93검사**가 Node.js 20/24에서 통과했다. 독립 Python 기준은 54개다.

이 단계는 [한영 원고](saddle-fluctuation-manuscript-status.md) 다음의 수치 구현이다. 독립 응답 검산기·실제 Worker·학습 화면·reader는 남는다. [20요건 중 10개](saddle-fluctuation-acceptance-matrix.json)를 충족하며 단원·원주제·전체 목표는 미완료다.

## 입력과 계산

입력은 chain/well의 두 모형으로 나뉜다. 자체 descriptor 필드만 복사하고 getter·추가/누락·symbol·혼합 모형 필드를 거부한다. Chain은 N2…32와 probe0…N, 양의 τ/m/ℏ≤16, 0≤ω²/λ≤16, |a|/|b|≤16을 받는다. Well은 양의ℏ와−16…16의κ를 받는다. 분자/분모 각각12자리의 원래 표기를 동결해 보존한다. Reader용 입력 모듈의 bundle에는 수치 엔진이나 브라우저 전역 접근이 없다.

주 계산은 m/δ를 분리한 정규화 Hessian A를 사용한다. λ=0은 정확분수 선형 소거, λ>0은 768비트 dyadic Newton과 정확 잔차의 infinity-norm 후퇴 탐색이다. 후보를 전역 최소점이라고 가정하지 않고 강볼록 하계로 위치·작용을 인증한다. 128반복/64후퇴 예산을 넘으면 오류이며 입력 범위를 바꾸지 않는다.

진짜 최소점의 위치 구간을 Hessian에 전달하고 방향성 LDL·삼각 선형계로 전체 역곡률과 Gaussian 공분산을 감싼다. 물리 Hessian은 내부(N−1)×(N−1), 끝점0을 붙인 역곡률/공분산은(N+1)×(N+1)이다. 양의 역행렬과 자유 Green 행렬의 알려진 경계, 대칭 성분의 교집합을 사용한다. 전체 성분을 유지하며 임의의 norm 보정은 하지 않는다.

`detH=(m/δ)^(N−1)detA`를 분리하고, Gaussian 앞 인자의 제곱×π는 `(m/δ)/(2ℏdetA)`로 계산한다. 작은 determinant와 작은 자유 단계 앞 인자를 따로 반올림해0으로 만드는 일을 피한다. π·제곱근·작용 지수 자체의 소수 값을 구한 것은 아니다.

삼차 제곱·사차·전체c₁·ℏc₁·Gaussian 사차 모멘트 구간을 제공한다. 관찰 끝점은512비트 바깥쪽 반올림에1quantum을 더하고, 후보/잔차/반경/작용의 원시 증명 필드는 정확분수로 유지한다. 이것은 모든 값의 균일한 상대 정확도 보증이 아니다.

## 조건과 해석

- λ=0은 모든 끝점에서 정확한 이차 적분비1이다.
- 잔차가 정확히0인 영점 정상 경로에서는 비음수 사차 나머지의 실제 적분비 경계를 제공한다. 양끝0은 충분조건이며, N2의 반대 끝점도 포함한다. 이 조건을 원고에 명확히 했다.
- 일반 비영 정상 경로에는 같은 경계를 제공하지 않는다. c₁의 인증과 버린 고차 항의 오차 보증은 다르다.
- 넓은[0,1] 경계도 그대로 유지한다. Gaussian 공분산을 실제 사차 공분산으로 표시하지 않는다. 정상점의 평균 해석도 이차 법칙·영점 반전 대칭·평균을 계산하지 않은 경우로 구별한다.
- Well은 하나/둘/퇴화 최소점과 중앙 극대점, 곡률·Gaussian 분산·최소 작용·합친 앞 인자·첫 계수·ℏ/κ²·임계ℏ¹ᐟ⁴척도를 계산한다. 전체 well 적분이나 그 나머지 오차, 전체 이중 우물 양자 경로·터널링은 계산하지 않는다.

식·단위·원시 증명과 관찰의 차이는 [모형 계약](saddle-fluctuation-model-contract.md)에 정리했다.

## 독립 기준과 검증 범위

`scripts/learning/generate-saddle-fluctuation-references.py`는 입력JSON만 읽는다. TypeScript 계산기나 응답을 실행/읽지 않는다. 350자리 Decimal에서 영점 초기값·잔차 제곱합 탐색·밀집 소거로 후보를 구하고 Gauss–Jordan으로 역행렬 후보를 만든다. 이후 정확 Fraction으로 잔차·resolvent·역행렬 잔차·Loewner 경계를 검사한다. 소수 반복의 수렴 자체를 인증으로 삼지 않는다.

Gaussian 모멘트는 생성함수의 계수를 열거한다. 생산기의9/6/72/24다항식을 호출하지 않는다. 기준의 후보/공분산 오차를 c₁과 모멘트에 전파하며, 각 실제 물리값을 감싸는 더 좁은 기준 구간을 만든다.

54기준은32예제와22추가 입력이다. 같은 수학값의 다른 원래 표기도 포함하므로54개의 서로 다른 연속 모형이라고 세지 않는다. N2…32, 알려진 결합 정상점, 혼합 분모, 작은 질량의 경계층, 부호가 다른 양끝, 작은 모든 인자, 큰 정밀도, 약한 사차 항, 큰 사차 항의 영점 정상 경로, 퇴화/작은κ 우물을 포함한다.

검사는 다음 범위를 다룬다.

- 정상점과 최소 작용, 모든 Hessian/역곡률/Gaussian 공분산 성분, pivot·정규화 determinant·앞 인자, 모든 보정/모멘트 구간을 독립 기준과 대조한다.
- 별도 Rational 산술로 반환 후보의 실제 미분·잔차·반경·작용·Hessian/역행렬 오차 식을 재구성한다. 이는 테스트의 증명 대조이며, 아직 외부 응답을 받는 생산 검산기가 아니다.
- ℏ를 바꿔도 정상점/Hessian이 같음, 영점 정상 경로에서λ를 바꾸면 Gaussian은 같고 적분비 경계가 달라짐, 작은 질량의 determinant 분리, 자유 사슬의 첫 보정 닫힌식을 확인한다.
- 잘못된 유리수/정수·음의0·입력 소유권·getter 비실행·모형 혼합을 거부한다. 실제 bounded snapshot에54개 전체 응답을 통과시킨다.

최소 작용 오차는 잔차의 제곱에 비례한다. 처음 기준 파일에서 그 경계를1024비트 표시로 반올림하자 더 좁은 생산 구간을 충분히 대조할 수 없었다. 생산 결과를 느슨하게 바꾸지 않고 기준의 원시 Fraction 경계를 보존해 strict containment 검사를 유지했다. 첫 실패 로그도 증거에 남긴다.

## 비용과 전송

| 항목 | 관찰값 |
|---|---:|
| 전체 생산/기준 대조 입력 | 54 |
| 최대 전체 위치 행렬 | 33×33 |
| 최대 JSON 응답 | 1546741bytes |
| 최대 문자열 합 | 1468138자 |
| 가장 긴 로컬 계산/snapshot/직렬화 관찰 | Node24 약3.23초, Node20 약3.24초 |

이 시간에는 독립 기준 대조 전체가 포함되지 않으며 Worker 기한 보증도 아니다. 독립 대조를 포함한93검사는 Node24 약252초, Node20 약266초였다. 기존 wire의 단일 문자열65536자·전체 문자열2000000자·배열256·node100000·깊이32를 바꾸지 않았다. 원래 입력과 전체 행렬 크기도 줄이지 않았다.

## 재현과 증거

```bash
python3 scripts/learning/generate-saddle-fluctuation-references.py
npx tsx --test scripts/learning-saddle-fluctuation-{numerics,manuscripts}.test.ts
npx -p node@20 node --import tsx --test scripts/learning-saddle-fluctuation-{numerics,manuscripts}.test.ts
npm run typecheck
npm run lint
npm run validate:content
```

기준 입력/결과는 `scripts/fixtures/saddle-fluctuation-reference-inputs.json`과`scripts/fixtures/saddle-fluctuation-references.json`에 저장했다. 증거는 `.runtime/learning/saddle-fluctuation-numerics/verification.json`, Node20/24로그·버전별 프로필·Python진행/결과·공개배포확인 JSON이다. 앞4728유도식 검사는 별도 수학 기반 증거로 유지한다. 새93검사와 동일 종류의 검사 수로 합산하지 않는다.

## 공개 상태와 다음 순서

기존8524ae2의 GitHub배포42774970922는 성공했다. 공개 Interactive Book·교재 목록·Euclidean 커널 페이지HTTP200과599단원 링크를 확인했다. 새 안장점 단원은 아직 그 목록에 없다.

등록599단원·1850안내·등록한영1198원고, 미등록초안1개(2원고), 입문315/362검증·47미완료를 유지한다. 전체51영역509주제·영역 감사·연구 심화도 미완료다.

다음은 독립 전체 응답 검산과 변조/귀속 거부, 실제 계산/검산 Worker·취소/복구·종료다. 이후 기존 스타일의 한영 개념도·실험·저장·실제reader와 범위 감사를 진행한다. 현재 홈페이지 출력과 기존 등록 단원 소스는 보존했다.
