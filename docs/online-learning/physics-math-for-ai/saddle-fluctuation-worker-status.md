# 안장점·요동·비가우스 근사: 독립 검산과 실제 Worker

2026-11-10. `saddle-fluctuation-and-nongaussian-approximation`의 독립 전체 응답 검산기, 계산/검산 module Worker, 제어기와 React hook을 구현했다. 새 **154검사**가 Node.js 20/24에서 통과했다. 앞 수치·원고93검사와 합쳐 관련247검사이며, 서로 다른 종류의 유도식4728검사는 별도 증거다.

[20요건 중12개](saddle-fluctuation-acceptance-matrix.json)를 확인했다. 학습 화면·시각화·저장·실제 reader·생산 자산과 범위 감사는 남는다. 미등록 초안이므로 등록599단원·1850안내·1198원고, 입문315/362검증·47미완료는 유지한다. 전체51영역509주제와 연구 심화도 미완료다.

## 독립 검산의 근거

응답의 정상점 후보는 믿을 수 있는 입력이 아니라 **검증 대상인 증명 후보**다. 먼저 descriptor snapshot과 정규화 분수/길이/차원/크기/dyadic 한도를 확인한다. 별도 Rational 산술로 미분·잔차·강볼록성에 따른 위치/작용 오차·Hessian 오차 식을 확인한다. 인증 반경2^−672 이하가 증명되기 전에는 후보를 정련에 사용하지 않는다. 이차 모형은 정확한 영점 잔차가 필요하다.

검산기 자체의1280비트 방향성 산술·continuant Green 함수·Newton3회로 후보를 정련하고, 다시 정확 잔차로2^−1100 위치 오차를 인증한다. 여기서 전체 Hessian·역곡률·Gaussian 공분산·pivot·분리 determinant·앞 인자를 재계산한다. Gaussian 모멘트는 생성함수 계수를 열거한다. 생산기의 분수/구간 엔진·LDL 계산 경로를 가져오지 않음을 browser bundle의 실제 의존 관계로 검사했다.

원시 증명과 모형/해석 필드는 정확히 대조하고, 관찰 구간은 독립 인증 구간을 포함해야 한다. 정규화된512비트 dyadic 끝점과 절대/상대 폭2^−480의 검산 한도로 공허한 구간을 거부한다. 유효한 물리 인증 구간의 반올림 표현이 한 가지일 필요는 없다. 조건부 적분비는 한쪽 방향의 안전한 경계로 별도 확인한다. 상세 규칙은 [모형 계약§12](saddle-fluctuation-model-contract.md)에 있다.

## 수치·전송 검사

54개 생산 입력의 전체 응답을 독립 검산했다. 정상점·최소 작용·전체 행렬·보정·모멘트·조건부 경계와 하나/둘/퇴화 우물이 포함된다. 기존 Python350자리/정확분수54기준 대조는 [수치 단계](saddle-fluctuation-numerics-status.md)에서 별도로 통과했다.

새 검사는 검산70개·Worker 봉투60개·제어기24개다. 대표 이차/사차/영점 정상점과 세 우물의 각 원시 항목, 최대 행렬의 경계/내부/대칭 성분과 determinant에 대한 **513개 변조**를 거부했다. 잘못된 증명 후보·추가/누락 필드·원래 분수 표기/입력 귀속·getter/상속/symbol/순환/복사 한도도 거부한다. getter는 실행되지 않았다.

제어기에서는 이전 요청과 captured callback, 계산/검산 단계 교체·취소·기한·오류·재시도·두 복사 도중 취소를 확인했다. 주 스레드의 실제 hook bundle에는 수치 엔진이나 BigInt 계산이 없으며 SSR은 Worker 전역에 접근하지 않는다.

## 실제 브라우저 증거

로컬 Chromium에서 실제 module Worker 두 개를 실행했다. 54요청의 계산→검산과 수치 변조54건·요청 번호 귀속54건 거부를 확인했고 두 Worker를 종료했다. 응답 hash는 Node 생산기와 일치했다. 이는 브라우저 실행의 동일성 검사이며 Python 독립 기준과의 수치 대조를 대신하지 않는다.

React hook 기본 흐름은 Worker17생성/17종료·활성0, 계산과 검산 중 취소/숨김·새 입력 교체·잘못된 입력·실패 후 재시도·unmount를 확인했다. 의도적으로 postMessage를 보류하는 기본 흐름과 별도로, 최대32구간의 실제 계산/검산을 시작한 뒤 취소하는 흐름도 검증했다. 이 흐름은 Worker7생성/7종료·활성0, 두 단계에서 타이핑 가능·취소 후 늦은 ready 없음·재시도 결과 일치를 확인했다.

숨김 검사는 visibility 이벤트 모사다. 물리적으로 숨긴 탭의 자동화나 실제 Next 출력 자산·reader 검증으로 표시하지 않는다. 입력/전송 한도와 전체 행렬은 수치 단계의 한도를 유지한다.

## 재현과 다음 단계

```bash
npx tsx --test scripts/learning-saddle-fluctuation-{validation,workers,controller}.test.ts
npx -p node@20 node --import tsx --test scripts/learning-saddle-fluctuation-{validation,workers,controller}.test.ts
npx tsx scripts/check-saddle-fluctuation-workers-native.ts
npx tsx scripts/check-saddle-fluctuation-hook-native.ts
SADDLE_FLUCTUATION_NATIVE_FLOW=active-cancel npx tsx scripts/check-saddle-fluctuation-hook-native.ts
npm run typecheck
npm run lint
npm run validate:content
```

로컬 브라우저 스크립트는 기존4175 preview에 각 검사 전용 URL을 intercept한다. 기존 홈페이지/서버를 수정하지 않는다. 증거는 `.runtime/learning/saddle-fluctuation-validation/verification.json`과 Node20/24 로그, `.runtime/learning/saddle-fluctuation-worker/`의 Worker/hook/실제 취소 JSON 및 의존 관계다. 초기 잘못된 테스트 기대·타입 오류 로그도 보존했다.

다음은 기존 홈페이지 색상/서체를 따르는 한영 개념도·정상 경로/요동 폭·행렬 열지도/Three.js·보정과 조건부 경계·여러 우물의 시각화다. 예제·예측 실험·해설 문제·입력/진행 저장과 접근성을 연결한 뒤 Interactive Book의 실제 reader에 등록한다. 일반 비영 정상점의 고차 오차 보증, 실제 사차 공분산, 전체 well 적분이나 양자 이중 우물/터널링을 계산했다고 표시하지 않는다.
