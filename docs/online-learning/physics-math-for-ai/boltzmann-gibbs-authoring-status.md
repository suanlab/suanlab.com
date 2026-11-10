# Boltzmann·Gibbs 분포 집필 상태

2026-11-07. **세 단원의 계산 모델·상세 한영 원고·개념·실험·퀴즈를 실제 Next 읽기 페이지에 통합하고 입문 검증을 마쳤다.** [최종 통합 기록](boltzmann-gibbs-integration-verification.md)과 `.runtime/learning/boltzmann-gibbs-integration/verification.json`을 따른다. 로컬521단원·1563안내, 원래286/362주제 입문 검증·76미완료다. 다음은 [상전이·질서변수](phase-transitions-authoring-plan.md)이며 전체 교육과정·연구 깊이는 진행 중이다. 아래 가족별 준비 기록은 해당 작성 시점의 스냅샷이다.

## 첫 가족: 기준 측도·Boltzmann 가중치·softmax 식별 가능성

ID: `reference-measure-boltzmann-softmax-identifiability`.

- 모델: `src/lib/learning-weighted-boltzmann.ts`.
- 원고: `content/learning/physics-math-for-ai/reference-measure-boltzmann-softmax-identifiability.{ko,en}.md`.
- 독립 참조: `scripts/learning/generate-weighted-boltzmann-oracle.py`, `scripts/fixtures/weighted-boltzmann-oracle.json`.
- 검증: `scripts/learning-weighted-boltzmann.test.ts`, `scripts/learning-weighted-boltzmann-content.test.ts`.

유한 준위1–32개와 기준 가중치를 명시한다. 실제 이진 입력의 상대 에너지·공통 이동·양의 실수 가중치를 보존하며, 금지 상태는 기준 에너지 검색 전부터 제외한다. 레이블 엔트로피 H, 평균 로그 기준 가중치, 기준에 대한 엔트로피 H+E[ln m]를 각각 제공한다. 임의 실수를 물리적 미시상태 다중도로 해석하지 않는다.

평균·쌍별 에너지 분산·열용량·자유에너지·log Z·확률·로그 확률을 방향성 유리 구간으로 계산한다. 128–8192비트 적응 정밀도에서 부호와2^-48이하 상대 폭을 확인한다. 표시 범위 밖 비영과 실제0, 정밀도 상한의 미해결, T→0⁺의 수학적 무한대를 구별한다. 확률이 표시상1이 되어도 음의 로그 확률을 유지한다. 정밀도가 부족하면 유한 허용 상태의 양의 확률·분산·엔트로피 등 증명 가능한 부호를 보존하면서 미해결로 보고한다.

원고는 언어별13개 절·11개 수식·2개 표·해설 연습6개다. 영어2518단어, 한국어5146한글자이며 세 층·네 방 직관과 직접 손계산을 포함한다. 정규화 사전의 음의 상대 엔트로피, 기준 가중치 재조정, 공통 에너지 영점과 척도의 모호성, 온도 의존 유효 에너지의 미분, softmax와kelvin의 별도 척도, 가중 바닥 퇴화와 유계 음의 온도를 설명한다. 이후 반응형 그림·실험을 구현했으며, 검증 범위와 다음 통합 단계는 [교수 화면 기록](boltzmann-gibbs-teaching-verification.md)을 따른다.

## 첫 가족의 준비 검증 기록

`.runtime/learning/weighted-boltzmann/verification.json`은 첫 가족 작성 시점의 스냅샷이다. worker 기반의 준비 시점은 `.runtime/learning/boltzmann-gibbs-worker/verification.json`이며, 최신 소스와 실제 읽기 근거는 `.runtime/learning/boltzmann-gibbs-integration/verification.json`에서 대조한다. 세 가족과 실험 구성의 준비 기록은 `.runtime/learning/quadratic-gibbs/verification.json`이다. 둘째 가족의 작성 시점 기록은 `.runtime/learning/finite-gibbs/verification.json`이다. 첫 가족의 모델·테스트·원고·독립 참조7파일은 당시 해시를 유지한다.

- 실제 입력 독립 Decimal16사례를400·800·1600자리에서 계산하고 추가80자리로 확인했다. 생성기를 다시 실행한 fixture도 바이트가 일치한다.
- 새 모델·원고34검증 통과: 관측값·부호·구간과 독립 참조 대조, 구간48비트 상대 폭, 물리적 다중도의 주변화, 기준 재조정, 영점·척도 불변성, 극소 확률과 큰 모멘트, 동률 바닥, 음의 온도, 입력 거부, 유한차분 응답 및 softmax 변환.
- 기존 분배함수3모델을 포함한 수치 회귀127검증 통과.
- 한영 수식 대조와 strict KaTeX, 키보드 접근 가능한 수식·표, 영어 한글 부재 검사 통과. ESLint와TypeScript 통과.
- 기존 등록 데이터·UI·스타일·정적 출력 보존을 해시로 대조한다. 이번 준비 파일은 아직 앱에서 import하지 않으며 새 빌드나 실제 브라우저 검증을 했다고 주장하지 않는다.

독립 참조의 유한 소수 정밀도는 구간 대조에서 명시한 충분히 작은 오차 예산으로 분리한다. 물리적 준비를 검증한 실험 데이터가 아니며 전체 통계물리 이론의 검증도 아니다. 큰 절대 정밀도를 요구해도 상한을 넘으면 미해결인 상태가 남을 수 있다.

## 둘째 가족: 조건부·갱신 순서·제약 지지집합

ID: `finite-gibbs-conditionals-scan-support`.

- 모델: `src/lib/learning-finite-gibbs.ts`.
- 원고: `content/learning/physics-math-for-ai/finite-gibbs-conditionals-scan-support.{ko,en}.md`.
- 독립 참조: `scripts/learning/generate-finite-gibbs-oracle.py`, `scripts/fixtures/finite-gibbs-oracle.json`.
- 검증: `scripts/learning-finite-gibbs.test.ts`, `scripts/learning-finite-gibbs-content.test.ts`.

정확한 이진 Hamiltonian을 조립하고 각 조건부를 국소 에너지 기준으로 정규화한다. 첫째·둘째 자리, random, systematic, joint block의5개 kernel과 평형 흐름·초기 점질량의 단계별 법칙·total variation 거리를 계산한다. 행벡터 관례의 sweep는 K1 K2이며 random 한 단계와 두 좌표 sweep의 갱신 수를 구별한다. 초기 상태와1–64범위의 증가하는 관측 단계1–8개를 지정할 수 있다.

정상성은 조건부 구성과 합성으로, 상세평형은 흐름 항등식으로, 연결성과 정상분포 유일성은 정확한 지지 그래프로 판정한다. 전체 곱에서 systematic의 비가역 흐름 차이는 βJ와 같은 부호이고, 합0 제약에서는 한 자리 kernel이 항등이다. π가 보존되고 상세평형을 만족해도 다른 허용 상태로 못 가는 경우를 별도로 표시한다. 수치 잔차 구간에0이 든다는 사실을 정상성 증명으로 쓰지 않는다. `fullyResolved`의 범위는 과학적 관측값이며, 별도의 수치 잔차 진단 구간은 포함하지 않는다고 명시한다.

둘째 좌표의 두 상태 chain에서 `λ=조건부 확률 차이의 곱=Corr(X,Y)²`와 `TV(q_t,π)=λ^(t−1) TV(q_1,π)`를 유도했다. 직접 뺄셈이 취소를 일으키는 경우에도 dyadic 구간을 투영 전에 곱하여 양의 작은 거리를 보존한다. 독립 참조는 다른 경로인 Pearson 공분산/분산에서 λ를 구한다. 극소 상호작용·최대 온도에서 선형 표시 범위와 고정점 양자보다 작은 넷째 sweep 거리도 부호·상대 구간을 확인했다. 경로 난수나 물리 시계를 생성한 것은 아니다.

원고는 언어별14개 절·15개 수식·표1개·해설 연습6개다. 영어2530단어·한국어4878한글자이며 네 방 직관, 실제 두 조건부 행렬, 순서 반전, 상세평형 반례, 제약 비혼합, 수축과 수렴,8비트 난수의1/256오차를 설명한다. T=0 조건부는 별도 극한 구성이 필요하여 명시적으로 거부한다. 유한 유계 음의 온도의 수학적 법칙은 허용한다.

## 둘째 가족 작성 시점의 검증

- 둘째 모델·원고34검증, 첫 가족과 기존 분배함수3모델까지 포함한 회귀161검증 통과.
- 실제 입력14사례의 목표 확률·로그·5개 kernel·평형 흐름·1/2/4단계 법칙·수축·거리를400·600·1600자리와 추가80자리 독립 참조에 대조했다. 생성 재현은 바이트가 일치한다.
- ideal-near 예제의 순차 평형 흐름8/125 대2/125와 random의 같은 흐름을 대조했다. 공통 에너지 이동·공통 에너지/온도 척도, 극소 전이의 연결성,64sweep 수축, 큰 Hamiltonian 합과 입력 거부도 검증했다.
- 한영 수식 대조·strict KaTeX·키보드 수식/표·영어 한글 부재, ESLint·TypeScript 통과.
- 기존 등록518단원·1554안내와 정적 출력4257파일을 유지한다. 실제 읽기 화면·worker·표본 경로·교재 등록은 이번 준비 검증에 포함하지 않는다.

## 셋째 가족: 연속 Gaussian Gibbs와 정규화 가능성

ID: `quadratic-gaussian-gibbs-normalizability`.

- 모델: `src/lib/learning-quadratic-gibbs.ts`, 방향성π: `src/lib/learning-directed-pi.ts`.
- 원고: `content/learning/physics-math-for-ai/quadratic-gaussian-gibbs-normalizability.{ko,en}.md`.
- 독립 참조: `scripts/learning/generate-quadratic-gibbs-oracle.py`, `scripts/fixtures/quadratic-gibbs-oracle.json`.
- 검증: `scripts/learning-quadratic-gibbs.test.ts`, `scripts/learning-quadratic-gibbs-content.test.ts`.

무차원 R²와 Lebesgue dx, 양의 온도를 명시한다. `A11>0,det A>0`을 정확한 이진 유리 연산으로 판정하고, 정규화 불가에서는 평균·공분산을 만들지 않는다. 정상화 가능한 경우에만 평균·공분산·precision·최저/평균 에너지·열용량·log Z·자유에너지·미분 엔트로피·최대 밀도와 상관, 대각 affine 좌표변환을 계산한다. source와 큰 공통 이동의 작은 차이도 최종 투영 전에 보존한다.

π는 Machin 교대급수의 항과 나머지 경계를 바깥쪽으로 반올림하여 감싼다. 제곱근은 상대 유효숫자 구간을 사용해 작은 양의 행렬식이 절대 양자 아래라는 이유로0이 되지 않게 한다. 행렬식 범위 초과·공분산 underflow·정확한 특이 행렬·미해결·수학적 발산을 구별한다. 수학적 π의 구간을 주장하면서 Math.PI로 대체하지 않는다.

원고는 언어별13개 절·13개 수식·표1개·해설 연습6개다. 영어2537단어·한국어4671한글자이며 그릇과 확률 타원, source 이동, 직접 역행렬, 밀도>1와 작은 셀 질량의 오차 경계, 기준 부피와 절댓값 Jacobian, 차원 있는 응답, 국소 Hessian의 전역 정규화 반례, 고전적 저온 경계를 설명한다. 고정 Hamiltonian의 양의 온도에서 C/kB=1이며, T→0⁺의 점질량은 원래 면적 밀도가 아니다.

## 세 가족·실험 구성 작성 시점의 검증

`src/lib/learning-boltzmann-gibbs.ts`에 세 단원의 안내9개와 예제27개를 구성했다. 수치 모델은 타입으로만 import한다. 예제 getter는 소유한 스냅샷을 반환한다. `scripts/learning-boltzmann-gibbs-config.test.ts`에서27계산·구성의 한영/범위·스냅샷 격리·경량 import를 확인한다. 이것은 실제 화면이나 등록된 교재 안내가 아니다.

- Gaussian 모델/원고36검증과 실험 구성30검증, 합66검증 통과. 기존 분배함수3모델과 앞 두 가족을 포함한 회귀227검증 통과.
- 실제 입력17사례를 Fraction과 Decimal400·1200자리, 추가80자리로 대조했다.14개 정상화 가능 사례와3개 비정부호 사례이며 생성 재현도 바이트가 일치한다.
- 역행렬·source·고정 Hamiltonian 열용량, 영점/공통 척도, 양·0·음 경계, 범위 밖 행렬식/평균/공분산, 양의 최소 온도, 밀도>1·음의 미분 엔트로피, 방향 반전/큰 Jacobian을 확인했다.
- π의 독립400자리 참조·정밀도 증가 포함관계·캐시 격리, 모델8192비트 상한 계산, 한영 수식·strict KaTeX·접근 가능한 수식/표·영어 한글 부재를 확인했다. lint·타입 검사 통과.
- 앞 두 가족의 모델·테스트·원고·참조14파일과 기존 등록 데이터·UI·정적 출력4257파일을 유지한다. 현재 등록은518단원·1554안내다.

검증 근거와 소스 해시는 `.runtime/learning/quadratic-gibbs/verification.json`에 둔다. 예전 proof는 해당 준비 시점의 스냅샷이며 최신 원고/진행 문서의 해시를 소급 변경하지 않는다.

## Worker·유한 비트 경로의 최신 검증

- `learning-boltzmann-gibbs-{protocol,controller,evaluator,worker-factory}.ts`, `learning-boltzmann-gibbs.worker.ts`와 두 calculation/series hook을 추가했다. 세 수치 모델과 경로 계산은 evaluator에서만 import한다.
- 입력은 지원·온도·밀집 배열·관측 범위와 선택적 경로 설정을 검증하고 기본값을 정규화한 불변 스냅샷을 사용한다. `1e-400` 같은 비영 표기가 binary64의0으로 사라지면 입력 오류로 처리한다.
- 결과는 요청ID·종류·입력 key, 관측값 상태·부호·양의 유리 분모·길이 제한, 지원 그래프·배열·정규화 불가 분기와 `fullyResolved` 범위를 검증한다.64sweep 수축의147,672자리 계수도 제한 안에서 수용한다. 이는 전송 검증이며 임의 송신자의 수학적 정답을 증명하는 검증이 아니다.
- 교체·취소·오류·해제마다 이전 worker를 종료하고 늦은 callback을 차단한다. hook은 새 입력 렌더와 effect 사이에도 이전 결과를 숨긴다. 유효하지 않은 series도 worker 생성 전에 실패한다. SSR에서 Worker를 읽거나 동기 수치 대체 계산을 하지 않는다.

`src/lib/learning-finite-gibbs-path.ts`는 이론적 법칙과 분리한 경로 계산이다.1–32비트 왼쪽 격자의 CDF ceiling을 유리 구간에서 인증하고, 정확한 에너지 대칭으로 동률 경계를 판정한다. 그리드 경계를 확정하지 못하면 경로를 보류한다. K1/K2를 각각 양자화한 뒤 random 혼합·systematic 곱·joint block을 구성하며 한 단계 TV 오차 상한을 기록한다. LCG32-high-bits seed 경로는 결정적 예시이며 독립 균등 난수나 정확한 실수 Gibbs 표본으로 부르지 않는다. 양자화가 양의 전이를 삭제할 수 있어 Q의 연결성도 원래 kernel과 분리한다.

독립 Python Decimal-CDF/정수 참조12사례가 실제 입력의 양자화 kernel·경로·빈도·최종 seed와 일치하며 재생성도 바이트가 일치한다. seed0·32비트·2048단계·합0 비혼합·블록·미약/희귀 전이를 포함한다. 새 worker/경로62검증과 기존 모델/원고/구성까지 회귀289검증, lint·타입 검사를 통과했다.

격리 esbuild harness에서 실제 worker와 React hook을 검증했다.27예제, 최대 경로, 무거운 계산 중 이벤트루프 응답, 교체·취소·재시도·언어/테마 렌더, 실제 worker 로드 실패, 불량 입력·series 교체·해제/재마운트를 확인했다. 최종40시작·40종료·0잔존, 잘못 노출된 계산/series 결과0, page error0이다. 과학 계산 표식은 격리 worker에만 존재했다.

이 harness는 Next의 worker URL을 별도 esbuild 출력으로 연결한 **격리 검증**이다. 실제 Next worker asset·교재 읽기 화면24조합을 검증한 것이 아니다. 소스·재현 명령은 [worker 준비 기록](boltzmann-gibbs-worker-preparation.md)에 둔다. 임시4186서버와 소유 브라우저를 정리했고 사용자4175미리보기는 유지했다. 기존4067소스·4257정적 출력과518단원·1554안내를 보존한다.

## 실제 읽기 통합과 다음 순서

전체7635테스트·lint·타입·콘텐츠·정적 빌드/출력 검사를 통과했다. 한영·명암·320/1440px의 실제24읽기 조합에서 개념96단계·고정 준비264선택·안내72회·예제216회·퀴즈144선택·저장 재로드24회를 확인했고, 페이지 전체 접근성336상태가 통과했다. 실제 Next worker의 큰 설정·오류·복구·client navigation과 모션·입력·저장 거부도 검증했다. 자세한 범위와 보존 근거는 [통합 기록](boltzmann-gibbs-integration-verification.md)에 둔다.

다음은 [상전이·질서변수의 집필 계약](phase-transitions-authoring-plan.md)이다. 유한 Curie–Weiss의 실제 입력 독립 참조와 모델부터 만들고 한영 원고·교수 자료로 이어간다. 변분 안정성과 극한 순서, 육차 Landau 가족도 순서대로 작성한다. 현재는 설계와 이상적 손계산 준비이며 새 단원으로 등록하지 않았다.

상관·요동, 일반 상세평형, 유한계·열역학 극한과 이후7영역도 남아 있다. 주차·분량 제한 없이 전체 교육과정과 연구 심화를 이어간다.
