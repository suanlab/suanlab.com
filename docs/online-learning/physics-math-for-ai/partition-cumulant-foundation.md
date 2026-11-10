# 유한 스펙트럼 분배함수·cumulant 모델과 원고 검증

현재 갱신: [실제 교재 입문 통합 검증](partition-function-authoring-status.md)을 마쳤다. 아래는 해당 준비·모델·분리 화면 단계의 계약과 당시 검증 기록이다.

추가 갱신: [실험·worker·고정 그림의 별도 브라우저 검증](partition-function-teaching-components.md)을 마쳤다. 아래의 화면 미작성 표현은 모델·원고 기초 검증 당시의 범위이며, 실제 교재 데이터 연결·등록·Next 읽기 화면 검증은 여전히 남아 있다.

2026-11-07. 분배함수의 첫 가족 `finite-spectrum-partition-cumulant-generators`에 수치 모델과 한영 원고를 작성했다. 세 가족의 모델·원고는 검증했으며, 실제 패널·고정 그림·안내·퀴즈·교재 등록은 아직 없다. 현재 홈페이지의515단원·1545안내, 원래284/362주제 입문 검증·78주제 미완료 수는 유지한다.

## 모델 계약

`src/lib/learning-partition-cumulants.ts`의 `partitionCumulantModel`은1..32개의 유한 상대 에너지, 유한 공통 기준과 온도를 받는다. 상태마다 다른 레이블을 유지하며, 퇴화를 에너지 표시값으로 재분류하지 않는다. 공통 기준과 상대 에너지를 정확한 이진 합으로 보존하여 입력에서 지정한 작은 간격을 잃지 않는다. 이미 반올림된 평평한 목록의 정보를 복구하지는 않는다.

양·음의 유한 비영 온도에서 U·에너지 분산·세 번째 cumulant·S·C·F·Z·ln Z와 첫 세 β 미분을 계산한다. T=0은 고정 Hamiltonian의 양의 쪽 극한이다. 바닥 퇴화도, 에너지 기준에 따른 Z의0/유한/무한 극한과 유한 F를 구별한다. 저장된 kB의 binary64 수치를 계산의 척도로 사용한다.

정확한 입력·유리 구간 연산과 기존 방향성 지수/로그 구간 커널을 결합했다. 분산은 양의 상태 쌍 합, 세 번째 cumulant는 구간 모멘트 다항식으로 계산한다. 정밀도128→8192비트를 적응적으로 늘리며, `interval-resolved`의 구간 상대 너비는2^-48 이하다. 이 목표는 최종 binary64 표시의 양자화 오차나 실제 장치의 측정 오차를 보증하지 않는다.

요구한 정확도를 얻지 못하면 값은 미해결로 남긴다. 구간이 이미 확인한 부호는 보존하며 작은 잔차를0으로 자르지 않는다. 닫힌 구간·구조적0·표시 범위 밖의 비영·미해결·수학적 무한대는 별도 상태다.

## 극저온 근사의 생략 한계

모든 들뜬 비용의 `u_i=gap_i/(kB|T|)`가8192보다 클 때만 별도 로그 꼬리 분기를 사용한다. 바닥 퇴화도를 g0, 들뜬 가중치 합을 R, 최대/최소 비영 간격의 비를 G라 쓰면, 입력 범위에서 `R<32·2^-8192`, `G<2^2100`이다. 실제 에너지의 공통 기준은 간격에서 소거된다.

평균 간격의 leading 식은 `Σgap_i exp(−u_i)/g0`다. 분산과 세 번째 cumulant도 간격 제곱·세제곱의 leading 합을 사용한다. 분산의 상대 생략은 `R(1+G²)`, 세 번째 cumulant는 `R+3RG+2R²G³`보다 작으며 모두2^-100보다 작다. 이 한계는 Python의 정확한 `Fraction`으로 독립 확인했다. `Math.log/exp` 자체의 오차가2^-100이라는 주장이 아니다. 로그 관측은 명시적으로 근사로 표시한다.

양의 절대 상한도 보수적으로 유지한다. 평균 간격은2^-6000 J, 분산은2^-5000 J², 세 번째 cumulant의 절댓값은2^-4000 J³보다 작다. `ln(g0+R)`는 `ln g0`와 `ln g0+2^-4096` 사이에 있다. 바닥이 하나인 경우에는 이 작은 양수를0이라고 선언하지 않고 로그로 보존한다. 거의 같은 거대한 로그의 차이도 먼저 반올림해 빼지 않는다.

## 독립 검산과 원고

- 실제 binary64 입력을 직접 해석한 Decimal 열거24사례를400..3200자리에서 계산하고80자리 추가 정밀도로 확인했다. 표시값 비교 외에 원래 참조값이 반환 구간을 별도 Python 경로로 확인했다. 최초405개였고, 공통 계산부 분리 후 확률 구간을 추가한 현재 결과는465개다.
- 극저온6사례는 독립적인 leading-log 계산을2400/2480자리에서 대조하고 보편적 생략 한계를 확인했다. 일반 참조와 극저온 참조는 재생성 시 바이트가 일치했다. 이전 이상적 유리 손계산은 실제 이진 입력용 oracle과 구분한다.
- 거의 균등한 표시 확률에서도 표현 가능한 세 번째 cumulant가 남는 경우,8192비트가 필요한 구조적 상쇄, 정밀도 상한의 미해결 값, 양의 온도에서도 음수가 될 수 있는 세 번째 cumulant, 큰 기준과 작은 간격, 음의 온도·퇴화·온도0·첫 로그 범위 초과를 확인했다.
- 한영 본문은 각각15절·16개 대응 수식·3표·해설6문제를 갖춘다. [한국어 원고](../../../content/learning/physics-math-for-ai/finite-spectrum-partition-cumulant-generators.ko.md)와 [영어 원고](../../../content/learning/physics-math-for-ai/finite-spectrum-partition-cumulant-generators.en.md)를 직접 읽을 수 있다. 영어는약2900단어, 한국어는5400자 이상의 한글 설명이다. source·Hessian·기준 이동·온도 의존 Hamiltonian·곱 상태공간·절단 오차와 AI log-sum-exp를 연결한다.
- strict KaTeX 렌더링·목차·표/수식의 키보드 접근과 본문의 손계산을 검증했다. 온도와 함께 준위를 움직이는 반례는 세 차분 간격에서 따로 확인했으며 차분 자체를 독립적인 참값 인증으로 사용하지 않는다.
- 모델43검사·본문6검사 총49개, 전역 타입 검사와 변경 TS의 lint가 통과했다. 기존 소스3988파일·정적 출력4250파일은 직전 통합 스냅샷과 일치한다. 홈페이지에 미등록 원고를 추가하지 않았고 기존4175 서버를 유지했다.

기계 판독 근거: `.runtime/learning/partition-cumulants-foundation-verification.json`. 전체 교육과정 완료나 분배함수 주제의 입문 완료를 뜻하지 않는다.

## 재현과 다음 순서

```bash
python3 scripts/learning/generate-partition-cumulant-oracle.py
python3 scripts/learning/generate-partition-cumulant-cold-reference.py
npx tsx --test scripts/learning-partition-cumulants.test.ts scripts/learning-partition-cumulants-content.test.ts
npm run typecheck
```

구간 검산은 현재 모델 결과 JSON을 만든 뒤 `generate-partition-cumulant-oracle.py --check-model <JSON>`으로 수행한다. 반환값의 부호·해결 상태·정확한 유리 구간을 보존해야 한다.

결합 이진 자리의 [모델·독립 기준·한영 원고](coupled-partition-foundation.md)도 검증했다. 이 과정에서 수치 계산을 `learning-partition-exact-core.ts`로 분리한 뒤 기존49검사와 독립465구간 검사를 다시 통과했다. 무한 에너지 사다리의 [정규화·절단 모델·독립 기준·한영 본문](energy-ladder-foundation.md)도 검증했다. 다음은 세 가족의 실제 worker·그림·안내·퀴즈·등록과 전체 정적 읽기 검증을 진행한다. 그 전까지 원래 분배함수 주제는 진행 중이다.

결합 자리의 구현에서는 J·h로 에너지를 만들며 부동소수점에서 먼저 `2h`나 합을 계산해 overflow/상쇄 정보를 잃지 않도록 한다. 공통 유한 상태 코어를 재사용한다면 정확한 이진 에너지 입력과 feature 공분산의 구간 계약을 보존하고, 이 가족의49검사·독립 참조를 다시 확인한다. 표시 확률만 곱해 상호작용 응답을 계산하거나 표시 에너지가 같은 상태들을 합치지 않는다.
