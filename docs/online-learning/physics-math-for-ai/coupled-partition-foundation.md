# 결합 이진계의 분배함수·독립성 모델과 한영 원고

현재 갱신: [실제 교재 입문 통합 검증](partition-function-authoring-status.md)을 마쳤다. 아래는 해당 준비·모델·분리 화면 단계의 계약과 당시 검증 기록이다.

추가 갱신: [실험·worker·고정 그림의 별도 브라우저 검증](partition-function-teaching-components.md)을 마쳤다. 아래의 화면 미작성 표현은 모델·원고 기초 검증 당시의 범위이며, 실제 교재 데이터 연결·등록·Next 읽기 화면 검증은 여전히 남아 있다.

2026-11-07. 분배함수 주제의 두 번째 가족 `coupled-binary-sites-partition-factorization`의 수치 모델과 상세 한영 원고를 작성하고 검증했다. 무한 사다리의 [모델·원고 검증](energy-ladder-foundation.md)도 마쳤으며, 세 가족의 실제 패널·고정 개념도·안내·퀴즈·worker·등록은 남아 있다. 현재 공개 교재의515단원·1545안내, 원래 주제의284개 입문 검증·78개 미완료 수는 유지한다.

## 준비와 계산 계약

`src/lib/learning-coupled-partition.ts`는 두 이름 붙은 스핀과 `E=−Jσ1σ2−h(σ1+σ2)`를 사용한다. 유한 J,h,T 및 전체 곱 공간 또는 합0 제약을 받는다. J,h의 정확한 이진 조합으로 에너지를 구성하므로 2h나 개별 에너지가 binary64 범위를 벗어나도 입력을 먼저 반올림하지 않는다. 금지 배치는 정규화에서 제외하고 확률0을 명시한다.

기존 유한 스펙트럼 계산을 `learning-partition-exact-core.ts`로 분리하고 상대 준위 API를 얇은 wrapper로 유지했다. 계산부는 최대4개 정수 feature의 평균·공분산을 추가한다. 같은 정확한 에너지 비용의 계수를 먼저 묶어 대칭적인0을 증명하고, 공분산을 상태 쌍 합으로 계산한다. 유리 구간에서 크기와 부호를 판단하며 표시값을 빼서 공분산을 만들지 않는다.

모델은 A=σ1σ2, M=σ1+σ2의 평균과 source Hessian, 개별 스핀 평균과 공분산을 반환한다. 차원 없는 source는 `(βJ,βh)`다. 고정 온도에서 물리적 응답은 공분산에 부호 있는 β를 곱한다. 음의 온도에서 source Hessian의 양의 준정부호성과 물리적 대각 응답의 부호를 구별한다.

상호정보량은 실제 두 주변 엔트로피에서 결합 엔트로피를 뺀 구간으로 계산한다. 작은 확률에 ln0을 적용하지 않으며 KL 비음수성은 구간에만 사용한다. 구조적 독립과 제약의 ln2 정보량은 별도로 증명한다. 미세한 양수를 표시0으로 바꾸거나 음의 잔차를0으로 자르지 않는다. 정밀도128→8192비트에서도 상대 정확도를 확보하지 못한 결과는 미해결로 남긴다.

비교량 `Δ=lnZ−2lnZ1`와 자유에너지 차이도 별도로 계산한다. Δ는 부호가 있고 상호정보량과 다르다. J=0의 전체 공간에서는 항등식으로 정확한0을 반환한다. T=0에서는 바닥 에너지 차이와 퇴화도로 Δ의 발산/유한 극한을 구하며, 유한 F 차이를 무한대끼리 빼서 계산하지 않는다.

## 온도0과 근사 범위

T=0은 고정 Hamiltonian의 양의 온도 쪽 극한으로, 바닥 배치에 균등 확률을 준다. source 공분산의 극한과 물리적 매개변수 미분을 같은 값으로 해석하지 않는다. 작은 양·음 매개변수 변화가 고르는 바닥집합에서 관측량 평균을 각각 비교해 국소적인0 또는 미분 부재를 구분한다. 특히 J=h=0에서 교차 공분산이0이어도 h 변화에 대한 정렬 응답은 미분 불가능하다.

공통 계산부의 에너지 범위는 `|E|<2^1027 J`, 정확한 격자는2^-1074 J 이상이다. 따라서 최대/최소 비영 간격 비를 보수적으로2^2110 이하로 제한할 수 있다. 기존 극저온 분기의 `u>8192` 조건에서 분산·세 번째 cumulant 생략 한계2^-100은 이 범위에서도 유지된다. feature 대각 분산의 양의 leading 합도 같은 명시적 근사 표기를 쓴다. 이는 부동소수점 로그·지수의 반올림 정확도를2^-100으로 인증한다는 뜻이 아니다. 모든 입력의 모든 관측이 정밀도 상한에서 해결된다고 주장하지 않는다.

## 독립 검산과 본문

- 독립 Python Decimal 열거16사례를400/600자리와 추가80자리로 확인했다. 실제 binary64 J,h,T,kB를 해석하고 직접 중심화한 feature 공분산, KL 형태의 정보량과 정규화 비교를 계산한다. TypeScript의 상태 쌍 공분산·엔트로피 차이와 다른 경로다.
- 각 참조값과 표시값뿐 아니라 정확한 유리 반환 구간을 비교했다. Decimal 참조의 수렴 확인과 보수적인 반올림 여유를 함께 사용하며, 이를 임의 정밀도의 상징적 참값 인증으로 표현하지 않는다. 참조 파일은 재생성 시 바이트가 일치한다.
- 양·음 결합, 편향된 주변분포, 음의 온도, 제약, 약한 결합, 표시 범위보다 작은 양의 정보량, 바닥 퇴화·3상태 교차, 표시 범위 밖의 실제 에너지와 잘못된 입력을 검사했다.
- 기존 첫 가족의49검사와 신규29검사, 총78개가 통과했다. 공통 계산 분리 후 기존24사례의 독립 Python 구간 검사를 다시 실행하여465개 구간을 확인했다. 확률 구간이 추가되어 이전405개보다 늘었다. 전역 타입 검사와 변경 TypeScript lint도 통과했다.
- [한국어 원고](../../../content/learning/physics-math-for-ai/coupled-binary-sites-partition-factorization.ko.md)는 한글5459자, [영어 원고](../../../content/learning/physics-math-for-ai/coupled-binary-sites-partition-factorization.en.md)는2492단어다. 각각14절·서로 대응하는 수식14개·표3개·해설6문제를 갖춘다. strict KaTeX와 목차, 수식·표 키보드 접근, 영어의 한글 미노출을 확인했다.
- 직전 통합 소스3988개와 출력4250개는 모두 해시가 일치한다. 새 페이지 빌드나 배포는 없고 기존 미리보기4175 서버는 유지했다.

기계 판독 근거는 `.runtime/learning/coupled-partition-foundation-verification.json`이다. 본문 렌더 검사는 실제 시각화 패널·worker·브라우저 읽기 화면 검증을 대체하지 않는다.

## 재현과 다음 순서

```bash
python3 scripts/learning/generate-coupled-partition-oracle.py
npx tsx --test scripts/learning-coupled-partition.test.ts scripts/learning-coupled-partition-content.test.ts scripts/learning-partition-cumulants.test.ts scripts/learning-partition-cumulants-content.test.ts
npm run typecheck
```

`infinite-energy-ladder-normalization-cutoff`의 모델·원고·독립 검산도 마쳤다. 다음은 [교육 화면 설계](partition-function-interaction-plan.md)에 따라 세 가족의 교육 화면과 등록·정적 export·실제 읽기 검증을 진행한다. 탐색 경로는 도서 → Interactive Book을 유지하며 Online Book 목록에 추가하지 않는다. 전체 교육과정·영역 종합 점검·연구 심화는 계속 진행 중이다.
