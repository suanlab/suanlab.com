# 유한 Curie–Weiss 모형·한영 원고 준비 검증

> 현재 단계(2026-11-07): 세 단원의 실제 교재 등록과 검증을 완료했다. 아래는 해당 준비 단계의 기록이며, 최신 결과는 [실제 읽기 화면 연결 검증](phase-transitions-integration-verification.md)을 따른다.


2026-11-07. 상전이·질서변수 주제의 첫 가족인 `finite-curie-weiss-magnetization-symmetry`를 작성했다. **독립 계산 모델과 한영 원고의 준비 검증 기록**이다. 후속 worker·교수 화면 준비는 아래 보고서에 연결했고, 실제 교재 등록·읽기 검증은 남아 있다. 기존521단원·1563안내, 원래286/362입문 검증·76미완료 수를 유지한다. 전체 교육과정·연구 심화는 계속 진행 중이다.

## 구현 범위

`src/lib/learning-curie-weiss.ts`는 N=1..128, K=0..8, b=−4..4의 유한 고전 완전 연결 모형이다. 실제 binary64 무차원 입력, 정확한 BigInt 이항 다중도, 방향성 유리 구간을 사용한다. 평균·절댓값·이차 모멘트·양의 쌍 분산·장 응답, 거시/조건부/미시 엔트로피, 정규화·열적 단위의 자유에너지·에너지와 자기 항 관례를 분리한다.

최빈값은 반올림된 확률이 아닌 로그 가중치 구간과 공통 다중도 소거로 판정한다. 유한 영장 평균0과 두 봉우리는 양립한다. 최소 비영 장의 부호, 표시 범위 밖 비영, 미해결과 실제0을 구별한다. 유한 argmax의 변화를 열역학 상전이로 판정하지 않는다. 수치 근거는 [도출 노트](curie-weiss-derivation-notes.md)에 둔다.

## 독립 참조와 원고

- `scripts/learning/generate-curie-weiss-oracle.py`와 `scripts/fixtures/curie-weiss-oracle.json`: 실제 입력19사례를 Decimal400/900자리와 추가80자리로 확인했다. 방향성 구간·부호와1781관측값을 비교하고729개 다중도·확률 행 및 최빈값을 대조했다. N≤7의11사례는344개 이름 붙은 배치를 직접 열거한다. fixture 재생성도 바이트가 일치한다.
- `content/learning/physics-math-for-ai/finite-curie-weiss-magnetization-symmetry.{ko,en}.md`: 언어별14절·같은17수식·2표·해설6문제. 영어 약2600단어, 한국어 약5000한글자다. 계수·엔트로피·응답의 N 계수·자기 항·영장 예외·실제 log(2) 입력의 비동률·극한 순서를 설명한다.
- `scripts/learning-curie-weiss.test.ts`:31검사. `scripts/learning-curie-weiss-content.test.ts`:7검사. 실제 렌더러의 strict KaTeX·접근 가능한 수식/표, 한영 수식 대응과 풀이 수치를 확인한다. 기존 관련 모델을 포함한 회귀234검사·lint·타입 검사도 통과했다.

## 재현과 다음 순서

```bash
python3 scripts/learning/generate-curie-weiss-oracle.py
npx tsx --test scripts/learning-curie-weiss*.test.ts
npm run lint
npm run typecheck
```

증거와 최신 소스 해시는 `.runtime/learning/curie-weiss/verification.json`에 둔다. 기존4099소스·4264정적 출력은 그대로 보존한다. 사용자4175미리보기는 유지하고 이번 단계에 별도 브라우저나 서버를 만들지 않았다.

후속 [변분 Curie–Weiss](curie-weiss-variational-foundation.md)와 [육차 Landau](landau-sixth-order-foundation.md)의 독립 계산·한영 원고 검증도 마쳤고, [전용 worker 기반](phase-transitions-worker-foundation.md)을 추가했다. [고정 개념·안내·퀴즈·시각화의 독립 검증](phase-transitions-teaching-components.md)도 마쳤다. 다음은 세 가족의 실제 교재 등록과 읽기 검증이다. 큰 N의 계산을 학습 화면의 main thread에서 실행하지 않는다. 해당 통합 검증 전에는 상전이 주제나 새 단원을 완료 수에 넣지 않는다.
