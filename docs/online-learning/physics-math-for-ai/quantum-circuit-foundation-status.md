# Quantum circuit·channel: bilingual drafts and circuit foundation

2026-11-09. 원래 미완료 주제 `quantum circuit·channel`을 네 단원으로 분리하고 한영 초안8개를 작성했다. **등록·홈페이지 연결·브라우저 검증은 아직 하지 않았다.** 공개 경로와 기존 교재를 유지한다.

## 작성한 내용

| 단원 | 본문의 핵심 예제 | 현재 준비 |
|---|---|---|
| `finite-qubit-circuit-order-interference` | HZH와 ZHH, HTH 간섭, Bell 준비, 제어 위상 | 한영7절, 정확 회로 kernel |
| `qubit-channel-kraus-choi-verification` | γ=16/25 감쇠, TP/CP/unital 분리, 전치의 Bell 반례 | 한영8절, 독립 손검산 |
| `system-environment-stinespring-reference` | 등거리/유니터리 완성, 복소 입력 부분추적, Kraus 기저 변경 | 한영7절, 독립 손검산 |
| `quantum-instruments-records-feedforward` | 확률0 갈래, 같은 효과/다른 교란, 고전 기록과 결맞는 보정 | 한영8절, 독립 손검산 |

모든 초안은 선수지식, 직관, 수식 전개, 구체적 예제와 해설 문제를 포함한다. 초안의 길이는 최종 분량 제한이 아니며, 시각화와 안내 실험을 만들면서 설명을 더 보완한다. 자세한 범위와 공개 조건은 [집필 계획](quantum-circuit-channel-authoring-plan.md)에 기록했다.

## 계산과 검증

`src/lib/learning-qubit-circuit-exact.ts`는 Q(√2,i)의 정확 진폭으로 I/X/Y/Z/H/S/S†/T/T†/CX/CZ/SWAP을 적용한다. 1–3큐비트·최대16게이트이며 site0은 최상위 비트다. 매 단계 진폭, 제곱 노름과 역회로를 제공한다. 정규화하지 않은 입력을 묵시적으로 변경하지 않는다. 이는 타입이 지정된 내부 kernel이며 JSON 파서·작업 취소·Worker 응답 검증을 대신하지 않는다.

독립 생성기는 SymPy 전체 행렬 경로를 사용한다. 복소 입력과 모든 계산 기저 입력, 두 제어 방향, 세 선 위치 및 합성 예제를 포함한 **557회로 사례**를 비교하고 역연산으로 복원했다. 감쇠 Choi 스펙트럼·부분추적·비unital성, 전치의 음의 방향, 유니터리 완성, Kraus 혼합, 참조계 주변 상태, 기록 블록과 최종 dephasing도 정확식으로 검사했다. 이 손검산을 일반적인 제품 채널 검증기라고 부르지는 않는다.

새 검사 **566개**, 기존 unitary 집필 회귀 **13개**, TypeScript와 ESLint를 통과했다. 네 단원의 양 언어를 기존 `getLearningContent`로 렌더링해 수식과 목차를 확인했고 영어 본문에 한글이 없음을 검사했다. 독립 fixture 재생성의 바이트 일치와 기존 소스/증거·정적 출력 보존은 `.runtime/learning/quantum-circuit-foundation/verification.json`에서 확인한다.

## 다음 순서

1. 채널 제품 계산기와 독립 검증기: Kraus·Choi·TP/CP·수반·합성, 계/환경/참조계, instrument와 확률0 조건부 상태.
2. 본문 예제별 관측값 검증, 상세 개념 그림·안내 실험·퀴즈, 접근 가능한 2D/Three.js 시각화.
3. 계산/검증 Worker·입력 검증·취소·재시도·저장 복구 및 실제 상호작용 검사.
4. 한영·명암·모바일/데스크톱 실제 reader와 production 자산을 검증하고 **도서 → Interactive Book**에 등록.
5. `quantum information`, 나머지 원래 주제, 전체 영역 감사 및 연구 깊이 보완.

현행571등록 단원·1713안내 실험, 원래 입문309/362검증·53미완료를 그대로 유지한다. 이 단계에서 전체 주제 완료를 주장하지 않는다. 전체51영역509주제, 추가 증명과 연구 연결도 계속 남아 있다. 이번 작업은 commit/push·공개 배포를 포함하지 않는다.
