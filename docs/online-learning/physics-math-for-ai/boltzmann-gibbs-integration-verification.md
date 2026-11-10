# Boltzmann·Gibbs 세 단원의 읽기 통합 검증

2026-11-07. 다음 세 단원을 로컬 Interactive Book에 등록하고 실제 Next.js 정적 읽기 페이지를 검증했다. 원래 주제의 **입문 범위** 검증이며, 전체 교육과정·주제의 모든 심화·연구 깊이 완료를 뜻하지 않는다.

| 단원 | 읽기 경로 |
| --- | --- |
| 기준 측도와 Boltzmann 가중치 | `/book/online/physics-math-for-ai/reference-measure-boltzmann-softmax-identifiability/` |
| 조건부 Gibbs 갱신과 연결성 | `/book/online/physics-math-for-ai/finite-gibbs-conditionals-scan-support/` |
| 연속 Gaussian과 정규화 가능성 | `/book/online/physics-math-for-ai/quadratic-gaussian-gibbs-normalizability/` |

## 등록과 교수 자료

기존518단원의 값·질문·선수 관계·순서와518개 안내 묶음의 전체 값·순서를 스냅샷과 대조했다. 다른 교육과정 정의를 보존하고 열역학 영역에만3단원을 추가했다. 로컬 등록은 **521단원·1563안내**다. 개념·실험·표지 dispatcher와 선수 링크가 전용 컴포넌트를 사용한다. 한영 원고6개, 고정 개념12단계·준비33개, 안내9개·예제27개·해설 퀴즈6개를 연결했다.

진입은 **도서 → Interactive Book**이다. Online Book 본문에는 새 교재 링크를 넣지 않았고 기존 읽기 URL은 유지했다. 실제 한영 메뉴8사례, breadcrumb6개, 검색6개, 도서관과 sitemap 연결을 확인했다.

## 실제 빌드와 읽기 검증

- 전체7635테스트, lint·타입·콘텐츠 검사와 정적 빌드 통과. 정적 출력1759HTML의 내부 링크·자산·fragment·canonical·sitemap 검사 통과.
- 세 단원×한영×명암×320/1440px의24조합에서 개념96단계·고정 준비264선택·안내72회·예제216회·퀴즈144선택·학습 표시와 저장 재로드24회를 확인했다.
- 페이지 전체 접근성336상태가 통과했다. 페이지 넘침·과학 표기의 줄 갈라짐·SVG의 비유한 좌표/잘린 이름·영어 본문의 한글 혼입·내부 자산 오류를 검사했다.
- 실제 scalar/list 입력, 잘못된 underflow·온도·가역 좌표 거부, 지원 변경 시 초기 상태 범위, 최대 seed/32비트/2048단계 경로, 전역 중지·감소 모션·화면 밖 중지와 수동 단계, 저장 거부 시 실험·퀴즈 동작을 확인했다.
- 실제 Next worker에서 큰 저장 설정의 계산과 언어·테마 응답, client navigation 정리, 자산 로드 실패·재시도, 개념 series 오류·복구·빠른 교체를 확인했다. 기록한 worker669시작·669종료·0잔존, stale 노출0, page error0이다.

실제 worker URL은 `/_next/static/chunks/boltzmann-gibbs-calculation.60af0f2c611d0704.js`다. 새 세 모델의 고유 계산 표식이 이 worker에만 있는지 정적 자산과 실행된 URL로 대조한다. 이전 esbuild 격리 검증과 실제 Next 자산 검증을 구분한다. 소유 브라우저는 모두 종료했고 기존4175미리보기는 유지했다.

## 재현과 증거

```bash
node --import tsx --test --test-concurrency=2 scripts/*.test.ts scripts/blog/*.test.ts
npm run lint
npm run validate:content
npm run build
npm run typecheck
npm run check:export
# 4175 정적 미리보기를 실행한 상태에서 순서대로 검사한다.
node scripts/check-boltzmann-gibbs-readers.mjs
node scripts/check-boltzmann-gibbs-reader-interactions.mjs
node scripts/check-boltzmann-gibbs-reader-lifecycle.mjs
node scripts/check-boltzmann-gibbs-navigation.mjs
```

최종 근거는 `.runtime/learning/boltzmann-gibbs-integration/verification.json`이다. 원래4082파일의 수치 모델·원고·worker 준비와 이전4093파일 기준을 대조하며, 등록·dispatcher·역사적 검사에서 의도한 변경 경로를 별도로 기록한다. 기존 주제 검사는 해당 시점의 등록 prefix를 검증하고 새 통합 검사는 전체521개 등록과 보존 조건을 검증한다.

원래362주제 중 **286입문 검증·76미완료**다. 열역학은 상전이·질서변수 → 상관·요동 → 일반 상세평형 → 유한계·열역학 극한 순서로 이어간다. [다음 집필 계약](phase-transitions-authoring-plan.md)의 유한 Curie–Weiss·변분 안정성·육차 Landau 가족은 설계와 이상적 손계산만 준비했으며 아직 모델·원고·독자용 화면으로 등록하지 않았다.
