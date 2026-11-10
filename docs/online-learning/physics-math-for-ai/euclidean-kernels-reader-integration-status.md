# Euclidean 커널: 실제 교재 등록과 생산 자산 검증

2026-11-10. `euclidean-quantum-kernels-and-time-slicing`을 Interactive Book의 599번째 단원으로 연결했다. **599단원·1850안내·한영1198원고**이며 미등록 초안은 없다. 경로적분 원주제의 네 처리 중 두 번째 단원이다. 입문 원주제315/362검증·47미완료, 전체51영역509주제의 감사와 연구 심화는 별도 미완료다.

## 실제 교재 연결

한영 제목·선수 단원·8개념·6예측 안내·8해설 문제를 등록했다. Brownian bridge → Euclidean 커널의 이전/다음 연결, 장이론 영역의 검색 결과와599개 표지 목록에서 접근한다. 실험은 지연 로딩하고 교재의 공통 개념/퀴즈와 중복하지 않는다. 입력/표시·실험 안내의 저장과 공통 개념/퀴즈/읽기 완료 저장을 각각 복원한다.

도서 메뉴의 **Interactive Book** 진입 구조를 유지한다. 실제 단원 URL은 `/book/online/physics-math-for-ai/euclidean-quantum-kernels-and-time-slicing/`이며 Online Book 목록에는 새 콘텐츠 진입 링크를 추가하지 않았다. 기존 홈페이지의 CSS와 색 변수, SVG 기준 그림·행렬 열지도·Three.js 막대를 사용한다.

## 검증 범위

- 기존211수치/검산/전송/수명/원고/학습 검사와 신규6교재 통합 검사, 합계217개를 Node.js20/24에서 확인했다. 기존211검사의 통과 로그는 이전 단계에서 보존했고 이번 단계에서6개를 추가 실행했다. TypeScript·ESLint·1834URL콘텐츠 검증과 분리된 환경의 생산 빌드가 통과했다.
- 1198등록 원고의 엄격한 수식 렌더링·영문 한글 혼입·제목 구조를 확인했다. 이전598단원의 내용/순서·안내와51영역의 기존 범위는 SHA-256 기준으로 보존한다. 전체 reader의 의존성 그래프에는 정확 수치 엔진과 독립 검산기를 포함하지 않는다.
- 한영×명암×320/1440px8조합에서64개념·64정답/오답 해설·읽기 완료/입력/퀴즈 복원, 3D 키보드와 context loss/retry,선수/이전 링크·검색 결과를 확인했다. 문서 가로 넘침과 영문 한글 혼입이 없었고8접근성 검사가 통과했다.
- 실제320px reader에서32구간·1089공분산 셀·전체 길이의 부호 있는 분수·N2정밀도 한 셀의3D·터치 회전·표시만의 시간 재생·취소/오류 재시도·새로고침 복원을 확인했다.14+2Worker가 모두 종료됐다. 한영 입력 손상·표시 손상·저장 불가6흐름까지,총7흐름/7접근성 검사를 통과했다. 일부 오류/취소 재현은 postMessage 보류/오류 주입을 사용했다.
- 생산 Next 계산/검산 자산의 HTTP바이트와 디스크 SHA-256을 대조했다. 독립Python51기준과20예제를 포함한71요청의 **전체 응답 해시**가 일치했다. 수치 변조71건과 요청 귀속 변조71건을 거부했고 배치2Worker를 종료했다. 예제 반복 요청을 포함하며 서로 다른71모형이라고 세지 않는다.
- 1837생성HTML의 내부 링크·자산·앵커·사이트맵·canonical 검사를 통과했다. 이전65Worker자산을 로컬 출력에 바이트 그대로 보존하고 이전4541파일을 백업한 뒤,4547파일의 검증 출력을 로컬 홈페이지에 반영했다. 이 로컬 자산 보존이 GitHub Actions의 과거 배포 자산 보존을 보증하지는 않는다.

## 실행과 증거

```bash
npx tsx --test scripts/learning-euclidean-kernel-reader-integration.test.ts
npm run lint
npm run typecheck
npm run validate:content
npm run build
npm run check:export
EUCLIDEAN_READER_LANGUAGE=en EUCLIDEAN_READER_THEME=dark npx tsx scripts/check-euclidean-kernel-reader.ts
npx tsx scripts/check-euclidean-kernel-reader-maximum.ts
npx tsx scripts/check-euclidean-kernel-production-assets.ts
```

브라우저 검사의 기본 주소는4202포트이며`EUCLIDEAN_KERNEL_PREVIEW_BASE`로 바꾼다. 생산 자산 검사는`EUCLIDEAN_KERNEL_EXPORT_ROOT`로 디스크 출력 경로를 바꾼다. 증거는`.runtime/learning/euclidean-kernel-reader-integration/verification.json`과 각 로그/JSON/그림,이전211검사는`.runtime/learning/euclidean-kernel-teaching/`에 있다. 독립 검산은 표현 한도를 유지하며 Mehler 소수 그림은 인증 오차 구간이 아니다.

첫 생산 자산 검사는 검증 스크립트 이식 중`toString(20)`으로 SHA-256을 인코딩해 비교에 실패했다.16진수로 고치고 입력 해시 사전 대조·각 요청 진행 기록을 추가한 뒤71요청을 재실행했다. 첫 실패 로그도 보존했다. 모바일 검사의 alert 선택자는 Next의 전역 route announcer와 겹쳐 실험 영역으로 좁힌 후 전체7흐름을 재확인했다. 두 수정은 검사 도구의 수정이다.

## 다음 순서

이 단원의18요건은 충족했지만 원래 경로적분 주제는 미완료다. 다음은`안장점·요동·비가우스 근사`의 일차 자료와 모형 계약,한영 상세 원고/수치/검산/학습 화면/reader다. 그다음`실시간 위상과 고전 극한`,이후 네 처리의 입문 범위를 감사한다. 원주제 완료 전에 입문 미완료47개를 줄이지 않는다.
