# 국소 기준·U(1) 연결·Wilson 루프의 실제 교재 통합

2026-11-10. `graph-u1-local-frames-wilson-loops`를 실제 Interactive Book에 등록하고 검증한 정적 출력을 로컬 홈페이지에 반영했다. **588등록 단원·1782안내 실험**이다. 대칭·게이지 원주제의 첫 단원이므로 기존 입문 검증은 **312/362·50미완료**로 유지한다. 전체51영역509주제·영역 감사·연구 심화도 미완료다.

## 연결한 학습 내용

한영16절 원고,8개념·6안내·7해설 문제·13예제와 본문/실험을 연결했다. 별도 학습 화면을 reader에 삽입하고 중복 개념/퀴즈는 숨겼다. 그림8개는 국소 기준·공변/불변·삼각형의0대8·닫힌/열린 곱·불완전한 관측·Hermitian 행렬·분리 성분·정확한Cayley위상을 각각 설명한다. 표지도 같은 기존 사이트 색상과 SVG 체계를 쓴다.

선수 단원은 복소 극형식,Wirtinger미분,Noether조건이다. 직전Fourier단원과 읽기 순서,전체 교육과정 지도,단원 카드와 한영 검색을 연결했다. 진입은 **도서 → Interactive Book**이며 Online Book 페이지에는 콘텐츠 카드를 추가하지 않았다.

## 실제 검증

- 관련149검사와 타입·lint·콘텐츠1823URL·정적1826HTML 검사를 통과했다.588단원의1176원고를 strict KaTeX로 렌더링했다.
- 이전587단원/안내의 내용과 순서를 해시로 대조하고51영역의 기존 주제 범위를 보존했다. 역사적 단원 검사의 출력은 현재 등록 수별로 분리하여 이전 검증 기록을 보존한다. 이전 스칼라장·Noether·Fourier의 생산Worker6자산도 바이트가 같다.
- 실제 reader의 한영×명암×320/1440px8조합,64개념 전환·56문제 해설·접근성8회,입력/퀴즈/완료 저장 복원과3D/WebGL복구를 확인했다.
- 실제 생산Worker65입력(독립Python참조 입력49·예제13·최대예산3),에너지 변조65거부·요청 귀속65거부를 확인했다. 모든 결과의 전체 해시를 비교하고 독립 수치 참조는 기반 검사로 대조한다.
- 취소·다시 계산·postMessage오류·재시도·계산 중 이탈에서9Worker 생성/종료가 일치했다.
- 모바일 기준 그림16상태의 글자 경계가 SVG안에 있고 실제 글자가17.5px이상임을 측정했다. 두 언어의 메뉴/검색/선수 경로와 기존 세 reader의 기본 에너지7/2,7/2,5/2를 확인했다.

별도 빌드에서 누락됐던 원본Tailwind/PostCSS설정을 포함하고 이전 캐시를 보존한 뒤 새로 빌드했다. 최종 CSS의 미처리Tailwind지시문이 없고 실제 사이트 유틸리티가 생성됨을 확인한 후 모든 브라우저 검사를 다시 수행했다. 최초 설정 누락 빌드의 검사는 최종 근거로 사용하지 않는다.

## 반영과 보존

현재 경로는 `/book/online/physics-math-for-ai/graph-u1-local-frames-wilson-loops/`다. 별도 후보 빌드를 검증한 후 소스와 출력을 반영했다. 이전4467출력 파일은 `.runtime/learning/graph-gauge-reader-integration/pre-adoption-export/`에 보존하고,현재4473파일은 검증한 후보 출력과 일치한다. 기존 소스의 의도한 변경 전 사본도 같은 디렉터리에 보존한다. 근거는 `verification.json`, `compiled-style-proof.json`, `reader.json`, `production-assets.json`, `reference-diagrams.json`, `discovery.json`이다. commit/push·공개 배포는 수행하지 않았다.

## 다음 집필

다음은 `gauge-orbits-tree-fixing-boundary-anchors`의 범위·일차 자료 검토와 집필이다. 트리 게이지 고정,독립 순환,잔여 위상,외부 기준과 비식별성을 다룬다. 이어 `graph-gauss-constraints-physical-phase-space`에서 정준 변수·전하·경계·Gauss제약을 준비한다. 세 단원을 원래 요구사항에 대조한 뒤 대칭·게이지의 입문 완료 여부를 판단한다. 장이론의 다른8원주제와 나머지4영역41주제,전체 감사·연구 깊이 검증도 계속 남아 있다.
