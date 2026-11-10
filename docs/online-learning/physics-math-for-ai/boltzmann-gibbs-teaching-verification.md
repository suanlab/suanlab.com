# Boltzmann·Gibbs 교수 화면 격리 검증

2026-11-07. 세 단원의 실제 교수 컴포넌트를 구현했다. 이 기록은 기존 사이트의 CSS·폰트와 실제 계산 worker를 사용하는 **격리 화면**의 검증이다. Next 읽기 페이지·worker 자산 통합은 아직 검증하지 않았으며, 등록 수는518단원·1554안내, 원래 입문 주제는285/362검증·77미완료를 유지한다.

## 구현한 교수 자료

- `learning-boltzmann-gibbs-concepts.ts`: 한영 직관 설명, 고정 개념12단계·준비33개, 해설 퀴즈6개. 실험 저장값과 독립적으로 계산한다.
- `boltzmann-gibbs-concept.tsx`: 준비별 비교표·확률 막대·조건부 전이 그래프·Gaussian 타원과 좌표 선택. 연속 미분 엔트로피와 유한 목록 엔트로피를 분리한다.
- `boltzmann-gibbs-lab.tsx`: 안내9개·예제27개, 엄격한 수치/목록 입력, 저장 복원, 취소·재시도, 비교 순회·수동 단계.
- `boltzmann-gibbs-view.tsx`: 실제0·미해결·범위 밖 비영을 구별한 표, 이론적 kernel·유한 단계 법칙·양자화 Q·seed 경로를 각각 표시한다.
- `learning-boltzmann-gibbs-visuals.ts`: 공분산의 로그 크기와 조건부 분산으로 중심 상대 타원을 만든다. 정확한 정규화 판정과 시각 근사를 구분하며 비정부호이면 구름을 그리지 않는다. 양의 정부호여도 픽셀보다 얇으면 수치와 설명을 유지한다.
- `boltzmann-gibbs-cover.tsx`: 기존 색을 쓰는 장식 표지. 계산을 실행하지 않는다.

개념 예제와 실험 결과의 접근성 영역 이름을 구분했다. 입력 오류·현재 textarea 값·select 선택지가 컨트롤 이름에 섞이지 않도록 명시적인 이름을 연결했다.

## 검증 범위

렌더링·고정 개념·퀴즈·타원 기하·SSR·접근성 이름47검사를 통과했다. 실제 worker를 연결해3단원×한영×명암×320/1440px의24조합을 확인했다. 각 조합에서4개 개념 단계, 모든 고정 준비, 안내3개, 예제9개, 퀴즈6선택을 조작하고 페이지 접근성·넘침·영어 한글 혼입·유효한 SVG를 검사했다.

추가 조작 검사는 비영 underflow 거부, 저장 후 재로드, 취소·worker 생성 실패·재시도, 지원 변경 시 초기 상태 범위, 다섯 kernel, 최대 seed/32비트/2048단계 경로, 경로 수동/자동 단계, 전역 중지·감소 모션·화면 밖 중지, 비가역 좌표 거부·비정규화 구름 보류, 언어/테마 변경, 저장 거부 시 실험·퀴즈 동작을 다룬다. 개념96단계·준비264선택·안내72회·예제216회·퀴즈144선택과 접근성24검사를 통과했다. UI 검사480개와 추가 조작 검사29개, 총509개 worker를 모두 종료했고 잔존·stale 노출·page error·자산 로드 오류는0이다. 임시 서버와 소유 브라우저도 종료했다.

증거: `.runtime/learning/boltzmann-gibbs-teaching/verification.json`, `browser-result.json`, `controls-result.json`, `bundle-boundary.json`, `panel-tests.log`, `lint.log`, `typecheck.log`. 이전 과학 모델·한영 원고·worker의4082파일과 기존 정적 출력4257파일은 해시로 보존을 확인한다.

## 재현과 다음 단계

```bash
npx tsx --test scripts/learning-boltzmann-gibbs-panels.test.ts
node scripts/learning/build-boltzmann-gibbs-teaching-probe.mjs
python3 -m http.server 4187 --bind 127.0.0.1 --directory .runtime/learning/boltzmann-gibbs-teaching/browser
# 별도 터미널에서 실행한 뒤 임시 서버를 종료한다.
node scripts/check-boltzmann-gibbs-teaching-browser.mjs
node scripts/check-boltzmann-gibbs-teaching-controls.mjs
```

사용자 미리보기4175는 그대로 유지한다. harness의 worker URL 매핑은 esbuild 전용이므로 Next 자산 경계의 증거로 사용하지 않는다. 다음은 기존518단원의 값·순서·안내 스냅샷을 보존하면서3단원과 안내9개를 append하고, 개념·실험·표지 dispatcher와 교육과정을 연결하는 작업이다. 이어 전체 검사·정적 빌드·실제 읽기24조합·검색·메뉴·breadcrumb·sitemap·Next worker 경계를 검증한다. 그 전에는 주제 완료 수를 늘리지 않는다.
