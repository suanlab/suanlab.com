# Euclidean 커널: 실제 Worker와 요청·수명 검증

2026-11-10. 정확분수 계산기와 독립 검산을 별도 모듈 Worker로 연결했다. **관련201검사가 Node.js20/24에서 모두 통과**했고 실제 브라우저71요청·React 연결·진행 중 취소를 확인했다. 한영 원고와 수치 기반은 준비됐지만 개념 그림·학습 UI·실제 reader·생산 Next 자산은 아직 없다.598등록 단원·1844안내·등록1196원고·미등록 초안1개를 유지한다.

## 요청과 소유권

계산/검산 entry는 각각 descriptor 기반 요청 검증과 기존 bounded wire 복사를 사용한다. version·양의 안전 정수requestId·kind·inputKey·원래 분수 표기·중첩 결과의 입력이 모두 일치해야 한다. getter·prototype·과대 데이터와 현재 요청의 잘못된 envelope는 오류로 끝난다. 필드 삽입 순서는 바뀌어도 같은 입력 키를 유지한다.

실제 수학 검산은 독립 Worker에서만 한다. hook/controller의 번들에는 생산 계산기·독립 검산·정확 산술이 없다. SSR은 Worker 전역을 읽지 않고 idle을 렌더한다. 검산 Worker의 runtime 의존 그래프에도 생산 계산기와 생산 산술이 없다.

## 수명과 복구

계산 Worker를 종료한 후 검산 Worker를 만들고,검산이 끝나면 그 포트도 종료한다. 응답의 소유 복사와 승인 결과의 소유 복사는 짧은 작업으로 나누며 활성 token·취소·교체·기한을 매번 확인한다. 기본 전체 기한은120초다. 이전 요청의 callback은 새 입력을 덮어쓸 수 없다.

24개 controller 검사는 취소/교체·복사 중단·지연 callback·기한·검산 포트 생성 실패·postMessage/onerror/onmessageerror·자원 초과·dispose를 포함한다. 복사·검산·최종 승인 중의 기한 초과도 확인했다. React hook은 이전 입력의 결과를 즉시 숨기고 숨김/비활성/unmount에서 종료하며 복귀/재시도 시 새 요청을 사용한다.

## 실제 브라우저 증거

- 별도 Python51기준과20예제를 포함한71계산/검산 요청을 실제 Chromium module Worker에서 수행했다. 반복 예제를 포함한 수이며 독립 입력71개라는 뜻은 아니다. 전체 결과의 SHA가 Python 기준과 일치했다. 수치 차원 변조71건·요청 ID 불일치71건을 거부했고2Worker가 모두 종료됐다.
- React 연결은 가장 무거운 서로 다른 분모의32구간 입력에서 계산 중 타이핑,숨김 시작/검산 취소·복귀·입력 교체·잘못된 입력의 무Worker 오류·필드 순서·주입한 postMessage 실패/재시도·pending unmount를 통과했다. 생성19/종료19·활성0이었다. 이 검사에는 의도적인 postMessage 보류와 visibility 상태/이벤트 모의가 포함된다.
- 별도`active-cancel` 흐름은 postMessage를 보류하지 않았다. 실제 계산 started 후 취소하고,다음 요청의 검산 전송 후에도 취소했다. 두 단계에서 타이핑이 가능했고 취소100ms 뒤 idle을 유지했다. 마지막 재시도의 전체 결과를 Python 기준과 대조했다. 생성7/종료7·활성0이었다.
- 물리적 탭 숨김은 자동화하지 않았다. 실제 reader/생산 Next 자산 검증과 구분한 로컬 esbuild 모듈 Worker 검사다. 최대 입력·전체1089공분산 항목·정확분수·기존 wire 한도를 유지했다.

## 재현

```bash
npx tsx --test --test-concurrency=2 scripts/learning-euclidean-kernel-{numerics,validation,manuscripts,workers,controller}.test.ts
npx tsx scripts/check-euclidean-kernel-workers-native.ts
npx tsx scripts/check-euclidean-kernel-hook-native.ts
EUCLIDEAN_KERNEL_NATIVE_FLOW=active-cancel npx tsx scripts/check-euclidean-kernel-hook-native.ts
```

브라우저 검사는 Playwright Chromium과4175포트의 로컬 정적 미리보기가 필요하다. Worker 검사가 생성한`calculate.js`,`validate.js`를 hook 검사에서 사용한다. 증거는 `.runtime/learning/euclidean-kernel-worker/`의 전체/Node20 검사 로그·의존 그래프·native Workers/hook/active-cancellation JSON이다. TypeScript·ESLint·콘텐츠 검사와 기존 홈페이지 출력 보존도 확인한다.

## 게시와 다음 단계

앞 Brownian 통합의 GitHub 실행42753006350은 테스트/빌드는 성공했지만 Pages 배포의10분 polling 기한에서 중단됐다. 같은 유형의 앞 실패42746345597도 확인했다. [공식 deploy-pages 설정](https://github.com/actions/deploy-pages/blob/v4/action.yml)의 millisecond 단위를 확인하고 액션 기한20분·외부 step 기한22분으로 조정한8de1980을 push했다. 이 변경만으로 공개 배포 성공을 선언하지 않는다.

다음은 기존 홈페이지 스타일의 한영8개념·설명 그림·2D/Three.js·입력/표시/학습 저장·예측 안내/퀴즈다. 이후 실제 Interactive Book 등록과 생산 자산을 검증한다. 경로적분 두 번째 처리의 구현13/18요건을 확인했으며 원주제·전체51영역509주제·영역 감사·연구 심화는 미완료다. 입문315/362검증·47미완료를 유지한다.
