# 다체계 Worker 기반: 소유 요청·계산 분리

2026-11-09. 다섯 수치 가족·다섯 한영 원고 뒤에 **요청·저장·계산 dispatch와 실제 Worker 실행 기반**을 추가했다. Boson의 raw/coherent 두 경로를 포함한 여섯 종류다. **응답 구조/수학 검증기·취소 controller·학습 화면·reader 등록은 아직 남아 있다.**

## 구현과 검증

`learning-many-body-{input,protocol,evaluator,worker-factory}.ts`와 `learning-many-body.worker.ts`를 추가했다. Canonical 동결 입력·소유 descriptor·종류별 유효성·어휘 정수 JSON·저장 version·request ID를 검사하고, 유효한 요청에 started와 결과를 보낸다. 입력 snapshot 뒤의 caller 변경은 진행 중 요청을 바꾸지 않는다.

새214검사가 통과했다. 독립 기준207준비의 입력/저장/identity, 잘못된 envelope·종류 혼합·추가/symbol/accessor·상속·reflection·JSON 반올림 등을 검사했다. Reader용 요청/factory 번들에 무거운 엔진이 없고 SSR import가 Worker를 생성하지 않는다.

Node의 bundled 실제 entry로207준비를 실행해 전체 독립 수학 기준과 프레임을 대조했다. 정확464907·방향성18274관측과10260개 Schmidt 구간 잔차 대조를 포함한다. 이 수치는 여섯 dispatch 직접 기준과 Node Worker 전체 기준의 합이다. Chromium의 네이티브 module Worker에서도207준비를 각각 생성·종료하여 같은 독립 기준을 대조한다. 타입 검사·lint도 확인한다.

증거는 `.runtime/learning/many-body-worker/{tests.log,node-worker-verification.json,native-verification.json,observation-metrics.json,request-dependency-graph.json,verification.json}`과 소스/근거 manifest다. Native 실행은 격리된 preview-origin harness와 Blob 번들이다. 실제 reader·새 Next 생산 자산·취소 controller를 검증했다고 세지 않는다.

## 남은 작업과 전체 범위

[요청·dispatch 계약](quantum-many-body-worker-contract.md)의 다음 관문에 따라 정상 응답뿐 아니라 오염된 응답을 거부하는 검증기를 구현한다. 이후 controller·hooks·개념/실험/예측·세 안내·해설 quiz·저장·Three.js/접근 가능한 표와 실제 reader를 연결한다. 다섯 원고와 수치 가족은 모두 준비됐지만 등록된 다섯 완성 단원은 아니다.

기존 **566개 한영 단원·1698개 안내, 원래 입문308/362개 검증·54개 미완료**와 전체51영역·509주제·연구 깊이를 유지한다. 홈페이지 정적 출력·미리보기 서버를 보존하며 commit·push·공개 배포는 수행하지 않았다. 새 콘텐츠 진입은 **도서 → Interactive Book**이다.
