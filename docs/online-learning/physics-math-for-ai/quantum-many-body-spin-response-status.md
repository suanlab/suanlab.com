# 다체계 응답 검증 첫 가족: Spin chain

2026-11-09. `spin-chain`의 **소유 구조·요청 귀속·수학 응답 검증기**를 구현했다. 여섯 계산 종류 중 하나이며, fermion·boson raw/coherent·MPS·Schmidt는 남아 있다. 다체계 다섯 수치 가족·한영 원고는 준비됐지만 학습 단원으로 등록하지 않았다.

## 구현한 것

- 공통 descriptor-only 복사: 소유 데이터·공유 subtree·동결·cycle/array/prototype/accessor와 복잡도 상한을 처리한다.
- 정확 schema와 수학 귀속: 모든 root/nested field·배열·파라미터·ID/key·관측 metadata·정규화 좌표를 확인한다.
- 독립 spin 계산: Kronecker H·직접 partial trace·모멘트·잔차·pair 상관·Z/parity·대각 spectrum과 ground/gap·비용을 입력에서 도출한다.
- `readSpinChainSuccessForInput`은 검증된 동결 복사본을 반환한다. 원래 packet이나 공급된 residual을 신뢰하지 않는다.

[구조·수학 감사](quantum-many-body-spin-response-audit.md)에 범위와 변조 검사를 기록했다. `learning-many-body-{owned-wire,wire-match,spin-validation,contracts}.ts`는 무거운 engine을 reader 번들에 넣지 않는다.

## 검증 범위

독립42준비의 정상 응답·정확16024/방향성775관측을 대조하고 잘못된 유리수·필드·확률·분류·다른 준비 결과 등을 거부한다. 새51검사·1150변조 거부와 기존 Worker214회귀·타입·lint를 확인한다. Native Worker42개를 생성·종료하고 브라우저 검증기로42결과를 수락, wrong ID42개·오염 metadata42개를 거부했다. Native 결과도 독립 수학 기준과 대조했다.

근거는 `.runtime/learning/many-body-spin-response/{verification.json,metrics.json,tests.log,native-verification.json,validation-dependency-graph.json}`과 소스/근거 manifest다. Native는 격리 preview-origin harness의 Blob Worker이며 실제 reader·생산 자산·controller 검증이 아니다. 최대 검증 시간 관측값은 약119ms로 실제 화면의 성능 관문은 남아 있다.

## 다음 순서와 전체 범위

**Fermion 응답 → boson raw/coherent 응답 → MPS 응답 → Schmidt 응답 → 취소 controller/hooks → 학습 화면·실제 reader**로 이어간다. 모든 응답 가족의 검증이 끝나기 전에는 공통 성공 처리나 reader 연결을 하지 않는다.

기존 **566개 한영 단원·1698개 안내, 원래 입문308/362개 검증·54개 미완료**, 전체51영역·509주제와 연구 깊이를 유지한다. 홈페이지 정적 출력과 미리보기 서버를 보존하며 commit·push·공개 배포는 수행하지 않았다. 진입은 **도서 → Interactive Book**이다.
