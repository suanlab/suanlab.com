# 다체계 응답 검증 네 번째 종류: Coherent projection

2026-11-09. `boson-coherent`의 **소유 구조·정확 준비·무한/유한 비교 응답 검증기**를 구현했다. Spin·fermion·raw boson과 함께 여섯 계산 종류 중 넷이 준비됐으며 MPS·Schmidt는 남아 있다. 다섯 원고·수치 가족과 실제 학습 단원 등록을 구별한다.

## 구현한 내용

`readBosonCoherentSuccessForInput`은 유리수 α·Poisson 가중치·정확 field 계수에서 전체 준비를 만들고 기존 유한 boson 검증 계산으로 모든 cutoff 관측을 대조한다. Infinite coherent 상태·비정규화 투영·재정규화 상태는 별도 schema와 수학을 가진다. Fake raw parameters나0으로 지운 경계 잔차를 허용하지 않는다.

직접512차 교대 Taylor 합과2048비트 방향성 경계로 exp(−x)를 독립 인증한다. 생산 transcendental/root/field 관측 engine을 가져오지 않고 실제 값의 인증 구간이 응답 안에 포함되는지 확인한다. Fidelity 제곱·overlap·tail·정규화 거리와 두 모드 곱 관계도 각각 검사한다. [가정·인증 감사](quantum-many-body-coherent-response-audit.md)에 자세히 기록했다.

## 검증 결과

- 새42검사·863변조 거부와 기존413회귀·타입·lint를 확인했다.
- 독립 coherent32준비의 정확178986·방향성5221관측, 독립 exp 기준102회를 대조했다.
- Native Worker152개 생성/종료에서 coherent32·이전120의 브라우저 검증기 수락과 wrong ID/metadata 각각152개 거부를 확인했다.
- 네 종류의 전체 정확428617·방향성8770관측을 독립 기준과 대조했다.

근거는 `.runtime/learning/many-body-coherent-response/{verification.json,metrics.json,tests.log,native-verification.json,validation-dependency-graph.json,historical-graph-preservation.json}`과 소스/근거 manifest다. Native는 격리된 preview-origin Blob Worker이며 실제 reader·생산 자산·controller가 아니다. 최대 약1초인 coherent 검증 시간을 실제 화면 성능 통과로 세지 않는다.

## 다음 순서와 전체 범위

**MPS 응답 → Schmidt 응답 → controller/hooks·반응성 → 학습 화면·실제 reader**로 이어간다. 여섯 종류를 검증하고 반응성을 확보하기 전에는 실제 학습 화면에 공통 성공 처리로 연결하지 않는다.

기존566개 한영 단원·1698개 안내, 원래 입문308/362개 검증·54개 미완료, 전체51영역·509주제와 연구 깊이를 유지한다. 홈페이지 출력·미리보기 서버를 보존하며 commit·push·공개 배포는 수행하지 않았다. 진입은 **도서 → Interactive Book**이다.
