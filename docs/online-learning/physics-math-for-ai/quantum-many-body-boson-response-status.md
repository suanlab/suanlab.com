# 다체계 응답 검증 세 번째 종류: Raw boson Fock

2026-11-09. `boson-fock`의 **소유 구조·요청 귀속·정확 field 응답 검증기**를 구현했다. Spin·fermion과 함께 여섯 계산 종류 중 셋이 준비됐으며 coherent·MPS·Schmidt는 남아 있다. 다섯 원고와 수치 가족의 준비를 실제 학습 단원 등록과 구별한다.

## 구현한 내용

`readBosonFockSuccessForInput`은 입력에서 국소 ladder·Kronecker H·moments·밀도/γ·number·finite CCR·quadrature·probe·유한 대각 spectrum을 도출해 전체 응답과 대조한다. Q(√2)[√3]의 이차쌍 연산과 정확 부호 비교로 계수·부호·실제 구간·정규화를 확인한다. 제품 field/Fock/root/interval engine을 reader 번들에 넣지 않는다.

[정확 field·cutoff 감사](quantum-many-body-boson-response-audit.md)에 경계항·확률·실제 관측의 의미를 기록했다. 공통 소유 복사 한도는 정상 최대 응답396196방문을 수용하도록500000으로 조정했고, 좁은 구간의 미해결 metadata 위조도 거부하도록 보완했다. 기존 수치 engine·원고는 변경하지 않았다.

## 검증 결과

- 새94검사(raw41·field53)와593변조 거부, 기존319회귀·타입·lint를 확인했다.
- Raw33독립 준비에서 정확201702·방향성2079관측을 대조했다.
- 독립 field52준비에서1248계수·312부호·156정규화 구간을 대조했다.
- Native Worker120개 생성/종료, raw33·이전87의 브라우저 검증기 수락과 wrong ID/metadata 각각120개 거부를 확인했다.

근거는 `.runtime/learning/many-body-boson-response/{verification.json,metrics.json,field-metrics.json,tests.log,native-verification.json,validation-dependency-graph.json,historical-graph-preservation.json}`과 소스/근거 manifest다. Native는 격리된 preview-origin Blob Worker이며 실제 reader·생산 자산이 아니다. 최대 raw 검증 약1.12초는 성능 관문이 남았다는 근거이며 실제 화면의 반응성 완료로 세지 않는다.

## 다음 순서와 전체 범위

**Coherent 응답 → MPS 응답 → Schmidt 응답 → controller/hooks·반응성 → 학습 화면·실제 reader**로 이어간다. 여섯 종류를 모두 검증하고 반응성을 확보하기 전에는 학습 화면에 공통 성공 결과로 연결하지 않는다.

기존566개 한영 단원·1698개 안내, 원래 입문308/362개 검증·54개 미완료, 전체51영역·509주제와 연구 깊이를 유지한다. 홈페이지 출력·미리보기 서버를 보존하며 commit·push·공개 배포는 수행하지 않았다. 진입은 **도서 → Interactive Book**이다.
