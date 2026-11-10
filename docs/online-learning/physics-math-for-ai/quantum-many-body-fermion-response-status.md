# 다체계 응답 검증 두 번째 가족: Fermion Fock

2026-11-09. `fermion-fock`의 **소유 구조·요청 귀속·수학 검증기**를 구현했다. Spin과 함께 여섯 계산 종류 중 둘이 준비됐고 boson raw/coherent·MPS·Schmidt는 남아 있다. 다섯 수치 가족·상세 한영 원고와 실제 학습 단원 등록을 구별한다.

## 구현한 내용

`readFermionFockSuccessForInput`은 전체 packet을 소유 descriptor로 동결 복사한 뒤 입력에서 만든 정확 의미 트리와 대조한다. 제품 bit-flip engine 없이 JW Kronecker 행렬·CAR·복소 H/역순 pairing adjoint·모멘트·잔차·γ/anomalous·number·parity·twirl·probe·naive 비교를 계산한다. 대각 full Fock/sector 최소와 비대각 null도 확인한다.

[구조·부호 감사](quantum-many-body-fermion-response-audit.md)에 가정·확률의 의미·변조 범위를 기록했다. 공통 matcher와 소유 복사는 유지했으며 기존 contracts 모듈에는 fermion 계약 문자열만 추가했다. 수치 engine과 기존 원고는 변경하지 않았다.

## 검증 결과

새 **54검사·541변조 거부**, 독립45준비의 정확31905·방향성695관측을 대조했다. 기존 **265회귀**(spin 응답51·Worker214), 타입·lint를 확인한다. 소멸자 row/sign·생성 순서·pairing adjoint·formal coherence·twirl과 sector/full ground를 잘못 바꾼 응답도 거부한다.

Native Worker45개를 생성·종료하고 브라우저 검증기로45결과를 수락했다. Wrong ID45개·오염 metadata45개를 거부하고 독립 수학 기준과 대조했다. 검증 번들에는 제품 계산기·root isolation·로그 interval engine이 없으며 최종 번들의 native 검증 bytes 일치도 확인했다.

근거는 `.runtime/learning/many-body-fermion-response/{verification.json,metrics.json,tests.log,native-verification.json,validation-dependency-graph.json,final-validator-bundle-match.json}`과 소스/근거 manifest다. 격리된 preview-origin Blob Worker이며 실제 reader·생산 자산·controller가 아니다. 최대 검증 시간 약178ms라는 관측만으로 실제 화면 성능을 통과 처리하지 않는다.

## 다음 순서와 전체 범위

**Boson raw 응답 → boson coherent 응답 → MPS 응답 → Schmidt 응답 → controller/hooks → 학습 화면·실제 reader**로 이어간다. 여섯 종류의 검증이 모두 끝나기 전에는 공통 성공 처리와 reader 연결을 하지 않는다.

기존 **566개 한영 단원·1698개 안내, 원래 입문308/362개 검증·54개 미완료**, 전체51영역·509주제와 연구 깊이를 유지한다. 홈페이지 정적 출력과 미리보기 서버를 보존하며 commit·push·공개 배포는 수행하지 않았다. 진입은 **도서 → Interactive Book**이다.
