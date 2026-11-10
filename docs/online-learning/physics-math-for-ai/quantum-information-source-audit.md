# 양자정보 첫 집필의 자료와 검증 경계

2026-11-09 통합 결과: [아홉 단원의 실제 reader 검증](quantum-information-integration-status.md)을 완료했다. 아래의 유한 모형·가정·자료 범위를 유지하며 입문 검증을 일반 보안·채널 용량·연구 심화 완료로 확대하지 않는다.

2026-11-09. 공식 공개 일차 자료의 관련 절을 확인했다. 교재 PDF 전체를 읽거나 재배포한 것으로 기록하지 않는다. 원고는 자체 설명과 계산 예제를 사용하며 책의 본문·그림을 복제하지 않는다.

| 자료 | 확인한 범위 | 적용 |
|---|---|---|
|[Watrous, TQI 3장](https://cs.uwaterloo.ca/~watrous/TQI/TQI.3.pdf)|3.1.1의 이진 상태 판별, 3.2의 fidelity 정의|가중 trace norm과 최적 측정; root fidelity와 본문의 squared fidelity 구별|
|[Watrous, TQI 5장](https://cs.uwaterloo.ca/~watrous/TQI/TQI.5.pdf)|엔트로피 정의·부분계 정보량·정보원 부호화·5.3.3 Holevo bound의 관련 부분|로그 밑2, 상태와 측정의 구분, IID/충실도 조건, 측정 정보량과 상계의 구별|

첫 장의 실제 수치 검증은 SymPy 행렬 고유분해·행렬 제곱근 정의에서 생성한 독립325사례다. 제품 코드의 Bloch 폐쇄식으로 참조를 계산하지 않는다. 생성 도구는 `scripts/learning/generate-quantum-discrimination-references.py`, 참조는 `scripts/fixtures/quantum-discrimination-independent.json`이다.

둘째 장의 유한 블록 검증은 독립 bit 문자열 열거다. 본문의 두 복사본 예제는 고유값 `(16,4,4,1)/25`에서 직접 계산한다. 이 계산을 일반적 유한 블록 양자 압축 정리나 부호기 구현의 증거로 확대하지 않는다.

BB84·텔레포테이션·오류 정정·LOCC·용량 단원은 집필 때 각 정리의 일차 자료와 가정·기저 순서·자원 규약을 다시 대조한다. 특히 과거 강의 노트의 미해결 문제를 2026년 현재의 연구 현황으로 옮기지 않는다. 일반 용량·보안의 최신 연구 주장에는 별도 최신 자료 점검이 필요하다.

## 앙상블·상대 엔트로피 집필 후 추가 확인

2026-11-09. 위 공식5장의5.3.3절 Theorem5.49와 단조성 Theorem5.35를 대조했다. 전자는 선택한 측정의 정보량과 접근 가능 정보·Holevo 정보의 관계에, 후자는 두 가설에 같은 CPTP 채널을 적용하는 조건에 사용한다. 두 새 본문의 행렬·확률표·계산과 반례는 자체 작성했다. 고정밀 행렬 참조448개와 방향 구간을 대조하며, 수치 예제를 일반 정리의 증명으로 기록하지 않는다.

## 복제 불가·BB84 및 두 통신 프로토콜

2026-11-09. [BB84 원 논문의 아카이브와 공개 스캔](https://arxiv.org/abs/2003.06557)에서 비직교 신호, 기저 측정·교란과 고전 채널 가정을 확인했다. 2020년은 아카이브 제출 시점이며 원 논문은1984년이다. 고전 채널의 수동/능동 도청 구분을 현재 본문의 인증 요구와 연결하되 현대적 유한 키 보안을 증명한 자료로 확대하지 않는다.

[Bennett 등의 텔레포테이션 출판 기록](https://research.ibm.com/publications/teleporting-an-unknown-quantum-state-via-dual-classical-and-einstein-podolsky-rosen-channels)과 [Bennett–Wiesner의 초밀집 부호화 출판 기록](https://research.ibm.com/publications/communication-via-one-and-two-particle-operators-on-einstein-podolsky-rosen-states)의 저자기관 초록·날짜·자원 구성을 확인했다. 이 두 기록은 전체 논문을 읽었다는 증거로 기록하지 않는다. 본문의 레지스터 순서·가지 전개·복소 예제·연산자 순서·참조계 항등 증명은 자체 유도하고 독립 조밀 회로와 대조한다. 원문 PDF·그림을 저장하거나 재배포하지 않았다.

## 오류 정정과 순수 상태 얽힘 자원

2026-11-09. [Preskill의 공식7장](https://www.preskill.caltech.edu/ph219/chap7_26_6A.pdf)7.2–7.3절의 오류 정정 조건(7.17)과 참조계 분리 관점을 대조했다.3큐비트 반복 코드에 대한 자체 행렬·syndrome·논리 채널 계산을 구성했으며 노트 전체나 표면 코드·임계값·현재 연구 현황까지 검증했다고 기록하지 않는다. 개인 열람용 PDF를 저장·재배포하지 않았다.

[Nielsen 원 논문](https://arxiv.org/abs/quant-ph/9811053)의 주요화 정의와 순수 상태 전환 방향, [Vidal 원 논문](https://arxiv.org/abs/quant-ph/9902033)의 최적 꼬리비와 운영적 가정을 확인했다. 논문 렌더링에 나타나는2024년 날짜를 최초 출판 시점으로 사용하지 않는다. 자체 비교 불가능 쌍과 실제 Bell 희석 절차를 작성하고, 결정론적100쌍은 누적합과 다른 정확 순열 볼록 껍질 판정으로 대조한다. 확률적 비율은 Vidal 정리의 공식 대조이며 독립 프로토콜 합성이나 실행 검증이라고 주장하지 않는다.

## 채널 용량과 결합 엔트로피 보완

2026-11-09. [Preskill의 공식 Shannon 이론10장](https://www.preskill.caltech.edu/ph219/chap10_6A_2025.pdf)의10.7–10.8절에서 정규화·degradability·일반화 탈위상·보조 자원 구분을 확인했다.2025년 노트의 연구 미해결 현황을2026년 최신 사실로 옮기지 않는다. [Devetak–Shor 원 논문](https://arxiv.org/abs/quant-ph/0311131)의 공식 초록과 일반화 탈위상의 단일 문자 결과, [BSST 원 논문](https://arxiv.org/abs/quant-ph/0106052)의 공식 초록에서 얽힘 보조 용량의 입력/출력/정화 엔트로피식을 확인했다. 전체 논문을 읽었다고 기록하지 않는다. 본문의 큐비트 환경·degrading map·Pauli 평균·끝점·수치 예제는 자체 유도했다.

결합 상태의2x2 PPT 필요충분성은 [Horodecki 등의 원 논문 공식 초록](https://arxiv.org/abs/quant-ph/9605038)과 대조했다. 이를 높은 차원으로 확대하지 않는다. 직접 구성한 두 매개변수 가족의 고유값·부분전치·조건부 정보는 독립 행렬 계산과 대조한다. 원문 PDF나 그림을 저장·재배포하지 않았다.

## 독립 엔트로피 로그 평가기

2026-11-09. [NIST DLMF4.6.1](https://dlmf.nist.gov/4.6.E1)의 ln(1+z) 급수를 대조했다. 자체 검증기는 생산 atanh evaluator와 달리 signed log1p 급수·정확한 범위 축소·방향 고정소수점 반올림을 사용한다. |u|≤1/2에서 N항 뒤의 절대 꼬리를2^-N/(N+1)로 감싸는 상계는 기하급수로 직접 유도했다.608항·512bit 출력의 코드와 고정밀43점 corroboration을 구분하며, 십진 참조를 엄밀한 구간 증명으로 과장하지 않는다.

## 앙상블 응답 검증

2026-11-09. 공식 [Watrous3장 Theorem3.4](https://cs.uwaterloo.ca/~watrous/TQI/TQI.3.pdf)와 [5장 Theorem5.49](https://cs.uwaterloo.ca/~watrous/TQI/TQI.5.pdf)의 관련 부분을 다시 대조해 최소 오류와 선택 측정의 정보량/상계를 구분했다. 검증기는 평균 밀도행렬·양의 고유공간 trace·별도 이분법 root와signed log1p를 사용한다. 원래/자른 구간 관계는 보고된 관측 구간 사이의 정확 유리수 연산으로 확인한다. 생산 계산 재실행을 독립 검증이라고 기록하지 않으며, 원문 전체를 읽거나 재배포한 것으로 기록하지 않는다.

## 상대 엔트로피 독립 응답 검증의 근거

2026-11-09. [Watrous5장](https://cs.uwaterloo.ca/~watrous/TQI/TQI.5.pdf)의 정의5.18(식5.85)과 정리5.35(식5.174)를 대조했다. 정확 support와 같은 채널의 두 상태에 대한 단조성을 확인한 것이며 전체 장의 증명을 읽었다고 기록하지 않는다. 자체2x2 특성방정식·고유공간 trace와 signed log1p 구간으로 예제 응답을 검증한다. 구간 비교는 이 예제의 수치 검사이며 일반 DPI 증명·Petz 복원·범용 채널 계산을 구현한 것으로 기록하지 않는다. 원문 PDF·그림을 저장하거나 재배포하지 않았다.

## 복제 불가·BB84 독립 응답 검증의 근거

2026-11-09. [Richard Jozsa의 Cambridge 공식 노트](https://www.qi.damtp.cam.ac.uk/files/PartIIIQC/Part%20IIC%20QIC/PartIIC%20QIClectures%20Full.pdf)의5장 중 intercept/resend 설명과 정보 조정·privacy amplification의 범위 구분을 확인했다. 노트의 Breidbart 기저 예제를 우리 Z/X 기저 공격의 수치 참조로 쓰거나, 보안 임계값을 이 계산기의 결과로 옮기지 않는다. 자체 기저 전이·공개 관측별 사후분포와 기존 독립 ket 참조25조합을 대조한다. 복소 field·CNOT 후보·겹침은 자체 정확 유도다. 전체 노트를 읽거나 일반 보안을 증명했다고 기록하지 않으며 원문 PDF·그림을 저장·재배포하지 않았다.

## 텔레포테이션·초밀집 부호화 응답 검증

2026-11-09. 앞서 확인한 Bennett 등의 [텔레포테이션 기관 출판 기록](https://research.ibm.com/publications/teleporting-an-unknown-quantum-state-via-dual-classical-and-einstein-podolsky-rosen-channels)과 Bennett–Wiesner의 [초밀집 부호화 기관 출판 기록](https://research.ibm.com/publications/communication-via-one-and-two-particle-operators-on-einstein-podolsky-rosen-states)의 공유 EPR/고전 통신·단일 입자 전송 자원 구분을 재확인했다. 검증기의 site 순서·진폭 전개·Pauli 전역 부호·Kraus 열은 자체 유도하고 독립 조밀 회로 fixture와 대조했다. 출판 초록을 회로의 정확 위상 규약에 대한 직접 수치 증거로 쓰거나 전체 원문을 읽었다고 기록하지 않는다. PDF·그림을 저장·재배포하지 않았다.

## 오류 정정 독립 응답 검증의 근거

2026-11-09. [Preskill 공식7장](https://www.preskill.caltech.edu/ph219/chap7_26_6A.pdf)의7.2–7.3절 중 식7.17과 그 앞뒤의 필요충분성·퇴화 오류 설명을 재확인했다. 코드 image 내적으로 자체 `V†Ea†EbV`를 구하고, 선택한 오류 span의 복구 존재와 고정 최소 무게 X 복구의 실제 결과를 구분한다. IID 채널·정규화 Choi·충실도는 자체 다수결/Bell 유도로 대조한다. 전체92쪽이나 표면 코드·최신 임계값 연구까지 검증했다고 기록하지 않는다. 원문 PDF·그림을 저장하거나 재배포하지 않았다.

## 주요화·얽힘 자원 응답 검증의 근거

2026-11-09. [Nielsen 원 논문 공식 기록](https://arxiv.org/abs/quant-ph/9811053)과 [Vidal 원 논문 공식 기록](https://arxiv.org/abs/quant-ph/9902033)의 pure-state/단일 복사본·국소 연산과 고전 통신 범위를 재확인했다. 앞서 대조한 정리의 방향·최적 꼬리비를 유지하며 자체 hinge 비교·안정 순위·꼬리 제약 포화·Bell 가지를 구현한다. deterministic100쌍은 별도 순열 볼록 껍질 참조다. Vidal 확률은 정리 공식/제약의 독립 산술 검사이며 일반 최적 프로토콜의 합성/실행을 구현했다고 주장하지 않는다. 공식 초록 재확인을 전체 원문 재독으로 기록하지 않으며 PDF·그림을 저장·재배포하지 않았다.

## 채널 용량 독립 응답 검증의 근거

2026-11-09. [Devetak–Shor 공식 기록](https://arxiv.org/abs/quant-ph/0311131)의 generalized dephasing 단일 문자 범위와 [BSST 공식 기록](https://arxiv.org/abs/quant-ph/0106052)의 공유 얽힘·입력/출력/정화 엔트로피 최적화 범위를 재확인했다. 앞서 대조한 Preskill의 탈위상/대칭 논의를 유지하고, 자체 matrix-unit 작용·환경 행렬식·Choi·정보량으로 수치 응답을 검증한다. 선택 입력의 정보를 용량으로 바꾸거나 임의 채널에 가족 정리를 확대하지 않는다. 공식 초록 재확인을 전체 원문 재독으로 기록하지 않으며 PDF·그림을 저장·재배포하지 않았다.

## 엔트로피·압축 교수 화면의 해석 범위

2026-11-09. [Watrous 공식5장](https://cs.uwaterloo.ca/~watrous/TQI/TQI.5.pdf)의 엔트로피/정보원 부호화 구성과 기존 원고의 Schumacher 정리 범위를 참조한다. 장 전체나 정리의 증명을 이번에 재독했다고 기록하지 않는다. 화면은 자체 두 매개변수 상태와 유한 IID projector 무게를 계산한다. 점근 압축률·선택 부분공간의 무게·메모리 차원·실제 codec의 충실도를 구분하며, encoder/decoder나 유한 블록 보장을 구현했다고 주장하지 않는다. 원문 PDF·그림을 저장·재배포하지 않았다.

## 앙상블 교수 화면의 상계와 기록 해석

2026-11-09. [Watrous 공식5장](https://cs.uwaterloo.ca/~watrous/TQI/TQI.5.pdf)의 Scenario5.45, 식5.244–5.248과 Theorem5.49의 관련 부분을 대조했다. 선택 측정의 정보량과 모든 측정의 최적값, Holevo 상계를 구분한다. 최종 추측과 원시 기록의 구별·동일 평균의 두 앙상블·직교 Y의 Z 관측은 자체 예제다. 장 전체나 일반 정리 증명을 새로 검증했다고 기록하지 않는다. 원문 PDF/그림은 저장하거나 재배포하지 않았다.

## 상대 엔트로피 교수 화면의 순서·지지집합·같은 채널

2026-11-09. [Watrous 공식5장](https://cs.uwaterloo.ca/~watrous/TQI/TQI.5.pdf)의 상대 엔트로피 정의와 Theorem5.35(식5.174)의 관련 부분을 대조했다. 두 상태에 같은 채널을 적용하는 단조성과 실제/참조 순서·지지집합 조건을 설명한다. 겹치는 순수 상태·거의 순수 전계수·X/Y 소거·작은 양수의 비교 그림과 예제는 자체 작성했다. 장 전체나 일반 정리의 증명을 새로 검증했다고 기록하지 않으며 Petz 복원이나 범용 채널 솔버를 구현한 것이 아니다. 원문 PDF/그림을 저장하거나 재배포하지 않았다.

## 복제 불가·BB84 교수 화면의 범위

2026-11-09. [Bennett–Brassard 원 논문의 아카이브](https://arxiv.org/abs/2003.06557)의 저자·초록·원1984/아카이브2020 날짜와 비직교 신호·교란의 설명을 재확인했다. 이번 확인을 원문 스캔 전체 재독이나 현대적 보안 증명의 점검으로 기록하지 않는다. CNOT 출력과 목표 tensor·주변 상태·내적 보존 필요조건은 자체 유도하고 기존 독립 검증으로 확인한다. 공식 TQI2장에서는 cloning 항목을 찾지 못했고 Nature 원 논문/IBM 강의 페이지는 접근하지 못했으므로 그 내용을 이번에 확인했다고 주장하지 않는다. f/e/m의80가지 가로채기·재전송 확률과 IID 표본은 자체 모델이며 일반 공격·비밀키율·보안 임계값을 제공하지 않는다. 원문 PDF/그림은 저장하거나 재배포하지 않았다.

## 텔레포테이션·초밀집 교수 화면의 통신/위상 규약

2026-11-09. Bennett 등의 [텔레포테이션 기관 출판 기록](https://research.ibm.com/publications/teleporting-an-unknown-quantum-state-via-dual-classical-and-einstein-podolsky-rosen-channels)과 Bennett–Wiesner의 [초밀집 부호화 기관 출판 기록](https://research.ibm.com/publications/communication-via-one-and-two-particle-operators-on-einstein-podolsky-rosen-states)의 저자·날짜·초록과 공유 EPR/고전 결과·주변 통계/공동 측정의 자원 설명을 다시 확인했다. 기관 초록을 전체 원문 재독이나 이 화면의 위상 규약을 직접 증명하는 자료로 기록하지 않는다. A0/A1/B 순서·네 가지 raw 진폭/보정·Pauli 역순 전체 부호·복소/√2 예제는 자체 회로와 기존 독립 검증으로 대조한다. 미리 공유한 Bell 자원과 실행 중 전송, 조건부 가지와 Bob의 무조건 상태를 분리한다. 원문 PDF·그림은 저장하거나 재배포하지 않았다.

## 오류 정정 교수 화면의 가족 조건과 고정 복구

2026-11-09. Knill–Laflamme [원고의 HTML](https://arxiv.org/html/quant-ph/9604034v1)3.2절 Theorem3.2의 식19/20과 복구 연산자에 독립적인 조건 설명을 대조했다. 코드 기저의 교차 내적이 논리 레이블을 구별하지 않아야 한다는 조건이며 단일 입력의 충실도 검사를 대신 쓰지 않는다. 전체 원고나 일반 정리의 증명을 이번에 전부 확인했다고 기록하지 않는다. 반복 부호의 syndrome·고정 최소 무게 X 복구·별도 IID 논리 채널·퇴화/항등 추가 가족과 비영 행렬 지도는 자체 모델/독립 검증을 사용한다. 임의 통과 가족의 복구 구성이나 내결함성 임계값을 구현한 것이 아니다. 원문 PDF/그림은 저장·재배포하지 않았다.

## 주요화 교수 화면의 확률·누적합·실제 측정

2026-11-09. Nielsen [원고 HTML](https://arxiv.org/html/quant-ph/9811053v2)의 Theorem1/식2와 Vidal [원고 HTML](https://arxiv.org/html/quant-ph/9902033v2)의 식1–5를 대조했다. Schmidt 확률의 주요화 방향과 단일 복사본 순수 상태 최적 성공 확률의 꼬리 제약이 근거다. 전체 원고/일반 프로토콜 구성 증명을 이번에 검증했다고 기록하지 않는다. 교차 쌍·작은 위반·안정 정렬/0채움·명시적 유리 진폭 Bell 측정의 그림/수치는 자체 모델과 독립 검산을 사용한다. HTML 변환 날짜를 원 논문 출판 시점으로 읽지 않으며 과거 미해결 문제를 현재 연구 현황으로 전재하지 않는다. PDF/그림은 저장·재배포하지 않았다.

## 채널 용량 교수 화면의 과제·입력·점근 범위

2026-11-09. [Devetak–Shor 공식 기록](https://arxiv.org/abs/quant-ph/0311131)의 일반화 탈위상 단일 문자 범위와 [BSST 공식 기록](https://arxiv.org/abs/quant-ph/0106052)의 공유 얽힘·입력/출력/정화 엔트로피 최적화 설명을 재확인했다. 이 초록 확인을 전체 원문/일반 부호화 정리의 증명 검증으로 기록하지 않는다. 앞서 대조한 Preskill 노트의 degradability/대칭 조건을 유지하며 큐비트 환경·순수 입력·완전 탈위상·알려진 Z·p와1−p·작은 양수의 그림과 값은 자체 모델/독립 검산으로 확인한다. 선택 입력의 Ic나 한 번의 추측을 용량으로 바꾸지 않으며 비밀·양방향 보조·유한 블록 보장을 구현하지 않는다. 원문 PDF/그림은 저장하거나 재배포하지 않았다.
