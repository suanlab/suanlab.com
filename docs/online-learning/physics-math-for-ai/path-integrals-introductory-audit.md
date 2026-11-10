# 경로적분 네 단원: 입문 범위 종합 검토

2026-11-11. [최초 범위](path-integrals-primary-source-audit.md)의 Brownian bridge·Euclidean 커널·안장점·실시간 네 처리를 원고, 계산, 독립 검산, 실제 화면과 대조했다. **24개 입문 요구를 충족**한다고 판정해 입문 **316/362검증·46미완료**로 진행한다. 전체 51영역 509주제, 영역 감사와 연구 심화는 아직 완료하지 않았다.

## 원래 요구와 네 처리

[24요건 행렬](path-integrals-introductory-audit-matrix.json)에 한영 원고의 절 번호와 판정 이유를 기록한다. 파일 존재나 완료 표시만으로 판정하지 않는다.

| 처리 | 대조한 핵심 | 해석의 경계 |
|---|---|---|
| Brownian bridge, 20절 | 증분 정규화·조건화·전체 공분산·Schur 소거·정련·면적 관측·거칠기·보조 원천 | 매끈한 평균은 전형적인 경로가 아니다. 유한 일치만으로 연속 측도의 존재를 증명하지 않는다. |
| Euclidean 커널, 24절 | Hamiltonian·대칭 potential 규칙·determinant·Mehler 비교·우측 규칙 오차·trace·τ=βℏ | 양의 커널, 조건부 법칙, 열 밀도, Born 확률을 구별한다. 임의 potential의 수렴을 보증하지 않는다. |
| 안장점, 28절 | 강볼록 정상 경로·잔차 인증·Gaussian 체적·삼차/사차 보정·조건부 적분비 경계·여러 최소점·퇴화 | mode와 국소 폭은 실제 평균·공분산과 다르다. 계수 인증은 전체 오차 보증이 아니며 터널링을 계산하지 않는다. |
| 실시간, 34절 | 간섭·음의 모드·영 pivot·Abel branch·delta 초점·packet·두 고전극한·fold/Airy | 복소 응답은 확률 공분산이 아니다. 일반 Wick 회전·비선형 실시간 측도·sign problem을 해결하지 않는다. |

실시간 단원의 23요건은 [별도 단원 감사](real-time-phase-scope-audit.md)를 통과했다. 원주제의 입문 판정은 네 처리의 개별 계약과 상호 구별을 함께 검토한 결과다.

## 보완한 비교와 손계산

마지막 단원 33절에 네 대상의 측도·정규화·해석을 비교하는 표와 네 손계산을 추가했다. 자유입자의 D=ℏ/(2m)를 맞추면 Brownian·Euclidean·λ=0 안장점의 양의 법칙이 같지만, 실시간 연결 응답은 iℏH⁻¹이다. N4의 중간 분산 1/4와 정련 때 달라지는 면적 분산 5/64→21/256, 기대 작용 2와 최소 작용 1/2를 구별했다.

N2에서는 Q_E=9/2·μ_E=4/9·C_E=2/9와 H_R=7/2·q*=4/7·형식 응답 2i/7을 같은 매개변수로 비교했다. λ=1의 정상점 1/2·국소 분산 8/39와 λ=0의 평균 37/72도 대조해 비가우스 실제 모멘트로 오해하지 않게 했다. 같은 정규화 크기나 비슷한 그림이 같은 측도를 뜻하지 않는다.

독립 Python Fraction 프로그램은 시간 edge에서 밀집 정밀도를 만들고 Gauss–Jordan으로 inverse·mean·determinant를 계산한다. N2/3/7/32, 두 시간, 두 질량/ℏ 조합의 **16모형·공분산 4712항목**을 대조했다. 추가 검사는 정련 관측·Hessian 부호·사차 국소 폭·초점 packet을 확인한다. 새 종합 6검사와 원고/통합 검사를 합친 20검사가 Node20/24에서 통과했다.

이미 구현된 Euclidean 계산기를 계획으로 표현하던 문장을 한영에서 고쳤다. 실시간 원고의 미연결 안내도 현재 계산과 진단의 구별로 바꿨다. 기존 1200원고 중 1198개는 byte가 같고, Euclidean 두 원고는 명시한 문구만 역교정했을 때 원래 전체 hash와 같아야 통과한다. 역사적 fixture를 새 hash로 덮어쓰지 않았다.

## 실제 화면과 수치 근거

수정 원고의 Node20/Next14 빌드와 1839HTML의 링크·자산·목차·sitemap·canonical을 검사했다. 네 reader의 한영×명암×320/1440px **32조합**에서 312개념·256문제의 정오답·200안내·접근성 32회를 확인했다. 영어 한글 혼입과 문서 가로 넘침이 없다. 비교표의 첫 열을 보정한 후 해당 8상태의 키보드 초점·단원 링크·접근성과 단어 줄바꿈을 다시 확인했다. 본문 링크로 bridge를 열고 돌아오는 흐름도 확인했다.

기존 생산 자산의 63·71·54·38요청과 변조/귀속 거부는 이번 실행 횟수에 더하지 않는다. 현재 계산/검산 의존성 19·19·21·37개가 각 검증 후보와 byte가 같고, 후보 자산도 당시 hash와 일치함을 대조했다. 이번에는 별도 폴더에서 원래 실시간 62모형의 41191정확검사와 13Decimal진단도 재실행했다. Decimal 진단이나 Node/브라우저 일치를 독립 적분 오차 증명으로 부르지 않는다.

등록 601단원·1863안내·1202원고 수는 유지한다. 보완한 출력은 이전 홈페이지를 백업하고 정적 자산을 보존한 뒤 반영한다. 공개 배포 성공은 commit/push와 구별해 GitHub Pages 결과로 확인한다.

## 재현과 다음 순서

```bash
node --import tsx --test scripts/learning-path-integral-synthesis.test.ts scripts/learning-real-time-phase-manuscripts.test.ts scripts/learning-real-time-phase-reader-integration.test.ts scripts/learning-saddle-fluctuation-reader-integration.test.ts
node --import tsx scripts/audit-path-integral-production-lineage.ts
PATH_INTEGRAL_PREVIEW_BASE=http://127.0.0.1:4206 node --import tsx scripts/check-path-integral-topic-readers.ts
PATH_INTEGRAL_PREVIEW_BASE=http://127.0.0.1:4206 node --import tsx scripts/check-path-integral-comparison-style.ts
```

다음 입문 원주제는 **statistical field theory**의 일차 자료·범위 검토다. 기존 상관함수와 장의 기초를 재등록해 완료 수를 부풀리지 않고, 요구와 새로 필요한 내용을 먼저 대조한다. 상호작용 연속체·fermion/gauge 측도·굽은 공간의 순서·재규격화·일반 실시간 측도 등 연구 확장과 전체 교과 감사는 계속 남는다. 근거는 `.runtime/learning/path-integral-topic-audit/verification.json`에 보존한다.
