# 실시간 위상·고전극한: 일차 자료와 원래 범위

2026-11-10. 예정 ID는 `real-time-phase-and-classical-limit`이다. 경로적분의 네 처리 중 마지막이며, 앞 단원의 양의 Euclidean 가중치에서 복소 실시간 위상으로 넘어간다. **자료/유한 수학 기반만 준비했으며 새 단원은 등록하지 않았다.**

## 직접 읽은 자료

- [Andreas Wipf, Path Integrals, Chapter3 §3.1](https://www.tpi.uni-jena.de/~wipf/lectures/pfad/pfad3.pdf#page=1), 인쇄18–22쪽/PDF1–5쪽: 유한 이차 작용·삼중대각 Hessian·determinant 점화식·고전 작용과 커널·초점에서 파동묶음에 대한 작용을 확인했다. 본 교재의 대칭 양끝 규칙은 직접 다시 유도한다. 해당 자료의 왼쪽 endpoint 규칙을 그대로 대입하지 않는다.
- [P. A. Horváthy, The Maslov Correction in the Semiclassical Feynman Integral, §§2–3](https://arxiv.org/pdf/quant-ph/0702236#page=2), PDF2–5쪽: 부호 있는 Fresnel 적분·음의 요동 방향과 위상·초점에서의 parity/delta kernel을 확인했다. PDF4–5쪽은 이미지로도 읽었다. 텍스트 추출은 식(3.4)의 작용 분모에 불필요한 절댓값을 넣거나 식(3.6)의 각 곱 인자의 절댓값을 빠뜨릴 수 있다. 이미지에서는 작용의 sin은 부호를 유지하고, 곱에는 절댓값이 있다. 저자의 오기로 판정하지 않는다.
- [Evans–Zworski, Lectures on Semiclassical Analysis, Lemma3.13/Theorem3.15](https://math.mit.edu/~vwg/semiclassicalEvansZworski.pdf#page=38), PDF38–41쪽: 매끄러운 compact amplitude에서 비정상 위상의 부분적분과 비퇴화 정상 위상의 차수/위상·조건을 확인했다. 고정된 위상에서의 점근식이며 합류하는 정상점 전체에 균일한 보증으로 확대하지 않는다.
- NIST DLMF의 [§9.2](https://dlmf.nist.gov/9.2), [§9.5](https://dlmf.nist.gov/9.5), [§9.7](https://dlmf.nist.gov/9.7): Airy 방정식/초기값·실수 및 복소 적분·실수축 점근식과 생략항 경계 조건을 확인했다. 중앙/큰 인수의 다른 계산 경로와 상쇄/underflow 문제를 구별한다.
- [DLMF9.6.2](https://dlmf.nist.gov/9.6#E2)와 [10.32.9](https://dlmf.nist.gov/10.32#E9)의 연결을 확인했다. 양의 실수 인수에서 K₁ᐟ₃의 양의 적분 표현은 Ai의 양성을 보여준다. 따라서 실수 정상점이 없는 fold에서도 진폭은 일반적으로 정확한0이 아니다.

Tong의 AQM 페이지/PDF는 열기가 시간 초과였다. 검색 결과나 목차만 본 자료를 읽은 근거로 포함하지 않는다. 다운로드한 PDF와 확인 이미지의 hash/경로는 `.runtime/learning/real-time-phase-foundation/`에 남긴다.

## 학생이 구별해야 할 내용

1. `|exp(iS/ℏ)|=1`인 위상과 양의 확률밀도. 전파 kernel과 파동함수·Born 밀도도 구별한다.
2. 정상 작용과 최소 작용. Hessian의 음의 방향을 제거하거나 determinant의 절댓값만 남기면 위상을 잃는다.
3. 개별 LDL pivot0과 전체 Hessian의 특이성. 첫 pivot0인 가역 모형도 지원해야 한다.
4. 초점의 비유일/불가능한 양끝 문제, distribution kernel, 유한하고 정규화된 파동묶음의 전파.
5. 유한 양의 Abel 조절자와 그 제거. 조절한 복소 이차 응답은 실제 위치 공분산이 아니며 유한 조절자는 유니터리 전파가 아니다.
6. 고정한 초기 폭의 ℏ→0과, 폭 제곱이ℏ에 비례하는 국소화 족. 전자는 일반적으로 한 점의 고전 궤적으로 붕괴하지 않는다.
7. 고정N의 작은ℏ, 고정시간의 N정련, 조절자 제거와 특수점 접근의 서로 다른 극한.
8. 두 정상점의 간섭·fold 합류·복소 정상점과 지수적으로 작은 진폭. 조화계의 정확한 delta 초점과 Airy fold를 같은 특이성으로 부르지 않는다.

## 계산할 모형

주 모형은 양끝을 고정한 유한 대칭 시간 분할의 실시간 이차 사슬이다. 진동수 제곱은 부호를 허용해 정상 진동자·자유·역진동자를 비교한다. 전체 Hessian/역응답, signature/Maslov index, 고전 symplectic map, 정확한 특이점 분류, 양의 조절자를 유지한다.

동일한 유한 map으로 일반 chirped Gaussian 파동묶음의 평균·전체2×2공분산·위상과 두 고전극한 족을 다룬다. 별도1좌표 canonical fold 적분은 Airy 함수와 두 실수 정상점의 합을 비교한다. 이는 일반 비선형 Hamiltonian의 전체 경로 적분이나 임의 caustic의 완전한 uniform 근사를 계산하는 모형이 아니다.

범위를 줄이지 않고 [수학 계약](real-time-phase-model-contract.md), 독립 유도 검산, 상세 한영 원고, 생산 계산/독립 검산/Worker, 기존 스타일의 개념도·실험·문제, 실제 reader와 범위 감사를 순서대로 완성한다. 현재 등록600단원·1857안내·1200원고와 입문315/362검증·47미완료는 유지한다. 원주제 감사·전체51영역509주제·영역 감사·연구 심화는 별도로 남는다.
