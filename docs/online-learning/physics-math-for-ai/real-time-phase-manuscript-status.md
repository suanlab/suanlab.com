# 실시간 위상·고전극한: 한영 상세 원고

2026-11-10. `real-time-phase-and-classical-limit`의 학생용 한영 원고를 작성했다. **언어별 33절·22표시수식·2표·6비교 실험·8해설 문제**와 동일한 실행 Python 예제를 포함한다.

- [한국어 원고](../../../content/learning/physics-math-for-ai/real-time-phase-and-classical-limit.ko.md)
- [영어 원고](../../../content/learning/physics-math-for-ai/real-time-phase-and-classical-limit.en.md)

등록 전 초안이다. [23요건 중 8개](real-time-phase-acceptance-matrix.json)는 앞 수학/범위 기반 7개와 이번 원고다. 생산 입력/정밀도/전송 비용, 계산기·독립 검산·Worker, 실제 그림/실험/저장, reader와 범위 감사는 남는다. 본문의 비교 안내와 문제를 실제 인터랙티브 화면 완료로 세지 않는다.

## 설명과 손계산

복소 화살표의 상쇄에서 시작해 비정상 위상의 부분적분, 비퇴화 정상 위상의 조건과 오차 차수를 설명한다. 유한 실시간 작용의 양끝 절반 규칙, 전체 Hessian/forcing, continuant/Green 응답, signature와 자유 단계 정규화를 유도한다. 유니터리성과 연속 시간 정확도는 구별한다.

N2,v3의 최대 정상점과 N3,v2의 영 첫 pivot을 가진 가역 행렬, N2,v2의 호환/비호환 초점을 손계산한다. 점별 커널의 delta distribution과 정규화된 상태를 구별하고 Abel 복소 평균을 Born 공분산으로 해석하지 않는다. 각 pivot sqrt의 곱이 determinant의 주 sqrt와 다른 branch 예도 설명한다.

정규화 chirped Gaussian의 전체 공분산, symplectic 전파와 복소 폭을 연결한다. 고정 폭의 고전 ensemble와 σ²=ℏr의 국소화 족을 구별한다. 자유 χ=−1 예는 최종 위치 분산이 0으로 가도 운동량 분산이 남을 수 있음을 보여준다.

별도 canonical fold에서 두 정상점의 작용/곡률·간섭, 합류에서 Gaussian 근사의 실패와 Airy의 ℏ¹ᐟ³ 척도를 설명한다. 실수 정상점이 없는 쪽의 비영 진폭, 중앙 급수와 큰 인수 scaled/log 표현, 인증된 나머지와 단순 수치 수렴의 차이를 다룬다. 조화 delta 초점과 generic fold를 동일시하지 않는다. AI의 Laplace 근사·동역학·불확실성·sign problem과의 연결도 범위를 명시한다.

## 실행과 검증

```bash
node --import tsx --test scripts/learning-real-time-phase-manuscripts.test.ts
npx -p node@20 node --import tsx --test scripts/learning-real-time-phase-manuscripts.test.ts
npm run typecheck
npm run validate:content
```

새 2검사가 Node.js 20.20.2/24.13.0에서 통과했다. 실제 `getLearningContent`의 strict KaTeX 처리, 고유 목차, 한영 동일 표시수식/Python, 영어 한글 혼입 없음과 코드/수식/표 스크롤 속성을 확인했다. 타입 검사와 1835URL 콘텐츠 검증도 통과했다. 별도 생산 빌드나 브라우저/접근성 검증을 이번 원고 단계의 증거로 주장하지 않는다.

Python은 Fraction만 사용하며 다음을 정확히 검사한다.

| 항목 | 결과 |
|---|---|
| N2,v3 | q*=−1, S*=1/4, ν=1 |
| N3,v2 | detH=−1, H⁻¹=H |
| N2,v2 | 호환 조건 a+b=0 |
| 고정 σ²=2/3,χ=−1,ℏ=1/2 | 최종 위치 분산 3/32, 운동량 분산 73/96 |
| 동일 조건 ℏ=1/8 | 위치 분산 3/512, 운동량 분산 1033/1536 |
| σ²=ℏr,χ=0,r=2/3 | 최종 위치 분산/ℏ=25/24, 운동량 분산/ℏ=3/8 |

증거는 `.runtime/learning/real-time-phase-manuscripts/verification.json`, 언어별 렌더링/본문 Python 실행 JSON에 있다. 앞 41,191정확분수 검사와 13소수 진단은 별도 수학 기반이며 새 2프로젝트 검사와 합산하지 않는다.

## 수량과 다음 순서

등록 600단원·1857안내·등록 한영 1200원고를 유지한다. **미등록 초안 1단원·원고 2편**, 전체 Markdown 원고 1202편을 별도로 기록한다. 입문 315/362검증·47미완료와 전체 51영역509주제·영역 감사·연구 심화는 계속 남는다.

기존 홈페이지 반영 commit `1e11ee1`의 GitHub Pages 배포 성공을 확인했다. 이번 원고는 reader에 아직 등록하지 않았다. Online Book 메뉴에 신규 콘텐츠 링크를 추가하지 않았으며 기존 Interactive Book 진입을 유지한다.

다음은 세 mode(chain/packet/fold)의 소유 입력과 전체 N2…32·12자리 분수 한도를 유지하는 생산 계약이다. 정확 초점/near-caustic, 큰 작용/ℏ의 위상 reduction, Airy 상수/중앙·점근 구간과 scaled/log 표현, 손실 없는 전체 행렬 전송의 정밀도·실제 비용을 확정한 뒤 독립 기준과 계산기를 구현한다.
