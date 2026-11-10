# 안장점·요동·비가우스 근사: 한영 상세 원고

2026-11-10. `saddle-fluctuation-and-nongaussian-approximation`의 학생용 한영 원고를 작성했다. **언어별 28절·24표시수식·4표·6비교 실험 안내·8해설 문제**와 동일한 실행 Python 예제를 포함한다.

- [한국어 원고](../../../content/learning/physics-math-for-ai/saddle-fluctuation-and-nongaussian-approximation.ko.md)
- [영어 원고](../../../content/learning/physics-math-for-ai/saddle-fluctuation-and-nongaussian-approximation.en.md)

Interactive Book 등록 전 초안이다. 생산 계산기·검산/Worker·그림/실습·실제 reader와 범위 감사가 남는다. [20요건 중 5개](saddle-fluctuation-acceptance-matrix.json)는 앞 수학 기반 4개와 이번 원고 1개다. 단원·원주제·전체 목표는 미완료다.

## 본문 구성

깊이가 같은 그릇의 폭을 비교한 뒤 유한 시간 작용, 끝점 절반 potential, 커널 정규화를 선언한다. 일반 양끝의 비선형 정상 방정식과 삼대각 Hessian, 강볼록성, 최대 원리, Newton 후보와 잔차의 차이를 유도한다. 위치·작용·역행렬·determinant를 감싸는 인증 조건도 설명한다.

최소 작용의 지수에 Gaussian 부피를 붙이고 ξ=√ℏy로 재척도화해 첫 비가우스 보정을 구한다. 부분적분에 기반한 모멘트 재귀와 쌍짓기의 9·6·72·24계수를 설명한다. 삼차 모멘트가 0이어도 삼차 항 제곱은 첫 보정에 남는다.

N2의 정상점 0·1/2·2를 손계산으로 비교한다. c₁은 각각 −1/54, −232/19773, 19/6174다. 양의 마지막 계수는 사차 비용으로 전체 커널이 증가한다는 뜻이 아니다. 그 모형 자체의 최소점/Hessian에서 만든 Gaussian 기준에 대한 보정이다. 한 좌표 밀도의 좌우비로 최소점과 평균의 차이도 유도했다.

양끝 0에서는 비음수 Taylor 나머지로 실제 적분비를 감싸지만 일반 양끝에는 그 보증을 옮기지 않는다. 고정 N의 작은 ℏ, 약한 결합, 고정 시간의 N 정련, 점근 차수 증가는 구별한다. 사차 형식 급수의 계수비와 자유 사슬의 첫 보정 닫힌식도 설명한다.

보조 한 좌표 우물에서는 두 최소점의 부피 합, 중앙 극대점의 배제, 평균 0과 최소점의 차이, κ=0의 ℏ¹ᐟ⁴척도를 연결한다. 전체 이중 우물 양자 경로나 터널링을 계산했다고 표시하지 않는다. 마지막에는 수치 구적의 한계와 AI의 Laplace/evidence·사후 불확실성·대칭/다중 모드·측도 Jacobian을 설명한다.

## 실행과 검증

```bash
npx tsx --test scripts/learning-saddle-fluctuation-manuscripts.test.ts
npx -p node@20 node --import tsx --test scripts/learning-saddle-fluctuation-manuscripts.test.ts
npm run typecheck
npm run validate:content
```

새 2검사가 Node.js 20/24에서 통과했다. 실제 `getLearningContent` 렌더러의 엄격한 KaTeX 처리, 한영 동일 표시수식/Python 코드, 28개 고유 목차, 8해설 문제·4표를 확인했다. 영문 한글 혼입과 H1 중복이 없으며 코드/수식/표에 스크롤 속성이 붙는다. 전체 타입 검사와 1834URL 콘텐츠 검증도 통과했다.

Python은 표준 라이브러리 Fraction과 Gaussian 모멘트 재귀만 사용한다. 다음 정확한 값을 출력한다.

| 확인 항목 | 값 |
|---|---|
| 세 정상점의 c₁ | −1/54, −232/19773, 19/6174 |
| 양끝 0의 적분비 경계 | [53/54,17207/17496] |
| 후보 51/100의 잔차 | 97651/2000000 |
| 끝점 potential을 유지한 S* | 17277169/16777216 |
| 결합 Gaussian 역행렬 | R₁₁=44/105, R₁₂=16/105 |

좌우 작용차, 음의 이동 나머지, N3의 정상 방정식도 assert로 확인한다. 일반 정상점 해법이나 전체 비가우스 적분을 구한 코드로 표현하지 않는다.

증거는 `.runtime/learning/saddle-fluctuation-manuscripts/verification.json`과 렌더링/코드 실행 JSON·Node20/24 로그에 있다. 앞 단계의 정확분수 4701검사와 소수 27검사(26모형·18행)는 별도 수학 유도 증거다. 이를 새 프로젝트 2검사와 같은 검사 수로 합산하지 않는다. 실제 그림·모바일/명암·접근성·저장·Worker·생산 자산 검증은 아직 하지 않았다.

## 수량과 다음 순서

등록 599단원·1850안내·등록 한영 1198원고를 유지한다. **미등록 단원 초안 1개·초안 원고 2편**을 별도로 기록한다. 전체 1200Markdown 원고를 등록 수치와 혼동하지 않는다. 입문 315/362검증·47미완료와 전체 51영역509주제·영역 감사·연구 심화도 유지한다.

다음은 일반 양끝의 생산 정상점/잔차 인증, Gaussian 행렬/보정 구간과 N2…32·12자리 입력의 정밀도·전송/시간 비용 검증이다. 이후 독립 전체 응답 검산 → Worker 수명 → 기존 홈페이지 스타일의 한영 그림/실험/저장 → 실제 reader를 연결한다. 기존 홈페이지 출력은 변경하지 않았다. 배포 메모리 수정 ddf5e25의 공개 상태는 해당 GitHub 작업으로 별도 확인한다.
