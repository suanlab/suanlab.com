# 통계장론 첫 단원: 수학 기반과 한영 초안

2026-11-11. `finite-field-measure-and-thermal-free-energy`는 **등록 전 준비 단계**다. [일차 자료 검토](statistical-field-theory-primary-source-audit.md)와 [모형 계약](thermal-field-model-contract.md)에 따라 기존 상관함수를 반복하는 대신 절대 측도·자유에너지·온도 경로를 추가했다.

후속 단계에서 [생산 응답·독립 구간·실제 Worker 검증](thermal-field-producer-verification.md)을 진행했다. 아래는 첫 수학 기반 단계의 기록이며, 최신 다음 순서는 개념 그림·조작 시각화·안내 실험·해설 문제다. 실제 reader와 Next 자산 검증은 남아 있다.

## 작성한 내용

- [한국어 초안](drafts/finite-field-measure-and-thermal-free-energy.ko.md), [영어 초안](drafts/finite-field-measure-and-thermal-free-energy.en.md): 각각 21절, display 수식 9개, 비교표 1개, 해설 문제 8개, 동일한 실행 실습을 포함한다.
- `src/lib/learning-thermal-field.ts`: 1/2차원 유한 Gaussian 장의 정밀도·제약·평균·공분산·제한 determinant와 절대 log Z의 정확 기호 피연산자, 유효 에너지 평균과 열역학 U·Cv를 계산한다.
- `scripts/learning/verify-thermal-field-foundation.py`: 별도 Fraction/Taylor 대수, incidence 구성, Gauss–Jordan을 사용해 절대 log Z를 직접 미분한다. 구현의 LDL·Gaussian 열 관측량 모멘트 식을 가져오지 않는다.

핵심 구별은 `U=〈E−T E_T〉`, `Cv=Var(E−T E_T)/T²−T〈E_TT〉`다. 온도 의존 모형족에서 고정 Hamiltonian 공식을 그대로 쓰는 반례를 포함했다. 열역학 엔트로피와 보이는 장 분포의 Gibbs 엔트로피도 구별한다. 기준 측도와 배경의 변화가 정규화 법칙·절대 자유에너지에 미치는 영향을 따로 설명한다.

## 확인 결과

52모형 중 정규화 가능 49개, 정규화 불가능 3개를 대조했다. 정규화 가능한 모형의 Laplacian·정밀도·공분산·공간 역행렬 총 **9276항목**, 평균·determinant·최소 에너지·에너지 평균·U·Cv·log Z 피연산자가 독립 결과와 정확히 일치했다. 최대 32점 2차원 격자, 세 경계, 평균 영 제약, 온도 의존 원천과 배경을 포함한다.

별도 검사는 기준 측도 변경과 좌표 재척도의 차이, Gram 인자, 영모드, 가역 부정부호 행렬의 영 pivot, 두 점 주기 축의 평행 edge, 기존 Gaussian 단원과의 18공통 모형을 확인했다. 공액 Fourier determinant의 부동소수점 비교와 NumPy 실습은 수치 진단이며 정확 유리수 검산과 구별한다.

수학 9검사와 원고 2검사가 Node24·Node20에서 모두 통과했다. 한영 원고는 실제 learning Markdown/KaTeX 파이프라인을 임시 경로에서 사용해 확인했다. 수식 오류와 영어 한글 혼입이 없고, 한영 수식·Python 코드가 일치한다. NumPy 예제를 실제 실행하고 세 온도 경로의 결과와 자유도 배율을 확인했다. ESLint와 TypeScript 검사도 통과했다.

재현:

```bash
node --import tsx --test scripts/learning-thermal-field.test.ts scripts/learning-thermal-field-manuscripts.test.ts
npm run typecheck
npm run lint
```

실행 근거는 `.runtime/learning/thermal-field-foundation/`의 `verification.json`, `independent-fraction-references.json`, `manuscripts.json`, `python-exercise.json`에 있다. 원고 검사는 임시 폴더를 정리하며 기존 `content/learning/` 파일을 추가·수정하지 않는다.

## 남은 순서와 수량

다음은 생산 입력·기호/숫자 표시·전체 응답의 독립 검증, Worker 수명, 기존 홈페이지 스타일의 개념 그림·조작 시각화·안내 실험·해설 문제, 실제 한영 reader의 접근성·반응형 점검이다. 그 뒤 단원을 등록하고 개별 계약을 감사한다. 통계장론 원주제에는 국소 질서변수·상호작용 scalar 장·Legendre 유효 작용의 후속 처리도 필요하다.

등록 **601단원·1202원고**, 검증된 입문 원주제 **316/362**, 미완료 **46개**는 그대로다. 새 한영 초안은 **등록 전 1단원**으로 따로 기록한다. 준비 core는 아직 홈페이지에서 실행되지 않으며 이 단계 때문에 새 페이지나 완료 항목을 만들지 않는다. 기존 홈페이지·등록 원고·메뉴를 바꾸지 않았다. 공개 배포 여부는 별도로 GitHub Pages 결과로 확인한다.
