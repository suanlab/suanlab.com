import type { PromptSnippet } from './index';

// 즉시 복사 가능한 단문 프롬프트 라이브러리. 본문(content)은 한국어 기본.
export const promptSnippets: PromptSnippet[] = [
  // ── 코딩 / 구현 ──
  {
    id: 'snip-code-review',
    title: { ko: '코드 리뷰 요청', en: 'Request a code review' },
    description: { ko: '버그·보안·성능·가독성을 종합 점검하는 리뷰를 요청합니다.', en: 'Ask for a full review: bugs, security, performance, readability.' },
    category: 'coding',
    tags: ['리뷰', '품질'],
    content: `아래 코드를 리뷰해 주세요. 각 항목별로 평가해 주세요:
1. 잠재적 버그 / 논리 오류
2. 보안 취약점 (입력 검증, 민감정보)
3. 성능 개선점 (시간/공간 복잡도)
4. 가독성·네이밍·구조
5. 엣지 케이스 처리

각 이슈에 심각도(🔴치명/🟡주의/🟢제안)와 구체적 수정 코드를 함께 제시해 주세요.

\`\`\`
[여기에 코드 붙여넣기]
\`\`\``,
  },
  {
    id: 'snip-debug',
    title: { ko: '버그 원인 분석', en: 'Debug / root-cause analysis' },
    description: { ko: '에러 증상을 바탕으로 원인 가설과 최소 재현 케이스를 도출합니다.', en: 'Propose root-cause hypotheses and a minimal repro.' },
    category: 'coding',
    tags: ['디버깅'],
    content: `아래 에러/증상의 원인을 분석해 주세요.

증상:
[에러 메시지 또는 잘못된 동작 설명]

관련 코드:
\`\`\`
[코드]
\`\`\`

다음을 제시해 주세요:
1. 가능한 원인 가설 3가지 (가장 유력한 순)
2. 각 가설을 확인하는 최소 재현/점검 방법
3. 근본 원인(root cause)을 고치는 수정 코드
4. 재발 방지를 위한 테스트 케이스`,
  },
  {
    id: 'snip-refactor',
    title: { ko: '리팩토링 제안', en: 'Refactor for clarity' },
    description: { ko: '동작은 유지하면서 구조를 개선하는 리팩토링을 제안합니다.', en: 'Refactor while preserving behavior.' },
    category: 'coding',
    tags: ['리팩토링'],
    content: `아래 코드를 동작을 바꾸지 않고 리팩토링해 주세요. 목표:
- 함수 분리 (단일 책임)
- 중복 제거
- 네이밍 개선
- 복잡도 감소

Before/After를 나란히 보여주고, 각 변경의 이유를 한 줄로 설명해 주세요.

\`\`\`
[코드]
\`\`\``,
  },
  {
    id: 'snip-test-gen',
    title: { ko: '테스트 코드 생성', en: 'Generate test cases' },
    description: { ko: '정상/경계/예외 케이스를 아우르는 단위 테스트를 생성합니다.', en: 'Generate unit tests covering happy/edge/error paths.' },
    category: 'coding',
    tags: ['테스트'],
    content: `아래 함수/모듈에 대한 단위 테스트를 작성해 주세요.
- 프레임워크: pytest (또는 jest)
- 정상 케이스, 경계값(0/빈/최대), 예외 케이스를 모두 포함
- 각 테스트 이름은 의도를 드러내게 (test_xxx_상황_기대결과)
- 목(mock)/픽스처가 필요하면 함께 작성

\`\`\`
[코드]
\`\`\``,
  },
  {
    id: 'snip-explain-code',
    title: { ko: '코드 설명', en: 'Explain code' },
    description: { ko: '복잡한 코드를 단계별로 쉽게 설명합니다.', en: 'Walk through complex code step by step.' },
    category: 'coding',
    tags: ['설명'],
    content: `아래 코드를 처음 보는 사람도 이해할 수 있게 설명해 주세요.
1. 전체 목적 한 문장
2. 줄별/블록별 핵심 동작
3. 어려운 부분은 비유나 그림(텍스트)으로 보조 설명
4. 개선할 점이 있다면 제안

\`\`\`
[코드]
\`\`\``,
  },
  {
    id: 'snip-complexity',
    title: { ko: '복잡도 분석', en: 'Time/space complexity analysis' },
    description: { ko: '시간·공간 복잡도를 분석하고 최적화 방향을 제시합니다.', en: 'Analyze big-O and suggest optimizations.' },
    category: 'coding',
    tags: ['복잡도', '성능'],
    content: `아래 코드의 시간·공간 복잡도를 빅오(Big-O)로 분석해 주세요.
- 최악/평균 케이스 구분
- 핵심 연산 단계별 근거
- 병목 구간과 개선 가능한 최적화 전략(더 나은 자료구조/알고리즘)

\`\`\`
[코드]
\`\`\``,
  },
  {
    id: 'snip-from-scratch',
    title: { ko: '처음부터 구현', en: 'Implement from scratch' },
    description: { ko: '라이브러리 없이 알고리즘/자료구조를 직접 구현합니다.', en: 'Implement an algorithm/structure without libraries.' },
    category: 'coding',
    tags: ['구현', '학습'],
    content: `Python으로 다음을 처음부터 구현해 주세요 (외부 라이브러리 금지, NumPy만 허용 표시):
[구현할 것: 예) 이진 탐색 트리 / 트랜스포머 attention]

요구사항:
1. 동작하는 완전한 코드 + 타입 힌트
2. 각 단계 주석으로 원리 설명
3. 시간/공간 복잡도 명시
4. 사용 예제와 간단한 테스트`,
  },

  // ── 학습 / 분석 ──
  {
    id: 'snip-paper-summary',
    title: { ko: '논문 3줄 요약', en: 'Paper in 3 lines' },
    description: { ko: '논문 PDF를 읽고 핵심을 3문장으로 압축합니다.', en: 'Read a paper PDF and summarize in 3 sentences.' },
    category: 'learning',
    tags: ['논문', '요약'],
    content: `아래 논문 PDF를 읽고 다음을 작성해 주세요:
1. **3문장 요약**: 문제 / 제안 / 결과
2. **핵심 기여** 2~3개 (불릿)
3. **메인 결과**: 대표 수치 1~2개와 베이스라인 대비 개선
4. **한계점** 1~2개
5. **내 연구에 쓸 점**: 이 논문에서 차용할 아이디어/기법

PDF: [파일 경로 또는 파일명]`,
  },
  {
    id: 'snip-compare',
    title: { ko: '개념 비교표', en: 'Comparison table' },
    description: { ko: '여러 방법론/모델을 비교표로 정리합니다.', en: 'Build a comparison table of methods/models.' },
    category: 'learning',
    tags: ['비교', '정리'],
    content: `다음 항목들을 비교표(마크다운 표)로 정리해 주세요:
[비교 대상: 예) Transformer / Mamba / RWKV]

비교 기준:
- 핵심 아이디어
- 구조 (길이 제한, 병렬성)
- 계산 복잡도
- 강점 / 약점
- 대표 적용 분야
- 공개 구현 여부

표 뒤에 "어떤 상황에서 무엇을 선택해야 하는가"를 1문단으로 정리해 주세요.`,
  },
  {
    id: 'snip-quiz',
    title: { ko: '자가 점검 퀴즈', en: 'Self-check quiz' },
    description: { ko: '주제에 대한 이해도 점검 퀴즈를 만듭니다.', en: 'Generate a comprehension quiz.' },
    category: 'learning',
    tags: ['퀴즈', '학습'],
    content: `주제 "[주제]"에 대한 자가 점검 퀴즈를 만들어 주세요.
- 객관식 5문제 (4지선다, 오답 유혹항 포함)
- 단답형 3문제
- 서술/응용 2문제

정답과 해설은 퀴즈 뒤에 별도로 제시해 주세요. 난이도는 기초→응용으로 점진 상승.`,
  },
  {
    id: 'snip-related-work',
    title: { ko: '관련 연구 정리', en: 'Related work brief' },
    description: { ko: '주제의 핵심 관련 연구를 그룹화해 정리합니다.', en: 'Group and summarize related work.' },
    category: 'learning',
    tags: ['관련연구', '문헌'],
    content: `주제 "[주제]"의 관련 연구를 다음 구조로 정리해 주세요.
1. 연구 흐름을 3~4개 그룹으로 분류 (예: 접근법별 / 시기별)
2. 각 그룹의 대표 논문 3~5편: 제목(저자, 연도) + 핵심 기여 1문장
3. 각 그룹의 한계
4. 우리 연구와의 차별점 (입력: [우리 접근 요약])

허위 인용 금지. 불확실하면 [verification needed] 표시.`,
  },
  {
    id: 'snip-math',
    title: { ko: '수식 단계별 풀이', en: 'Step-by-step math derivation' },
    description: { ko: '수식/증명을 단계별로 풀어 설명합니다.', en: 'Derive an equation step by step.' },
    category: 'learning',
    tags: ['수식', '유도'],
    content: `아래 수식/증명을 단계별로 풀어 설명해 주세요. 각 단계마다 어떤 규칙/정리를 썼는지 명시.

[수식 또는 증명 명제]

출력:
1. 직관 (이 수식이 말하는 것)
2. 각 항/기호의 의미
3. 단계별 유도 (한 단계 = 한 줄, 근거 표기)
4. 최종 결과의 의미와 응용`,
  },

  // ── 데이터 분석 ──
  {
    id: 'snip-eda',
    title: { ko: 'EDA 계획 수립', en: 'EDA plan' },
    description: { ko: '데이터셋 탐색적 분석(EDA) 계획을 세웁니다.', en: 'Plan an exploratory data analysis.' },
    category: 'learning',
    tags: ['EDA', '데이터'],
    content: `아래 데이터셋에 대한 EDA 계획을 세워 주세요.
데이터셋 설명: [컬럼/목적/형태]

다음을 포함:
1. 데이터 품질 점검 (결측/이상치/중복/타입)
2. 변수별 요약 통계 + 시각화 종류 (히스토그램/박스플롯 등)
3. 변수 간 관계 (상관/교차표/산점도)
4. 타겟 변수(있으면)와의 관계
5. 가설 3가지와 확인 방법
6. Python(pandas/ matplotlib) 코드 스켈레톤`,
  },
  {
    id: 'snip-viz',
    title: { ko: '시각화 추천', en: 'Visualization recommendation' },
    description: { ko: '데이터/목적에 맞는 시각화를 추천합니다.', en: 'Recommend charts for the data/goal.' },
    category: 'learning',
    tags: ['시각화'],
    content: `목적: [보여주고 싶은 메시지]
데이터: [변수와 타입]

다음을 제시해 주세요:
1. 가장 적합한 시각화 3가지와 선택 이유
2. 각각의 Python(matplotlib/seaborn) 코드 템플릿
3. 색상/스케일/레이블 등 가독성 개선 팁
4. 잘못되기 쉬운 함정(예: y축 0 시작, 3D 차트 남용) 경고`,
  },
  {
    id: 'snip-stat-test',
    title: { ko: '통계 검정 선택', en: 'Choose a statistical test' },
    description: { ko: '데이터와 가설에 맞는 통계 검정을 추천합니다.', en: 'Recommend the right statistical test.' },
    category: 'learning',
    tags: ['통계'],
    content: `상황: [비교 대상, 데이터 형태, 표본 수, 독립/대응 여부]
가설: [귀무/대립 가설]

다음을 결정해 주세요:
1. 적합한 통계 검정 (t-test / Mann-Whitney / ANOVA / 카이제곱 / ... )과 이유
2. 전제조건(정규성/등분산성) 점검 방법
3. Python(scipy/statsmodels) 코드
4. 결과 해석 가이드 (p-value, 효과량, 신뢰구간)`,
  },
  {
    id: 'snip-data-quality',
    title: { ko: '데이터 품질 점검', en: 'Data quality audit' },
    description: { ko: '데이터셋의 품질 이슈를 체계적으로 점검합니다.', en: 'Audit a dataset for quality issues.' },
    category: 'learning',
    tags: ['데이터', '품질'],
    content: `아래 데이터셋(또는 스키마)의 품질을 점검하는 체크리스트와 코드를 작성해 주세요.

[데이터셋 설명 또는 스키마]

점검 항목:
1. 결측치 패턴 (MCAR/MAR/MNAR 추정)
2. 이상치 (IQR/Z-score, 도메인 기반)
3. 중복 / 불일치
4. 분포 편향 / 클래스 불균형
5. 데이터 누수 위험 (타겟 정보 포함 여부)
6. pandas 코드로 각 항목 점검 + 수정 제안`,
  },

  // ── 작성 ──
  {
    id: 'snip-abstract',
    title: { ko: '초록(Abstract) 작성', en: 'Write an abstract' },
    description: { ko: 'problem→gap→solution→result 구조로 초록을 씁니다.', en: 'Draft an abstract with problem-gap-solution-result.' },
    category: 'writing',
    tags: ['초록', '논문'],
    content: `아래 정보로 학회 논문 초록을 작성해 주세요 (150~250단어, 구조: problem → gap → solution → key result → impact).

- 주제/문제:
- 기존 한계(gap):
- 제안 방법:
- 핵심 결과(수치):
- 의의:

조건:
- 첫 문장은 hook
- 구체적 수치 1~2개 포함
- 과장/모호 표현 금지
- 영문 초록도 함께 제공 옵션`,
  },
  {
    id: 'snip-titles',
    title: { ko: '논문 제목 후보', en: 'Title candidates' },
    description: { ko: '논문 제목 후보를 여러 스타일로 제안합니다.', en: 'Propose title candidates in varied styles.' },
    category: 'writing',
    tags: ['제목'],
    content: `아래 내용으로 논문 제목 후보를 각각 3개씩 제안해 주세요:
- 연구 핵심:

스타일:
1. 직관/설명형 (("~: A method for ..."))
2. 질문/도발형
3. 간결 임팩트형 (3~6단어)

각 후보에 대해: 검색 노출(SEO) 관점 키워드, 학회 분위기 적합도 평가. 가장 추천하는 1개와 그 이유.`,
  },
  {
    id: 'snip-email',
    title: { ko: '학술 이메일 작성', en: 'Academic email' },
    description: { ko: '교수님/협력자/에디터에게 보낼 정중한 이메일을 씁니다.', en: 'Write a polite academic email.' },
    category: 'writing',
    tags: ['이메일', '커뮤니케이션'],
    content: `아래 상황의 이메일을 작성해 주세요 (정중하고 간결하게).

수신자: [예) 지도교수 / 협력 연구실 / 저널 에디터]
목적: [예) 추천서 요청 / 논문 피드백 요청 / revision 제출]
핵심 내용: [3~4문장]

조건:
- 적절한 인사말와 맺음말
- 본문은 한 화면(짧게)
- 명확한 요청/기한/다음 단계
- 영문 버전도 함께 (필요시)`,
  },
  {
    id: 'snip-cover-letter',
    title: { ko: '커버레터 작성', en: 'Cover letter' },
    description: { ko: '학회/저널 투고용 커버레터를 작성합니다.', en: 'Draft a submission cover letter.' },
    category: 'writing',
    tags: ['커버레터', '투고'],
    content: `아래 정보로 학회/저널 투고용 커버레터를 작성해 주세요.

- 투고처:
- 논문 제목:
- 핵심 기여(2~3문장):
- 왜 이 투고처가 적합한지:
- (선택) suggested reviewers / excluded reviewers:

구조: 인사 → 논문 소개 → 기여 요약 → 적합성 → 마무리. 1페이지 이내, 설득력 있고 간결하게.`,
  },

  // ── 프레젠테이션 ──
  {
    id: 'snip-slides',
    title: { ko: '슬라이드 개요', en: 'Slide outline' },
    description: { ko: '발표 슬라이드 개요(목차)를 빠르게 잡습니다.', en: 'Quickly draft a slide outline.' },
    category: 'writing',
    tags: ['발표', '슬라이드'],
    content: `아래 발표의 슬라이드 개요를 잡아 주세요.
- 주제:
- 청중:
- 시간:

각 슬라이드: (번호 / 제목 / 핵심 메시지 1문장 / 시각 자료 1줄)을 표로.
전체 스토리: hook → 문제 → 우리 접근 → 결과 → 의의. 분량은 시간에 맞게.`,
  },
  {
    id: 'snip-qa',
    title: { ko: '예상 Q&A 대비', en: 'Anticipate Q&A' },
    description: { ko: '발표 후 받을 질문을 예상하고 답변을 준비합니다.', en: 'Predict questions and prepare answers.' },
    category: 'writing',
    tags: ['Q&A', '발표'],
    content: `아래 발표 내용에 대해 예상되는 질문 8개를 만들고, 각각 2~3문장 답변을 준비해 주세요.

발표 요지: [핵심 내용]

질문 카테고리를 섞어 주세요:
- 기술적 디테일 (수식/구현)
- 베이스라인/비교 타당성
- 일반화/확장성
- 한계/실패 케이스
- 향후 연구

모른다는 점은 솔직히 인정하는 답변도 포함.`,
  },

  // ── 전략 / 생산성 ──
  {
    id: 'snip-roadmap',
    title: { ko: '학기 연구 로드맵', en: 'Semester research roadmap' },
    description: { ko: '학기/분기 단위 연구 로드맵을 수립합니다.', en: 'Plan a semester/quarter research roadmap.' },
    category: 'strategy',
    tags: ['계획', '로드맵'],
    content: `아래 정보로 연구 로드맵을 수립해 주세요.
- 목표(논문/과제/학위):
- 기간: [예) 이번 학기 16주]
- 자원(GPU/데이터/협력자):
- 현재 진척:

출력:
1. 주차별 마일스톤 (W1~W16)
2. 각 마일스톤의 산출물
3. 위험 요소와 대안(Plan B)
4. 데드라인(학회/중간발표 등) 역산 일정`,
  },
  {
    id: 'snip-prioritize',
    title: { ko: '연구 아이디어 우선순위', en: 'Prioritize research ideas' },
    description: { ko: '여러 아이디어의 우선순위를 기준 기반으로 매깁니다.', en: 'Score and rank research ideas.' },
    category: 'strategy',
    tags: ['의사결정', '우선순위'],
    content: `아래 연구 아이디어들을 우선순위로 정렬해 주세요.
아이디어 목록:
[아이디어 1]
[아이디어 2]
[아이디어 3]

평가 기준(각 1~5점):
- 임팩트(학계/산업 기여)
- 실현 가능성(자원/시간)
- 참신함(novelty)
- 리스크

기준별 점수표 → 가중합 → 최종 순위. Top 1을 당장 시작하기 위한 첫 2주 액션을 제시.`,
  },
  {
    id: 'snip-meeting',
    title: { ko: '미팅 준비/요약', en: 'Meeting prep & notes' },
    description: { ko: '연구 미팅(랩미팅/어드바이저)을 준비하고 정리합니다.', en: 'Prepare for and summarize a research meeting.' },
    category: 'strategy',
    tags: ['미팅', '정리'],
    content: `아래 연구 미팅을 준비/요약해 주세요.
종류: [랩미팅 / 어드바이저 1:1 / 협력처]
상황: [진행 중인 작업 요약]

출력 두 파트:
**[사전]** 논의 안건 3~5개, 내 입장/질문, 보여줄 자료 체크리스트
**[사후]** 결정 사항, 액션 아이템(담당자/기한), 다음 미팅까지 할 일 — 템플릿 제공`,
  },
  {
    id: 'snip-reproduce',
    title: { ko: '논문 재현 계획', en: 'Paper reproduction plan' },
    description: { ko: '논문을 재현하기 위한 체계적 계획을 세웁니다.', en: 'Plan to reproduce a paper.' },
    category: 'strategy',
    tags: ['재현', '실험'],
    content: `아래 논문을 재현하기 위한 계획을 세워 주세요.
논문: [제목/PDF]

출력:
1. 재현 우선순위 (핵심 결과 1개 먼저)
2. 필요 자원(GPU/데이터/코드)
3. 단계별 체크리스트: 환경 세팅 → 데이터 준비 → 모델 구현 → 학습 → 평가
4. 저자 코드 유무에 따른 분기
5. 재현 성공 기준(지표 허용 오차)
6. 막혔을 때 점검 포인트`,
  },
  {
    id: 'snip-literature-search',
    title: { ko: '빠른 문헌 검색', en: 'Quick literature search' },
    description: { ko: '주제의 핵심 논문을 빠르게 찾아 목록화합니다.', en: 'Quickly surface key papers for a topic.' },
    category: 'learning',
    tags: ['문헌', '검색'],
    content: `주제 "[주제]"의 핵심 논문을 찾아 정리해 주세요.

1. **반드시 읽을 논문 5편** (foundational / SOTA): 제목, 저자, 연도, 한 줄 요약, 왜 읽어야 하는지
2. **최근(최근 2년) 주요 논문 5편**
3. **관련 survey 논문 1~2편**
4. 각 논문의 공식 링크(arXiv/학회) — 실제 존재하는 것만. 모르면 "[verification needed]"

허위 인용 절대 금지.`,
  },
  {
    id: 'snip-idea-brainstorm',
    title: { ko: '아이디어 브레인스토밍', en: 'Idea brainstorm' },
    description: { ko: '주제에 대해 다각도로 연구 아이디어를 발산합니다.', en: 'Brainstorm research ideas from many angles.' },
    category: 'ideation',
    tags: ['아이디어', '브레인스토밍'],
    content: `주제 "[주제]"에 대해 연구 아이디어를 다음 각도에서 3개씩, 총 15개 발산해 주세요:
1. 기존 방법의 약점 공략
2. 다른 분야 기법 차용(물리/생물/통계)
3. 새로운 데이터/평가 관점
4. 스케일/효율성 극단화
5. 윤리/안전/신뢰성 관점

각 아이디어: 한 줄 설명 + 왜 흥미로운지 + 대략적 난이도. 평가는 나중에 하므로 양 위주로.`,
  },
];
