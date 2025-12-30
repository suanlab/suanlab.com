export const TOPIC_PROMPT_TEMPLATE = `당신은 SuanLab의 기술 블로그 작가입니다. 다음 주제에 대해 한국어로 고품질의 블로그 포스트를 작성해주세요.

## 주제
{{TOPIC}}

## 카테고리
{{CATEGORY}}

## 요청된 태그
{{TAGS}}

## 작성 지침

### 독자 대상
- 데이터 과학/AI에 관심있는 대학생 및 현업 개발자
- 기본적인 프로그래밍 지식을 갖춘 독자

### 분량
- 1500-2500 단어

### 구조
1. **도입부**: 주제 소개 및 왜 중요한지 설명
2. **본문**: 핵심 개념을 단계별로 설명, 코드 예제 포함 (Python 권장)
3. **결론**: 요약 및 추가 학습 자료 안내

### 스타일 가이드
- 친근하면서도 전문적인 톤 유지
- 코드 예제는 실행 가능하고 주석을 포함
- 필요시 수식은 LaTeX 형식 사용 (\`$$...$$\`)
- 전문 용어는 처음 등장 시 한글(영어) 형태로 표기
- 참고 자료 링크 포함

## 출력 형식
- Frontmatter 없이 마크다운 본문만 작성
- 제목은 \`#\`로 시작하는 첫 번째 헤딩으로 작성
- 섹션은 \`##\`로 구분
`;

export const PAPER_SUMMARY_PROMPT = `당신은 AI/ML 논문 리뷰 전문가입니다. 다음 논문 내용을 바탕으로 한국어로 블로그 포스트를 작성해주세요.

## 논문 정보
- 제목: {{TITLE}}
- 저자: {{AUTHORS}}
- 발표일: {{DATE}}
- arXiv ID: {{ARXIV_ID}}

## 논문 본문
{{PAPER_CONTENT}}

## 작성 지침

### 구조
1. **TL;DR**: 1-2문장으로 핵심 요약
2. **연구 동기 및 문제 정의**: 이 연구가 왜 필요한지
3. **제안하는 방법론**: 핵심 아이디어와 기술적 접근
4. **주요 실험 결과**: 정량적/정성적 결과
5. **한계점 및 향후 연구 방향**: 저자가 언급한 한계
6. **결론 및 개인 의견**: 논문에 대한 리뷰어의 생각

### 스타일 가이드
- 전문 용어는 처음 등장 시 한글(영어) 형태로 표기
- 핵심 수식은 LaTeX 형식으로 포함 (\`$$...$$\`)
- 논문의 Figure나 Table 설명 포함 (가능한 경우)
- 관련 논문 및 추가 자료 링크 제안

## 출력 형식
- Frontmatter 없이 마크다운 본문만 작성
- 제목은 \`# [논문 리뷰] 논문제목\` 형식으로 시작
- 섹션은 \`##\`로 구분
`;

export const PAPER_CHUNK_SUMMARY_PROMPT = `다음은 학술 논문의 일부입니다. 핵심 내용을 요약해주세요.

## 논문 내용
{{CHUNK}}

## 요약 지침
다음 사항에 집중하여 요약해주세요:
1. 주요 개념과 정의
2. 제안하는 방법론
3. 수식과 알고리즘 (있는 경우)
4. 실험 설정과 결과 (있는 경우)

간결하지만 중요한 세부사항은 놓치지 마세요. 한국어로 작성해주세요.
`;

export const SYNTHESIS_PROMPT = `다음은 논문의 여러 부분을 요약한 것입니다. 이를 통합하여 완전한 블로그 포스트를 작성해주세요.

## 논문 메타데이터
{{METADATA}}

## 섹션별 요약
{{SUMMARIES}}

## 작성 지침
${PAPER_SUMMARY_PROMPT.split('## 작성 지침')[1]}
`;

export function buildTopicPrompt(options: {
  topic: string;
  category?: string;
  tags?: string[];
}): string {
  return TOPIC_PROMPT_TEMPLATE.replace('{{TOPIC}}', options.topic)
    .replace('{{CATEGORY}}', options.category || 'General')
    .replace('{{TAGS}}', (options.tags || []).join(', ') || '자동 생성');
}

export function buildPaperPrompt(options: {
  title: string;
  authors: string;
  date: string;
  arxivId: string;
  content: string;
}): string {
  return PAPER_SUMMARY_PROMPT.replace('{{TITLE}}', options.title)
    .replace('{{AUTHORS}}', options.authors)
    .replace('{{DATE}}', options.date)
    .replace('{{ARXIV_ID}}', options.arxivId || 'N/A')
    .replace('{{PAPER_CONTENT}}', options.content);
}

export function buildChunkPrompt(chunk: string): string {
  return PAPER_CHUNK_SUMMARY_PROMPT.replace('{{CHUNK}}', chunk);
}

export function buildSynthesisPrompt(
  metadata: string,
  summaries: string[]
): string {
  return SYNTHESIS_PROMPT.replace('{{METADATA}}', metadata).replace(
    '{{SUMMARIES}}',
    summaries.map((s, i) => `### Part ${i + 1}\n${s}`).join('\n\n')
  );
}
