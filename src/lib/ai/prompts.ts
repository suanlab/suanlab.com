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
- 수식은 반드시 다음 형식 사용:
  - 인라인 수식: \`$수식$\` (예: \`$x^2 + y^2 = z^2$\`)
  - 블록 수식: \`$$수식$$\` (예: \`$$\\sum_{i=1}^{n} x_i$$\`)
  - 절대 \`( ... )\` 또는 \`\\( ... \\)\` 형식 사용 금지
- 전문 용어는 처음 등장 시 한글(영어) 형태로 표기
- 참고 자료 링크 포함

## 출력 형식
- Frontmatter 없이 마크다운 본문만 작성
- 제목은 \`#\`로 시작하는 첫 번째 헤딩으로 작성
- 섹션은 \`##\`로 구분
`;

export const PAPER_SUMMARY_PROMPT = `당신은 AI/ML 논문 리뷰 전문가입니다. 다음 논문에 대해 **심층적이고 포괄적인** 한국어 블로그 포스트를 작성해주세요.

## 논문 정보
- 제목: {{TITLE}}
- 저자: {{AUTHORS}}
- 발표일: {{DATE}}
- arXiv ID: {{ARXIV_ID}}

## 논문 본문
{{PAPER_CONTENT}}

## 작성 지침

### 분량 요구사항 (매우 중요!)
- **최소 3000단어 이상** 작성
- 각 섹션을 충실하고 상세하게 작성
- 핵심 수식 최소 3개 이상 포함

### 필수 섹션 구조

1. **TL;DR** (5-7문장): 핵심 문제, 해결책, 주요 결과, 의의

2. **연구 배경 및 동기** (400단어 이상):
   - 분야 현황과 기존 접근법의 구체적 한계점
   - 이 연구가 해결하는 gap과 연구 질문

3. **관련 연구** (300단어 이상):
   - 선행 연구 5개 이상 분석
   - 본 논문과의 차별점 표로 정리

4. **핵심 기여** (200단어 이상):
   - 주요 기여점 번호 목록으로 정리
   - 각 기여의 novelty 설명

5. **제안 방법론** (600단어 이상):
   - 핵심 아이디어와 이론적 근거
   - 모델 아키텍처 상세 설명
   - **핵심 수식** (각 수식의 의미와 각 항 설명)

6. **실험 설정** (400단어 이상):
   - 데이터셋, 평가 지표, 베이스라인
   - 하이퍼파라미터 표로 정리

7. **실험 결과 분석** (500단어 이상):
   - 주요 결과 표로 정리
   - 성능 향상률(%) 계산
   - Ablation study 분석

8. **비판적 평가** (300단어 이상):
   - 강점 3-5개
   - 한계점과 개선 방향
   - 재현성 평가

9. **향후 연구 방향** (200단어 이상):
   - 확장 가능성과 적용 분야

10. **실무 적용 가이드** (200단어 이상):
    - 구현 시 고려사항과 팁

11. **결론**: 핵심 기여 요약과 영향

12. **참고 자료**: 논문 링크, 코드 저장소, 관련 자료

### 스타일 규칙
- 전문 용어: 한글(영어) 형태로 표기
- 인라인 수식: \`$수식$\` 형식
- 블록 수식: \`$$수식$$\` 형식
- **절대** \`( ... )\` 또는 \`\\( ... \\)\` 형식 사용 금지
- 중요 내용 **굵게** 강조
- 비교 데이터는 표 활용

## 출력
- Frontmatter 없이 마크다운 본문만 작성
- 제목: \`# [논문 리뷰] 논문제목\`
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
