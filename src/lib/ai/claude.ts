import OpenAI from 'openai';

let openaiClient: OpenAI | null = null;

export function getOpenAIClient(): OpenAI {
  if (!openaiClient) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('OPENAI_API_KEY is not set in environment variables');
    }
    openaiClient = new OpenAI({ apiKey });
  }
  return openaiClient;
}

export interface GenerateOptions {
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
}

/**
 * Generate text using OpenAI GPT-4o
 */
export async function generateWithOpenAI(
  prompt: string,
  options: GenerateOptions = {}
): Promise<string> {
  const openai = getOpenAIClient();

  const {
    model = 'gpt-4o',
    maxTokens = 4096,
    temperature = 0.7,
    systemPrompt,
  } = options;

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];

  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt });
  }
  messages.push({ role: 'user', content: prompt });

  const response = await openai.chat.completions.create({
    model,
    max_tokens: maxTokens,
    temperature,
    messages,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    throw new Error('No content in OpenAI response');
  }

  return content;
}

/**
 * Generate text using Gemini API
 * Note: Gemini 2.5 Pro is a "thinking" model where internal reasoning tokens
 * are counted against maxOutputTokens. We need to set maxOutputTokens much higher
 * than the desired output length to accommodate thinking overhead.
 */
export async function generateWithGemini(
  prompt: string,
  options: GenerateOptions = {}
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not set in environment variables');
  }

  const {
    model = 'gemini-2.5-pro',
    maxTokens = 4096,
    temperature = 0.7,
    systemPrompt,
  } = options;

  const isThinkingModel = model.includes('2.5');

  const contents = [];

  if (systemPrompt) {
    contents.push({
      role: 'user',
      parts: [{ text: `System: ${systemPrompt}\n\nUser: ${prompt}` }]
    });
  } else {
    contents.push({
      role: 'user',
      parts: [{ text: prompt }]
    });
  }

  // For thinking models, maxOutputTokens includes both thinking and output tokens.
  // Set it to 65536 (max) to ensure sufficient room for both.
  const effectiveMaxTokens = isThinkingModel
    ? Math.max(maxTokens * 3, 65536)
    : maxTokens;

  const generationConfig: Record<string, unknown> = {
    maxOutputTokens: effectiveMaxTokens,
    temperature,
  };

  // For thinking models, set a thinking budget to control reasoning overhead
  if (isThinkingModel) {
    generationConfig.thinkingConfig = {
      thinkingBudget: 8192,
    };
  }

  // Set a 5-minute timeout for thinking models (they take longer)
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), isThinkingModel ? 300000 : 120000);

  let response: Response;
  try {
    response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents,
          generationConfig,
        }),
        signal: controller.signal,
      }
    );
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();

  // Extract text content from parts (skip thinking parts if present)
  const parts = data.candidates?.[0]?.content?.parts || [];
  const textParts = parts.filter((p: { text?: string; thought?: boolean }) => p.text && !p.thought);
  const content = textParts.length > 0
    ? textParts.map((p: { text: string }) => p.text).join('')
    : parts[0]?.text;

  if (!content) {
    const finishReason = data.candidates?.[0]?.finishReason;
    const thoughtTokens = data.usageMetadata?.thoughtsTokenCount || 0;
    throw new Error(
      `No content in Gemini response (finishReason: ${finishReason}, thoughtTokens: ${thoughtTokens})`
    );
  }

  return content;
}

/**
 * Check if content is a refusal message
 */
function isRefusalMessage(content: string): boolean {
  const refusalPatterns = [
    "I'm sorry, but I can't assist",
    "I cannot assist",
    "I'm unable to",
    "I can't help with",
    "I cannot help with",
    "I'm not able to",
    "sorry, but I cannot",
    "I apologize, but I cannot"
  ];
  return refusalPatterns.some(pattern =>
    content.toLowerCase().includes(pattern.toLowerCase())
  );
}

/**
 * Generate content with dual-AI enhancement
 * Step 1: OpenAI generates initial draft
 * Step 2: Gemini reviews and enhances the content
 * Fallback: If OpenAI refuses, use Gemini only
 */
export async function generateWithDualAI(
  prompt: string,
  options: GenerateOptions = {}
): Promise<string> {
  console.log('🤖 Step 1: OpenAI GPT-4o로 초안 생성 중...');

  // Step 1: Generate initial draft with OpenAI
  let initialDraft: string;
  try {
    initialDraft = await generateWithOpenAI(prompt, options);
  } catch {
    console.log('⚠️ OpenAI 오류 발생, Gemini로 대체합니다...');
    return generateWithGemini(prompt, options);
  }

  // Check for refusal or too short content (paper reviews should be 3000+ words)
  if (!initialDraft || initialDraft.length < 1000 || isRefusalMessage(initialDraft)) {
    console.log(`⚠️ OpenAI 거부 또는 응답 부족 (${initialDraft?.length || 0} chars), Gemini로 대체합니다...`);
    return generateWithGemini(prompt, { ...options, maxTokens: 16384 });
  }

  console.log(`✅ OpenAI 초안 생성 완료 (${initialDraft.length} chars)`);
  console.log('🔄 Step 2: Gemini로 콘텐츠 보강 중...');

  try {
    // Step 2: Enhance with Gemini
    const enhancementPrompt = `당신은 기술 블로그 편집자입니다. 다음 블로그 포스트 초안을 검토하고 보강해주세요.

## 보강 지침:
1. **정확성 검증**: 기술적 내용이 정확한지 확인하고, 필요시 수정
2. **예제 보강**: 코드 예제나 실제 사용 사례를 추가하거나 개선
3. **설명 명확화**: 어려운 개념에 대한 설명을 더 명확하게
4. **최신 정보**: 최신 트렌드나 버전 정보가 있다면 반영
5. **구조 개선**: 논리적 흐름이 자연스럽도록 조정

## 주의사항:
- 원본의 마크다운 형식(#, ##, ###, 코드블록 등) 유지
- 원본의 전체적인 구조와 스타일 유지
- 불필요하게 내용을 늘리지 말고, 품질 향상에 집중
- 수식은 반드시 \`$수식$\` (인라인) 또는 \`$$수식$$\` (블록) 형식 사용
- \`( ... )\` 또는 \`\\( ... \\)\` 형식의 수식은 \`$...$\` 형식으로 변환
- 결과물은 마크다운 형식의 블로그 포스트만 출력 (메타 설명 없이)

## 원본 초안:
${initialDraft}

## 보강된 버전:`;

    const enhancedContent = await generateWithGemini(enhancementPrompt, {
      ...options,
      maxTokens: 16384,
      temperature: 0.5,
    });

    // Validate enhanced content - check if Gemini actually enhanced it
    if (!enhancedContent ||
        enhancedContent.length < 100 ||
        enhancedContent.includes('초안이 제공되지 않았') ||
        enhancedContent.includes('초안을 제공해주시면')) {
      console.log('⚠️ Gemini 보강 실패, OpenAI 초안 사용');
      return initialDraft;
    }

    console.log(`✅ Gemini 보강 완료 (${enhancedContent.length} chars)`);
    return enhancedContent;
  } catch (error) {
    console.log('⚠️ Gemini 보강 중 오류 발생, OpenAI 초안 사용:', error);
    return initialDraft;
  }
}

// Backward compatibility alias
export const generateWithAI = generateWithDualAI;

export interface BlogGenerationResult {
  title: string;
  content: string;
  excerpt: string;
  suggestedTags: string[];
}

export function parseGeneratedContent(rawContent: string): BlogGenerationResult {
  // Remove markdown code block wrappers if present
  let content = rawContent.trim();
  if (content.startsWith('```markdown')) {
    content = content.replace(/^```markdown\s*\n?/, '').replace(/\n?```\s*$/, '');
  } else if (content.startsWith('```')) {
    content = content.replace(/^```\s*\n?/, '').replace(/\n?```\s*$/, '');
  }

  // Extract title from first heading
  const titleMatch = content.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : 'Untitled';

  // Extract first paragraph as excerpt
  const paragraphs = content
    .split('\n\n')
    .filter((p) => p.trim() && !p.startsWith('#'));
  const excerpt =
    paragraphs[0]?.replace(/[*_`]/g, '').slice(0, 200) + '...' ||
    'No excerpt available';

  // Try to extract tags from content
  const tagMatch = content.match(/태그[:：]\s*(.+)/i);
  const suggestedTags = tagMatch
    ? tagMatch[1].split(/[,，、]/).map((t) => t.trim())
    : [];

  return {
    title,
    content,
    excerpt,
    suggestedTags,
  };
}
