import OpenAI from 'openai';

let client: OpenAI | null = null;

export function getOpenAIClient(): OpenAI {
  if (!client) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('OPENAI_API_KEY is not set in environment variables');
    }
    client = new OpenAI({ apiKey });
  }
  return client;
}

export interface GenerateOptions {
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
}

export async function generateWithAI(
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
    throw new Error('No content in response');
  }

  return content;
}

export interface BlogGenerationResult {
  title: string;
  content: string;
  excerpt: string;
  suggestedTags: string[];
}

export function parseGeneratedContent(rawContent: string): BlogGenerationResult {
  // Extract title from first heading
  const titleMatch = rawContent.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : 'Untitled';

  // Extract first paragraph as excerpt
  const paragraphs = rawContent
    .split('\n\n')
    .filter((p) => p.trim() && !p.startsWith('#'));
  const excerpt =
    paragraphs[0]?.replace(/[*_`]/g, '').slice(0, 200) + '...' ||
    'No excerpt available';

  // Try to extract tags from content
  const tagMatch = rawContent.match(/태그[:：]\s*(.+)/i);
  const suggestedTags = tagMatch
    ? tagMatch[1].split(/[,，、]/).map((t) => t.trim())
    : [];

  return {
    title,
    content: rawContent,
    excerpt,
    suggestedTags,
  };
}
