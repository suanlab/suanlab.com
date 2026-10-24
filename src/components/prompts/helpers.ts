import { defaultPromptValues } from '@/lib/prompt-state';
import { type PromptBuilder, type PromptSnippet, type LocalizedText } from '@/data/prompts';
import type { CustomSnippet } from './types';

export function estimateTokens(text: string): number {
  if (!text) return 0;
  const korean = (text.match(/[가-힣]/g) || []).length;
  const other = text.length - korean;
  return Math.round(korean / 1.8 + other / 4);
}

export function asLocalized(s: string): LocalizedText {
  return { ko: s, en: s };
}

export function prefillBuilder(b: PromptBuilder, prevOutput: string, language: 'ko' | 'en'): Record<string, string | string[]> {
  const v = defaultPromptValues(b.fields, language);
  const target = b.fields.find(f => f.id === 'context') ?? b.fields.find(f => f.type === 'textarea');
  if (target) v[target.id] = prevOutput;
  return v;
}

export function customToSnippet(c: CustomSnippet): PromptSnippet {
  return { id: c.id, title: asLocalized(c.title), description: asLocalized(''), category: c.category, tags: c.tags, content: c.content };
}
