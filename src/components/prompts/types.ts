import type { PromptCategory } from '@/data/prompts';

export interface CustomSnippet {
  id: string;
  title: string;
  content: string;
  category: PromptCategory;
  tags: string[];
}
