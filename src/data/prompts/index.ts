export type PromptCategory =
  | 'ideation'
  | 'quality'
  | 'writing'
  | 'strategy'
  | 'learning'
  | 'coding';

export interface LocalizedText {
  ko: string;
  en: string;
}

export interface PromptCategoryInfo {
  id: PromptCategory;
  label: LocalizedText;
  color: string;
}

export type FieldType = 'text' | 'textarea' | 'select' | 'multiselect';

export interface PromptFieldOption {
  value: string;
  label: LocalizedText;
}

export interface PromptField {
  id: string;
  label: LocalizedText;
  type: FieldType;
  required?: boolean;
  placeholder?: LocalizedText;
  help?: LocalizedText;
  options?: PromptFieldOption[];
  default?: string;
}

export interface PromptBuilder {
  id: string;
  title: LocalizedText;
  description: LocalizedText;
  category: PromptCategory;
  icon: string;
  tags: string[];
  fields: PromptField[];
  generate: (values: Record<string, string | string[]>) => string;
}

export interface PromptSnippet {
  id: string;
  title: LocalizedText;
  description: LocalizedText;
  category: PromptCategory;
  tags: string[];
  content: string;
}

export const promptCategories: PromptCategoryInfo[] = [
  {
    id: 'ideation',
    label: { ko: '연구 설계', en: 'Ideation' },
    color: 'bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300',
  },
  {
    id: 'quality',
    label: { ko: '품질 검증', en: 'Quality' },
    color: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
  },
  {
    id: 'writing',
    label: { ko: '작성', en: 'Writing' },
    color: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300',
  },
  {
    id: 'strategy',
    label: { ko: '전략', en: 'Strategy' },
    color: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
  },
  {
    id: 'learning',
    label: { ko: '학습 / 분석', en: 'Learning' },
    color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300',
  },
  {
    id: 'coding',
    label: { ko: '코딩 / 구현', en: 'Coding' },
    color: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/40 dark:text-cyan-300',
  },
];

export { promptBuilders } from './builders';
export { promptSnippets } from './snippets';
