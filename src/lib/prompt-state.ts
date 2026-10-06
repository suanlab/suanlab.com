export interface PromptFieldShape {
  id: string;
  type: string;
  options?: { value: string }[];
  default?: string;
}
export type PromptValues = Record<string, string | string[]>;

export function defaultPromptValues(fields: PromptFieldShape[], language: 'ko' | 'en'): PromptValues {
  return Object.fromEntries(fields.map(field => [field.id,
    field.type === 'multiselect' ? [] :
      field.id === 'lang' && field.options?.some(o => o.value === language) ? language :
        field.default ?? (field.type === 'select' ? field.options?.[0]?.value ?? '' : ''),
  ]));
}

/** Shared links and stored drafts are untrusted input, including field types. */
export function normalizePromptValues(fields: PromptFieldShape[], input: unknown): PromptValues {
  if (!input || typeof input !== 'object' || Array.isArray(input)) return {};
  const result: PromptValues = {};
  for (const field of fields) {
    const value = (input as Record<string, unknown>)[field.id];
    if (field.type === 'multiselect') {
      if (Array.isArray(value)) result[field.id] = [...new Set(value.filter((v): v is string => typeof v === 'string' && !!field.options?.some(o => o.value === v)))];
    } else if (typeof value === 'string' && (field.type !== 'select' || field.options?.some(o => o.value === value))) result[field.id] = value;
  }
  return result;
}

export function encodePromptShare(builderId: string, values: PromptValues): string {
  return btoa(encodeURIComponent(JSON.stringify({ b: builderId, v: values })));
}
export function decodePromptShare(hash: string): { b: string; v: unknown } | null {
  try {
    const value = JSON.parse(decodeURIComponent(atob(hash.replace(/^#/, ''))));
    return value && typeof value.b === 'string' && value.v && typeof value.v === 'object' && !Array.isArray(value.v) ? value : null;
  } catch { return null; }
}

export function detectPromptVariables(content: string): string[] {
  return [...new Set(Array.from(content.matchAll(/\{\{([\p{L}\p{N}_-]+)\}\}/gu), match => match[1]))];
}
export function substitutePromptVariables(content: string, values: Record<string, string>): string {
  return content.replace(/\{\{([\p{L}\p{N}_-]+)\}\}/gu, (_, name) => values[name]?.trim() || `{{${name}}}`);
}
