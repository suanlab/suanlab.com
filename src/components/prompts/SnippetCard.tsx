'use client';

import { useState, useId } from 'react';
import { detectPromptVariables as detectVariables, substitutePromptVariables as substitute } from '@/lib/prompt-state';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Copy, Check, Star, Trash2, Edit3 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type PromptSnippet, type LocalizedText } from '@/data/prompts';


export default function SnippetCard({
  snippet, isCustom, expanded, onToggle, isFav, onToggleFav, values, onValueChange, onDelete, onEdit, catLabel, catColor, L, lang,
}: {
  snippet: PromptSnippet;
  isCustom: boolean;
  expanded: boolean;
  onToggle: () => void;
  isFav: boolean;
  onToggleFav: () => void;
  values: Record<string, string>;
  onValueChange: (name: string, val: string) => void;
  onDelete?: () => void;
  onEdit?: () => void;
  catLabel: string;
  catColor: string;
  L: (t: LocalizedText) => string;
  lang: 'ko' | 'en';
}) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const contentId = useId();
  const content = lang === 'en' && snippet.contentEn ? snippet.contentEn : snippet.content;
  const declared = detectVariables(content);
  const defaults = Object.fromEntries((snippet.variables ?? []).map(v => [v.name, v.default ?? '']));
  const filledValues = { ...defaults, ...values };
  const finalContent = substitute(content, filledValues);
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setCopyError(false);
    if (declared.some(name => !filledValues[name]?.trim())) { setCopyError(true); if (!expanded) onToggle(); return; }
    try {
      await navigator.clipboard.writeText(finalContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopyError(true);
    }
  };
  return (
    <Card id={`prompt-${snippet.id}`} tabIndex={-1} className="h-full flex flex-col transition-all hover:shadow-md">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base hover:text-primary transition-colors flex items-center gap-1.5">
            <button type="button" onClick={onToggle} aria-expanded={expanded} aria-controls={expanded ? contentId : undefined} className="text-left">{L(snippet.title)}</button>
            {isCustom && <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[9px] font-bold text-primary">MINE</span>}
          </CardTitle>
          <div className="flex items-center gap-1 shrink-0">
            {isCustom && onEdit && <button onClick={(e) => { e.stopPropagation(); onEdit(); }} aria-label="edit" className="text-muted-foreground hover:text-primary"><Edit3 className="h-4 w-4" /></button>}
            {isCustom && onDelete && <button onClick={(e) => { e.stopPropagation(); onDelete(); }} aria-label="delete" className="text-muted-foreground hover:text-red-500"><Trash2 className="h-4 w-4" /></button>}
            <button onClick={(e) => { e.stopPropagation(); onToggleFav(); }} aria-pressed={isFav} aria-label={lang === 'ko' ? '즐겨찾기 변경' : 'Toggle favorite'} className="text-muted-foreground hover:text-amber-500"><Star className={cn('h-5 w-5', isFav && 'fill-amber-400 text-amber-400')} /></button>
          </div>
        </div>
        <p className="text-sm text-muted-foreground line-clamp-2">{L(snippet.description)}</p>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        <div className="flex flex-wrap gap-1.5 mb-3">
          <span className={cn('inline-flex items-center rounded px-2 py-0.5 text-[10px] font-medium', catColor)}>{catLabel}</span>
          {snippet.tags.slice(0, 2).map((t) => <span key={t} className="inline-flex items-center rounded bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">#{t}</span>)}
          {declared.length > 0 && <span className="inline-flex items-center rounded bg-blue-100 px-2 py-0.5 text-[10px] text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">{declared.length} {lang === 'ko' ? '변수' : 'vars'}</span>}
        </div>
        {expanded && declared.length > 0 && (
          <div className="mb-3 space-y-2 rounded-md border p-2">
            {declared.map((name) => {
              const meta = snippet.variables?.find((vv) => vv.name === name);
              return (
                <div key={name}>
                  <label htmlFor={`${contentId}-${name}`} className="text-xs font-medium">{meta ? L(meta.label) : name}</label>
                  <Input id={`${contentId}-${name}`} value={values[name] ?? meta?.default ?? ''} onChange={(e) => onValueChange(name, e.target.value)} placeholder={name} className="mt-0.5 h-8 text-xs" />
                </div>
              );
            })}
          </div>
        )}
        {expanded && (
          <pre id={contentId} className="whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-3 text-xs font-mono leading-relaxed max-h-[40vh] overflow-y-auto mb-3">{finalContent}</pre>
        )}
        {copyError && <p role="alert" className="mb-2 text-xs text-red-700 dark:text-red-300">{lang === 'ko' ? '변수를 모두 채워 주세요. 복사가 실패하면 본문을 직접 선택해 복사하세요.' : 'Fill all variables. If copying fails, select and copy the text manually.'}</p>}
        <div className="mt-auto flex gap-2">
          <Button className="flex-1" size="sm" variant={expanded ? 'outline' : 'default'} onClick={onToggle}>{expanded ? (lang === 'ko' ? '접기' : 'Collapse') : (lang === 'ko' ? '보기' : 'View')}</Button>
          <Button size="sm" variant="outline" onClick={handleCopy}>{copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}<span className="ml-1">{copied ? (lang === 'ko' ? '복사됨' : 'Copied') : (lang === 'ko' ? '복사' : 'Copy')}</span></Button>
        </div>
      </CardContent>
    </Card>
  );
}
