'use client';

import { useState, useMemo, useEffect } from 'react';
import { defaultPromptValues, normalizePromptValues } from '@/lib/prompt-state';
import { Button } from '@/components/ui/button';
import { Copy, Check, Download, Star, ArrowLeft, RotateCcw, ExternalLink, Share2, RefreshCw, ChevronRight, Lightbulb as TipIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type PromptBuilder, type LocalizedText, type PromptWorkflow } from '@/data/prompts';
import FieldInput from './FieldInput';
import { estimateTokens } from './helpers';

export default function BuilderDetail({
  builder, initialValues, workflowCtx, onBack, isFav, onToggleFav, onShare, shared, shareFailed, onWorkflowDone, language, L,
}: {
  builder: PromptBuilder;
  initialValues: Record<string, string | string[]> | null;
  workflowCtx: { wf: PromptWorkflow; step: number } | null;
  onBack: () => void;
  isFav: boolean;
  onToggleFav: () => void;
  onShare: (id: string, values: Record<string, string | string[]>) => void;
  shared: boolean;
  shareFailed: boolean;
  onWorkflowDone: (output: string) => void;
  language: 'ko' | 'en';
  L: (t: LocalizedText) => string;
}) {
  const draftKey = `suanlab-prompt-draft-${builder.id}`;
  const [draft] = useState(() => {
    try {
      const value = JSON.parse(localStorage.getItem(draftKey) ?? '{}');
      return { values: normalizePromptValues(builder.fields, value?.values), output: typeof value?.output === 'string' ? value.output : '', edited: value?.edited === true };
    } catch { return { values: {}, output: '', edited: false }; }
  });
  const [values, setValues] = useState<Record<string, string | string[]>>(() => ({
    ...defaultPromptValues(builder.fields, language),
    ...(initialValues ? {} : draft.values), ...normalizePromptValues(builder.fields, initialValues),
  }));
  const [output, setOutput] = useState(initialValues ? '' : draft.output);
  const [edited, setEdited] = useState(!initialValues && draft.edited);
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  const [missing, setMissing] = useState<Set<string>>(new Set());
  const [showExample, setShowExample] = useState(false);
  const [workflowResult, setWorkflowResult] = useState('');
  const [storageBlocked, setStorageBlocked] = useState(false);

  const prompt = useMemo(() => {
    try {
      return builder.generate(values);
    } catch {
      return '';
    }
  }, [builder, values]);

  useEffect(() => { if (!edited) setOutput(prompt); }, [prompt, edited]);
  useEffect(() => {
    try { localStorage.setItem(draftKey, JSON.stringify({ values, output, edited })); }
    catch { setStorageBlocked(true); }
  }, [draftKey, values, output, edited]);

  const setValue = (id: string, val: string | string[]) => {
    setValues((prev) => ({ ...prev, [id]: val }));
    setMissing((prev) => {
      if (!prev.has(id)) return prev;
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  };

  const validate = () => {
    const reqMissing = builder.fields.filter((f) => f.required && f.type !== 'multiselect' && !String(values[f.id] ?? '').trim());
    const mulMissing = builder.fields.filter((f) => f.required && f.type === 'multiselect' && (values[f.id] as string[]).length === 0);
    if ([...reqMissing, ...mulMissing].length > 0) {
      setMissing(new Set([...reqMissing, ...mulMissing].map((f) => f.id)));
      return false;
    }
    return true;
  };
  const handleCopy = async () => {
    if (!validate() || !output.trim()) return;
    setCopyFailed(false);
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopyFailed(true);
    }
  };

  const handleDownload = () => {
    if (!validate() || !output.trim()) return;
    const blob = new Blob([`# ${L(builder.title)}\n\n${output}\n`], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${builder.id}.md`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const tokens = estimateTokens(output);

  return (
    <div>
      <div className="flex items-center justify-between gap-2 mb-4">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="mr-1.5 h-4 w-4" />
          {language === 'ko' ? '목록으로' : 'Back'}
        </Button>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={() => {
            setValues(defaultPromptValues(builder.fields, language));
            setEdited(false); setMissing(new Set()); setWorkflowResult('');
          }}><RotateCcw className="mr-1.5 h-4 w-4" />{language === 'ko' ? '초안 초기화' : 'Clear draft'}</Button>
          <Button variant="ghost" size="sm" onClick={onToggleFav}>
            <Star className={cn('mr-1.5 h-4 w-4', isFav && 'fill-amber-400 text-amber-400')} />
            {isFav ? (language === 'ko' ? '즐겨찾기됨' : 'Favorited') : (language === 'ko' ? '즐겨찾기' : 'Favorite')}
          </Button>
        </div>
      </div>

      {workflowCtx && (
        <div className="mb-4 rounded-md border border-primary/30 bg-primary/5 p-3 text-sm">
          <p className="font-medium">{language === 'ko' ? `워크플로우: ${L(workflowCtx.wf.title)}` : `Workflow: ${L(workflowCtx.wf.title)}`}</p>
          <p className="text-xs text-muted-foreground">{language === 'ko' ? `단계 ${workflowCtx.step + 1} / ${workflowCtx.wf.steps.length}` : `Step ${workflowCtx.step + 1} / ${workflowCtx.wf.steps.length}`}</p>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Form */}
        <div>
          <h2 className="text-xl font-bold mb-1">{L(builder.title)}</h2>
          <p className="text-sm text-muted-foreground mb-4">{L(builder.description)}</p>
          {builder.tips && builder.tips.length > 0 && (
            <div className="mb-4 rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
              <p className="text-xs font-semibold text-amber-700 dark:text-amber-400 mb-1 flex items-center gap-1"><TipIcon className="h-3.5 w-3.5" />{language === 'ko' ? '사용 팁' : 'Tips'}</p>
              <ul className="text-xs text-muted-foreground space-y-1 list-disc pl-4">
                {builder.tips.map((t, i) => <li key={i}>{L(t)}</li>)}
              </ul>
            </div>
          )}
          <div className="space-y-4">
            {builder.fields.map((f) => (
              <FieldInput key={f.id} field={f} value={values[f.id]} onChange={(val) => setValue(f.id, val)} invalid={missing.has(f.id)} L={L} />
            ))}
          </div>
          {builder.example && (
            <div className="mt-4">
              <button onClick={() => setShowExample((s) => !s)} className="text-xs text-primary hover:underline flex items-center gap-1">
                <ChevronRight className={cn('h-3 w-3 transition-transform', showExample && 'rotate-90')} />
                {language === 'ko' ? '이런 결과가 나와요 (예시)' : 'Example output'}
              </button>
              {showExample && <pre className="mt-2 whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-3 text-xs font-mono max-h-60 overflow-y-auto">{builder.example}</pre>}
            </div>
          )}
        </div>

        {/* Output */}
        <div className="lg:sticky lg:top-24 self-start">
          <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
            <span className="text-sm font-medium text-muted-foreground">
              {language === 'ko' ? '생성된 프롬프트' : 'Generated prompt'}
              <span className="ml-2 text-[11px] text-muted-foreground">≈ {tokens} tokens</span>
            </span>
            <div className="flex gap-2">
              <Button size="sm" variant="ghost" onClick={() => { setOutput(prompt); setEdited(false); }} aria-label={language === 'ko' ? '현재 폼으로 프롬프트 업데이트' : 'Update prompt from current form'} title={language === 'ko' ? '재생성' : 'Regenerate'}>
                <RefreshCw className="h-3.5 w-3.5" />
              </Button>
              <Button size="sm" variant="ghost" onClick={() => onShare(builder.id, values)} aria-label={language === 'ko' ? '입력값을 포함한 공유 링크 복사' : 'Copy share link containing inputs'} title={language === 'ko' ? '공유 URL 복사' : 'Copy share URL'}>
                {shared ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Share2 className="h-3.5 w-3.5" />}
              </Button>
              <Button size="sm" variant="outline" onClick={handleDownload} disabled={!output}><Download className="mr-1 h-3.5 w-3.5" />.md</Button>
              <Button size="sm" onClick={handleCopy}>
                {copied ? <Check className="mr-1 h-3.5 w-3.5 text-green-500" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
                {copied ? (language === 'ko' ? '복사됨!' : 'Copied!') : (language === 'ko' ? '복사' : 'Copy')}
              </Button>
            </div>
          </div>
          {missing.size > 0 && <p role="alert" className="text-xs text-red-700 dark:text-red-300 mb-2">{language === 'ko' ? '필수 항목을 입력해 주세요.' : 'Please fill required fields.'}</p>}
          <textarea
            value={output}
            aria-label={language === 'ko' ? '생성된 프롬프트 편집' : 'Edit generated prompt'}
            onChange={(e) => { setOutput(e.target.value); setEdited(true); }}
            spellCheck={false}
            className="w-full min-h-[280px] max-h-[60vh] overflow-y-auto whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-4 text-xs font-mono leading-relaxed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            placeholder={language === 'ko' ? '왼쪽 폼을 채우면 프롬프트가 생성됩니다. 직접 수정도 가능합니다.' : 'Fill the form to generate. You can also edit directly.'}
          />
          <p className="mt-2 text-[11px] text-muted-foreground flex items-center gap-1">
            <ExternalLink className="h-3 w-3" />
            {language === 'ko' ? (edited ? '직접 수정한 내용을 유지합니다. 폼 변경을 반영하려면 업데이트 버튼을 누르세요.' : '폼을 채우면 자동 생성됩니다. 초안은 이 브라우저에 저장됩니다.') : (edited ? 'Your edits are preserved. Use Update to apply form changes.' : 'The form generates a prompt. Drafts are saved in this browser.')}
          </p>
          {(copyFailed || shareFailed) && <p role="alert" className="mt-2 text-xs text-red-700 dark:text-red-300">{language === 'ko' ? '복사하지 못했습니다. 프롬프트를 선택하여 직접 복사하세요.' : 'Copy failed. Select the prompt and copy it manually.'}</p>}
          {storageBlocked && <p role="status" className="mt-2 text-xs text-muted-foreground">{language === 'ko' ? '이 브라우저에서 초안을 저장하지 못했습니다.' : 'Draft storage is unavailable in this browser.'}</p>}
          <p className="mt-2 text-xs text-muted-foreground">{language === 'ko' ? '공유 링크에는 폼 입력값이 포함됩니다.' : 'Share links include the form inputs.'}</p>
          {workflowCtx && <div className="mt-5 rounded-lg border p-3">
            <label htmlFor="workflow-result" className="text-sm font-medium">{language === 'ko' ? 'AI 도구에서 실행한 결과' : 'Result from your AI tool'}</label>
            <textarea id="workflow-result" value={workflowResult} onChange={e => setWorkflowResult(e.target.value)} className="mt-2 min-h-28 w-full rounded-md border bg-background p-3 text-sm" />
            <p className="mt-2 text-xs text-muted-foreground">{language === 'ko' ? '실제 응답을 붙여넣으면 다음 단계의 입력으로 이어집니다.' : 'Paste the actual response to carry it into the next step.'}</p>
            <Button className="mt-3 w-full" size="sm" disabled={!workflowResult.trim()} onClick={() => { if (validate()) onWorkflowDone(workflowResult.trim()); }}>{language === 'ko' ? '결과 저장 & 단계 완료' : 'Save result & complete step'}</Button>
          </div>}
        </div>
      </div>
    </div>
  );
}
