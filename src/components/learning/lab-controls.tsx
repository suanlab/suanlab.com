'use client';
import React, { useEffect, useId, useRef, useState } from 'react';
import { useOptionalLanguage } from '@/components/language-provider';
import { parseNumberDraft } from '@/lib/learning-number-input';
import { formatNumber as number, type Vec3 } from '@/lib/learning-simulations';

/** Valid edits remain live. Invalid drafts never replace the current model value. */
export function NumberField({ label, value, onChange, min, max, step = .5, integer = false, resetKey }: { label: string; value: number; onChange: (n: number) => void; min: number; max: number; step?: number | 'any'; integer?: boolean; resetKey?: unknown }) {
  const language = useOptionalLanguage()?.language ?? 'en', id = useId();
  const [draft, setDraft] = useState(String(value)), [error, setError] = useState<'range' | 'constraint' | null>(null);
  const field = useRef<HTMLInputElement>(null), proposal = useRef<number | null>(null), prior = useRef({ value, resetKey });
  useEffect(() => {
    const changed = prior.current.value !== value || prior.current.resetKey !== resetKey;
    prior.current = { value, resetKey };
    if (proposal.current !== null) {
      const accepted = value === proposal.current; proposal.current = null;
      if (!accepted) setError('constraint');
      else if (document.activeElement !== field.current) setDraft(String(value));
    } else if (changed) { setDraft(String(value)); setError(null); }
  }, [value, resetKey, draft]);
  const finish = () => {
    const next = parseNumberDraft(draft, min, max, integer);
    if (next === null) setError('range');
    else if (next !== value) setError('constraint');
    else { setDraft(String(value)); setError(null); }
  };
  const message = error === 'constraint'
    ? language === 'ko' ? '이 값은 준비 조건과 맞지 않아 적용하지 않았습니다. 이전 설정을 유지합니다. Escape로 입력을 되돌릴 수 있습니다.' : 'This value was not applied because it conflicts with the preparation constraints. The previous setting is retained. Press Escape to restore the field.'
    : language === 'ko' ? `${min}부터 ${max}까지의 ${integer ? '정수' : '유한한 수'}를 입력하세요. 작은 비영 값을 0으로 바꾸지 않습니다. Escape로 이전 값을 복원할 수 있습니다.` : `Enter a ${integer ? 'whole' : 'finite'} number from ${min} to ${max}. A nonzero underflow is not applied as zero. Press Escape to restore the previous value.`;
  return <label className="ai-number-field"><span id={`${id}-label`}>{label}</span><input data-shared-number-field data-number-integer={integer} ref={field} type="number" min={min} max={max} step={step} value={draft} aria-labelledby={`${id}-label`} aria-invalid={!!error} aria-describedby={error ? `${id}-error` : undefined} onChange={e => {
    const raw = e.target.value; setDraft(raw); setError(null);
    const next = parseNumberDraft(raw, min, max, integer);
    if (next !== null && next !== value) { proposal.current = next; onChange(next); }
  }} onBlur={finish} onKeyDown={e => {
    if (e.key === 'Enter') { e.preventDefault(); e.currentTarget.blur(); }
    if (e.key === 'Escape') { e.preventDefault(); proposal.current = null; setDraft(String(value)); setError(null); }
  }} />{error && <span id={`${id}-error`} role="alert" style={{ overflowWrap: 'anywhere' }}>{message}</span>}</label>;
}
export function VectorFields({ label, value, change, min = -5, max = 5, resetKey }: { label: string; value: Vec3; change: (v: Vec3) => void; min?: number; max?: number; resetKey?: unknown }) {
  return <fieldset className="ai-vector-fields"><legend>{label}</legend><div className="ai-three-fields">{value.map((v, i) => <NumberField key={i} resetKey={resetKey} label={`${label} · ${['x', 'y', 'z'][i]}`} value={v} min={min} max={max} onChange={n => change(value.map((x, j) => j === i ? n : x) as Vec3)} />)}</div></fieldset>;
}
export function Tuple({ value }: { value: number[] }) { return <>({value.map(number).join(', ')})</>; }
