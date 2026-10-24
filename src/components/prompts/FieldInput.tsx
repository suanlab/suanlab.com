'use client';

import { useId } from 'react';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { type PromptField, type LocalizedText } from '@/data/prompts';


export default function FieldInput({
  field, value, onChange, invalid, L,
}: {
  field: PromptField;
  value: string | string[];
  onChange: (v: string | string[]) => void;
  invalid: boolean;
  L: (t: LocalizedText) => string;
}) {
  const fieldId = useId();
  const label = (
    <label htmlFor={field.type === 'multiselect' ? undefined : fieldId} id={`${fieldId}-label`} className="text-sm font-medium flex items-center gap-1">
      {L(field.label)}
      {field.required && <span aria-hidden="true" className="text-red-700 dark:text-red-300">*</span>}
    </label>
  );
  const help = field.help && <p id={`${fieldId}-help`} className="text-xs text-muted-foreground mt-0.5">{L(field.help)}</p>;

  if (field.type === 'textarea') {
    return (
      <div>
        {label}
        <textarea
          id={fieldId} aria-required={field.required} aria-invalid={invalid} aria-describedby={field.help ? `${fieldId}-help` : undefined}
          value={value as string}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.placeholder ? L(field.placeholder) : undefined}
          className={cn('mt-1 flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2', invalid && 'border-red-500')}
        />
        {help}
      </div>
    );
  }
  if (field.type === 'select') {
    return (
      <div>
        {label}
        <select id={fieldId} aria-required={field.required} aria-invalid={invalid} aria-describedby={field.help ? `${fieldId}-help` : undefined} value={value as string} onChange={(e) => onChange(e.target.value)} className={cn('mt-1 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2', invalid && 'border-red-500')}>
          {field.options?.map((o) => <option key={o.value} value={o.value}>{L(o.label)}</option>)}
        </select>
        {help}
      </div>
    );
  }
  if (field.type === 'multiselect') {
    const arr = value as string[];
    const toggle = (val: string) => onChange(arr.includes(val) ? arr.filter((x) => x !== val) : [...arr, val]);
    return (
      <div>
        {label}
        <div role="group" aria-labelledby={`${fieldId}-label`} aria-describedby={field.help ? `${fieldId}-help` : undefined} className="mt-1.5 flex flex-wrap gap-1.5">
          {field.options?.map((o) => {
            const on = arr.includes(o.value);
            const lbl = L(o.label);
            return (
              <button key={o.value} type="button" onClick={() => toggle(o.value)} aria-pressed={on} className={cn('rounded-full border px-2.5 py-1 text-xs transition-colors', on ? 'border-primary bg-primary text-primary-foreground' : 'border-input bg-background hover:bg-accent')}>
                {lbl.length > 42 ? lbl.slice(0, 40) + '…' : lbl}
              </button>
            );
          })}
        </div>
        {help}
      </div>
    );
  }
  return (
    <div>
      {label}
      <Input id={fieldId} aria-required={field.required} aria-invalid={invalid} aria-describedby={field.help ? `${fieldId}-help` : undefined} value={value as string} onChange={(e) => onChange(e.target.value)} placeholder={field.placeholder ? L(field.placeholder) : undefined} className={cn('mt-1', invalid && 'border-red-500')} />
      {help}
    </div>
  );
}
