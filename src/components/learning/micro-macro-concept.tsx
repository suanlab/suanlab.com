'use client';
import { useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { microMacroConceptInput, type MicroMacroKind } from '@/lib/learning-micro-macro';
import { useMicroMacroCalculation } from './use-micro-macro-calculation';
import { MicroMacroConceptResult } from './micro-macro-view';
export function MicroMacroConceptView({ kind, step, language }: { kind: MicroMacroKind; step: number; language: 'ko' | 'en' }) {
  const input = useMemo(() => microMacroConceptInput(kind, step), [kind, step]);
  const calculation = useMicroMacroCalculation(input), ko = language === 'ko';
  return <div data-micro-macro-concept={kind} data-micro-macro-step={step}>
    {calculation.response ? <MicroMacroConceptResult response={calculation.response} step={step} language={language} /> : <p role="status">{calculation.status === 'error' ? (ko ? '고정 예제를 계산하지 못했습니다. 다시 시도할 수 있습니다.' : 'The fixed example could not be calculated. You can retry.') : (ko ? '고정 예제를 계산하고 있습니다.' : 'Calculating the fixed example.')}</p>}
    {calculation.status === 'error' && <Button type="button" variant="outline" onClick={calculation.retry}>{ko ? '다시 계산' : 'Retry calculation'}</Button>}
  </div>;
}
