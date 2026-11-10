'use client';

import { useId, useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import {
  quantumVariationalFields, quantumVariationalKinds, quantumVariationalNewSpace,
  quantumVariationalPresets, quantumVariationalPresetInput, quantumVariationalInputKey,
  editQuantumVariationalField, type QuantumVariationalKind, type QuantumVariationalInput,
} from '@/lib/learning-quantum-variational';
import { parseQuantumVariationalInputDraft } from '@/lib/learning-quantum-variational-protocol';
import { quantumVariationalSteps } from '@/lib/learning-quantum-variational-concepts';
import { useSavedQuantumVariationalInput } from './use-saved-quantum-variational-input';
import { useQuantumVariationalCalculation } from './use-quantum-variational-calculation';
import { QuantumVariationalResultView, QuantumVariationalButton as Button } from './quantum-variational-view';

function Field({ input, field, apply, language }: {
  input: QuantumVariationalInput; field: ReturnType<typeof quantumVariationalFields>[number];
  apply: (input: QuantumVariationalInput) => void; language: 'ko' | 'en';
}) {
  const [raw, setRaw] = useState(String(field.value));
  const [invalid, setInvalid] = useState(false);
  const id = useId();
  function commit() {
    const next = editQuantumVariationalField(input, field.id, raw);
    setInvalid(next === null);
    if (next) apply(next);
  }
  return <div style={{ minWidth: 0 }}>
    <label htmlFor={id} style={{ display: 'block', overflowWrap: 'anywhere' }}>{field.label[language]}</label>
    <input id={id} data-qv-field={field.id} value={raw} inputMode="numeric" aria-invalid={invalid}
      aria-describedby={invalid ? `${id}-error` : undefined} onChange={e => { setRaw(e.target.value); setInvalid(false); }}
      onBlur={commit} onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } }}
      style={{ width: '100%', padding: 8, background: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', borderRadius: 'var(--radius)' }} />
    {invalid && <p id={`${id}-error`} role="alert">{language === 'ko' ? '조건에 맞는 정수가 필요합니다. 마지막 유효 입력을 유지합니다.' : 'A valid integer satisfying the input constraints is required. The last valid input is retained.'}</p>}
  </div>;
}

function Inner({ kind }: { kind: QuantumVariationalKind }) {
  const { language } = useLanguage();
  const t = (ko: string, en: string) => language === 'ko' ? ko : en;
  const saved = useSavedQuantumVariationalInput(kind);
  const key = quantumVariationalInputKey(saved.input);
  const [stopped, setStopped] = useState<string | null>(null);
  const calculation = useQuantumVariationalCalculation(saved.input, saved.ready && stopped !== key);
  const [dimension, setDimension] = useState(3), [columns, setColumns] = useState(2);
  const [raw, setRaw] = useState<string | null>(null), [invalid, setInvalid] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const formId = useId();
  const ritz = kind === quantumVariationalKinds[1];
  const selectStyle = { background: 'hsl(var(--background))', color: 'hsl(var(--foreground))', padding: 8, border: '1px solid hsl(var(--border))', borderRadius: 'var(--radius)' };
  function apply(next: QuantumVariationalInput) {
    saved.setInput(next); setRaw(null); setInvalid(false); setStopped(null); setEpoch(value => value + 1);
  }
  const fields = quantumVariationalFields(saved.input);
  return <section className="ai-lab" id="experiment" aria-labelledby={`${formId}-title`} data-qv-lab={kind} data-qv-input={key} data-qv-state={calculation.status} data-qv-phase={calculation.phase ?? ''}>
    <div className="ai-lab-heading"><h2 id={`${formId}-title`}>{t('직접 실험하기', 'Experiment with the model')}</h2></div>
    <p>{t('예제를 고른 뒤 한 항목씩 바꾸고 결과를 예측하세요. H의 상삼각 원소가 Hermitian 행렬을 정합니다. 입력은 절댓값 1,000,000 이하의 정수이며 분모는 양수입니다. 비활성 차원의 항목은 0이어야 합니다.', 'Choose an example, predict the outcome, then change one entry. The upper-triangular entries of H define the Hermitian matrix. Inputs are integers with absolute value at most 1,000,000 and positive denominators. Entries outside the active dimensions must be zero.')}</p>
    <div className="ai-lab-actions" role="group" aria-label={t('실험 예제', 'Experiment examples')} style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
      {quantumVariationalPresets[kind].map((p, i) => <Button key={p.id} variant="outline" disabled={!saved.ready}
        aria-pressed={key === quantumVariationalInputKey(p.input)} onClick={() => apply(quantumVariationalPresetInput(kind, i))}>{p.label[language]}</Button>)}
    </div>
    <details className="ai-guided-lab"><summary>{t('안내 실험 세 가지', 'Three guided experiments')}</summary>
      {quantumVariationalSteps[kind].map((step, i) => <article key={i} data-qv-guide={i}>
        <h3>{step.title[language]}</h3><p>{step.predict[language]}</p><p>{step.action[language]}</p>
        <div className="ai-lab-actions">{step.presets.map(index => <Button key={index} variant="outline" disabled={!saved.ready} onClick={() => apply(quantumVariationalPresetInput(kind, index))}>{quantumVariationalPresets[kind][index].label[language]}</Button>)}</div>
      </article>)}
    </details>
    {!saved.storage && <p role="status">{t('브라우저 저장을 사용할 수 없습니다. 현재 화면에서는 계속 실험할 수 있습니다.', 'Browser storage is unavailable. You can continue experimenting in this session.')}</p>}
    <details><summary>{t('차원을 정해 새 공간 만들기', 'Build a new space with chosen dimensions')}</summary>
      <p>{t('적용하면 기존 입력을 대각 H=(0,…,n−1), 첫 좌표 상태, 표준 기저로 교체합니다. 기존 공간을 잘라내지 않습니다.', 'Apply replaces the input with diagonal H=(0,…,n−1), the first coordinate state and a canonical basis. It does not truncate the existing space.')}</p>
      <label htmlFor={`${formId}-n`}>{t('전체 차원', 'Ambient dimension')}</label>{' '}
      <select style={selectStyle} id={`${formId}-n`} value={dimension} onChange={e => { const n = Number(e.target.value); setDimension(n); setColumns(k => Math.min(k, n)); }}>{[1, 2, 3, 4].map(n => <option key={n}>{n}</option>)}</select>{' '}
      {ritz && <><label htmlFor={`${formId}-k`}>{t('시험 공간 차원', 'Trial-space dimension')}</label>{' '}<select style={selectStyle} id={`${formId}-k`} value={columns} onChange={e => setColumns(Number(e.target.value))}>{Array.from({ length: dimension }, (_, i) => <option key={i}>{i + 1}</option>)}</select>{' '}</>}
      <Button variant="outline" disabled={!saved.ready} onClick={() => apply(quantumVariationalNewSpace(kind, dimension, columns))}>{t('새 공간 적용', 'Apply new space')}</Button>
    </details>
    <details><summary>{t('모든 정수 매개변수 편집', 'Edit all integer parameters')}</summary>
      <p>{ritz ? t('B는 열 독립이어야 하고 계수는 영벡터가 아니어야 합니다. 여러 항목을 동시에 바꿀 때는 JSON 입력을 사용하세요.', 'B must have independent columns and coefficients must be nonzero. Use JSON to change several entries atomically.') : t('현재 상태는 영벡터일 수 없습니다. 경로 방향은 영벡터여도 됩니다.', 'The current state must be nonzero. The path direction may be zero.')}</p>
      <fieldset disabled={!saved.ready} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 12 }}>
        <legend>{t('정수 입력', 'Integer inputs')}</legend>
        {fields.map(field => <Field key={`${key}:${epoch}:${field.id}`} input={saved.input} field={field} apply={apply} language={language} />)}
      </fieldset>
    </details>
    <details><summary>{t('전체 입력 JSON', 'Complete input JSON')}</summary>
      <label htmlFor={`${formId}-json`}>{t('모든 항목을 함께 편집', 'Edit every field together')}</label>
      <textarea id={`${formId}-json`} data-qv-json value={raw ?? JSON.stringify(saved.input, null, 2)} aria-invalid={invalid}
        onChange={e => { setRaw(e.target.value); setInvalid(false); }} rows={12} spellCheck={false}
        style={{ display: 'block', width: '100%', background: 'hsl(var(--background))', border: '1px solid hsl(var(--border))' }} />
      <Button variant="outline" disabled={!saved.ready} onClick={() => { const next = parseQuantumVariationalInputDraft(raw ?? JSON.stringify(saved.input)); if (!next || next.kind !== kind) setInvalid(true); else apply(next); }}>{t('JSON 적용', 'Apply JSON')}</Button>
      {invalid && <p role="alert">{t('JSON 형식, 교재 종류, 차원·분모·상태·기저 조건을 확인하세요. 마지막 유효 입력을 유지합니다.', 'Check JSON structure, model kind, dimensions, denominators, state and basis constraints. The last valid input is retained.')}</p>}
    </details>
    <div className="ai-lab-actions">
      <Button variant="outline" disabled={!saved.ready} onClick={() => apply(quantumVariationalPresetInput(kind, 0))}>{t('초기 예제로 복원', 'Reset to first example')}</Button>
      {calculation.status === 'pending' && <Button variant="outline" onClick={() => { setStopped(key); calculation.cancel(); }}>{t('계산 취소', 'Cancel calculation')}</Button>}
      {(stopped === key || calculation.status === 'error') && <Button variant="outline" onClick={() => { setStopped(null); calculation.retry(); }}>{t('계산 다시 시도', 'Retry calculation')}</Button>}
    </div>
    <p role="status">{!saved.ready ? t('저장된 입력 확인 중', 'Restoring input') : stopped === key ? t('계산을 취소했습니다.', 'Calculation cancelled.') : calculation.status === 'pending' ? t('정확·구간 계산 중입니다. 큰 사례는 시간이 걸릴 수 있습니다.', 'Exact and interval calculation in progress. Large cases can take time.') : calculation.status === 'error' ? t('계산을 완료하지 못했습니다. 다시 시도하세요.', 'Calculation could not complete. Please retry.') : calculation.status === 'ready' ? t('현재 입력의 계산을 완료했습니다.', 'Calculation completed for the current input.') : ''}</p>
    {calculation.response && <QuantumVariationalResultView response={calculation.response} language={language} contextLabel={t('자유 실험', 'Free experiment')} />}
  </section>;
}

export default function QuantumVariationalLab({ kind }: { kind: QuantumVariationalKind }) {
  return <Inner key={kind} kind={kind} />;
}
