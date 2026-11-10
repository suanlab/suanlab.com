'use client';
import { useEffect, useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { qubitTeaching } from '@/data/qubit-teaching';
import { qubitTeachingPresets, qubitTeachingInput, qubitExperimentPresets, parseQubitTeachingDraft, type QubitTeachingId } from '@/lib/learning-qubit-teaching-input';
import { qubitInputKey } from '@/lib/learning-qubit-input';
import { useSavedQubitInput } from './use-saved-qubit-input';
import { useQubitCalculation } from './use-qubit-calculation';
import QubitResult from './qubit-result';

export default function QubitLab(props: { kind: QubitTeachingId; language: 'ko' | 'en' }) {
  return <QubitLabSession key={props.kind} {...props} />;
}
function QubitLabSession({ kind, language }: { kind: QubitTeachingId; language: 'ko' | 'en' }) {
  const t = (ko: string, en: string) => language === 'ko' ? ko : en;
  const saved = useSavedQubitInput(kind), [running, setRunning] = useState(true);
  const calculation = useQubitCalculation(saved.input, saved.ready && running);
  const [draft, setDraft] = useState(() => JSON.stringify(saved.input, null, 2));
  const [invalidDraft, setInvalidDraft] = useState(false), [experiment, setExperiment] = useState(0), id = useId();
  const presets = qubitTeachingPresets[kind], guide = qubitTeaching.find(item => item.unit.id === kind)!.guide;
  const key = qubitInputKey(saved.input), presetIndex = presets.findIndex(p => qubitInputKey(p.input) === key);
  useEffect(() => { setDraft(JSON.stringify(saved.input, null, 2)); setInvalidDraft(false); }, [saved.input]);
  const choose = (index: number) => { saved.setInput(qubitTeachingInput(kind, index)); setRunning(true); };
  const error = calculation.error;
  const errorMessage = error === 'invalid-input' ? t('입력 상태나 연산의 수학 조건이 맞지 않습니다. 영벡터·음의 확률·밀도행렬 조건을 확인하거나 예제를 선택하세요.', 'The state or operation does not satisfy its mathematical input conditions. Check for a zero vector, negative probabilities or invalid density matrix, or select an example.')
    : error === 'unavailable' || error === 'worker-failed' ? t('계산을 시작하거나 계속할 수 없습니다. 다시 시도하거나 다른 예제를 선택하세요.', 'The calculation could not start or continue. Retry or select another example.')
      : error === 'timeout' || error === 'resource-limit' ? t('이 입력의 계산이 실습의 시간·크기 범위를 넘었습니다. 더 작은 예제로 바꾸거나 다시 시도하세요.', 'This calculation exceeded the lab time or size budget. Try a smaller example or retry.')
        : t('계산값 검산을 통과하지 못했습니다. 결과를 표시하지 않았습니다. 다시 시도하세요.', 'The calculated values failed independent checks. No result was displayed. Please retry.');
  return <section className="ai-qubit-lab" id="qubit-experiment" aria-labelledby={`${id}-title`} data-qubit-lab={kind} data-qubit-phase={calculation.status}>
    <h2 id={`${id}-title`}>{t('값을 바꾸고 상태를 비교하기', 'Change the input and compare states')}</h2>
    <p>{t('예제를 선택해 먼저 결과를 예측하세요. 그림은 현재 입력의 계산값이며, 표시 구간의 중점을 사용합니다. 매우 작은 비영 값을 정확한0으로 단정하지 말고 정확 값과 구간을 확인하세요. 표가 넓으면 표에 초점을 두고 방향키로 이동하거나 좌우로 밀어 보세요.', 'Choose an example and predict the result first. Diagrams use interval midpoints for the current input. Do not interpret a tiny nonzero value as exact zero: inspect its exact value and interval. For wide tables, focus the table and use arrow keys or swipe horizontally.')}</p>
    {!saved.ready && <p role="status">{t('이전 입력을 확인하고 있습니다.', 'Checking the saved input.')}</p>}
    {saved.storage !== 'available' && <p role="status">{saved.storage === 'corrupt' ? t('저장된 입력을 읽을 수 없어 첫 예제로 복원했습니다.', 'The saved input was invalid; the first example was restored.') : t('이 브라우저에서 입력 저장을 사용할 수 없습니다. 현재 화면에서는 계속 실험할 수 있습니다.', 'Input storage is unavailable in this browser. You can continue experimenting in this session.')}</p>}
    <label className="qubit-field" htmlFor={`${id}-experiment`}>{t('안내 실험', 'Guided experiment')}<select id={`${id}-experiment`} aria-label={t('안내 실험', 'Guided experiment')} value={experiment} onChange={e => setExperiment(Number(e.target.value))}>{guide.experiments.map((item, i) => <option key={i} value={i}>{i + 1}. {item.title[language]}</option>)}</select></label>
    <p><strong>{t('예측', 'Predict')}: </strong>{guide.experiments[experiment].predict[language]}</p><p>{guide.experiments[experiment].action[language]}</p>
    <div className="qubit-actions" role="group" aria-label={t('비교할 예제', 'Examples to compare')}>{qubitExperimentPresets[kind][experiment].map(index => <Button key={index} variant="outline" aria-pressed={presetIndex === index} onClick={() => choose(index)}>{presets[index].label[language]}</Button>)}</div>
    <label className="qubit-field" htmlFor={`${id}-preset`}>{t('전체 입력 예제', 'All input examples')}<select id={`${id}-preset`} aria-label={t('전체 입력 예제', 'All input examples')} value={presetIndex < 0 ? 'custom' : String(presetIndex)} disabled={!saved.ready} onChange={e => { if (e.target.value !== 'custom') choose(Number(e.target.value)); }}>
      <option value="custom" disabled>{t('직접 작성한 입력', 'Custom input')}</option>{presets.map((p, i) => <option key={p.id} value={i}>{p.label[language]}</option>)}
    </select></label>
    <div className="qubit-actions"><Button variant="outline" onClick={() => { choose(0); calculation.retry(); }}>{t('첫 예제로 초기화', 'Reset to first example')}</Button>
      {calculation.status === 'pending' && <Button variant="outline" onClick={() => { setRunning(false); calculation.cancel(); }}>{t('계산 취소', 'Cancel calculation')}</Button>}
      {(calculation.status === 'error' || !running) && <Button onClick={() => { setRunning(true); calculation.retry(); }}>{t('다시 계산', 'Recalculate')}</Button>}
    </div>
    <details><summary>{t('직접 입력 · 정확한 상태와 연산 수정', 'Custom input · edit exact states and operations')}</summary>
      <p>{t('각 복소수는 real과 imaginary의 두 계수 [a,b]로 a+b√2를 나타냅니다. 분수는 "3/5"처럼 문자열로 적습니다. 회로의 sites·gates·input, 채널의 map·state, 환경의 parameter·state, 측정의 outcomes·state·correctZ를 수정할 수 있습니다. 적용 전 입력은 현재 결과에 영향을 주지 않습니다.', 'Each complex number uses two coefficients [a,b] in real and imaginary to represent a+b√2. Write fractions as strings such as "3/5". Edit sites/gates/input for circuits, map/state for channels, parameter/state for environments, or outcomes/state/correctZ for measurements. Draft edits do not affect results until applied.')}</p>
      <label className="qubit-field" htmlFor={`${id}-draft`}>{t('정확 입력 JSON', 'Exact input JSON')}<textarea id={`${id}-draft`} aria-label={t('정확 입력 JSON', 'Exact input JSON')} value={draft} maxLength={65536} spellCheck={false} aria-invalid={invalidDraft} aria-describedby={invalidDraft ? `${id}-draft-error` : undefined} onChange={e => { setDraft(e.target.value); setInvalidDraft(false); }} /></label>
      {invalidDraft && <p role="alert" id={`${id}-draft-error`}>{t('이 단원에서 지원하는 입력 형식을 확인하세요. JSON·분수 문자열·행렬 크기·게이트 수를 확인하세요. 현재 입력은 유지했습니다.', 'Check the input format supported by this lesson: JSON, fraction strings, matrix sizes and gate count. The current input was retained.')}</p>}
      <div className="qubit-actions"><Button onClick={() => { const next = parseQubitTeachingDraft(kind, draft); if (next) { saved.setInput(next); setRunning(true); calculation.retry(); setInvalidDraft(false); } else setInvalidDraft(true); }}>{t('입력 적용', 'Apply input')}</Button><Button variant="outline" onClick={() => { setDraft(JSON.stringify(saved.input, null, 2)); setInvalidDraft(false); }}>{t('현재 입력으로 되돌리기', 'Restore current input')}</Button></div>
    </details>
    <div className="qubit-status" role={error ? 'alert' : 'status'} aria-live="polite" data-qubit-calculation-status>
      {error ? errorMessage : !running ? t('계산을 취소했습니다. 다시 계산하거나 다른 예제를 선택하세요.', 'Calculation cancelled. Recalculate or select another example.') : calculation.status === 'ready' ? t('현재 입력의 계산과 독립 검산을 마쳤습니다.', 'Calculation and independent checks for the current input are complete.') : calculation.status === 'pending' ? calculation.phase === 'validating' ? t('결과를 독립적으로 검산하고 있습니다.', 'Independently checking the result.') : t('현재 입력을 계산하고 있습니다.', 'Calculating the current input.') : t('계산을 준비하고 있습니다.', 'Preparing the calculation.')}
    </div>
    {calculation.response && <QubitResult key={key} response={calculation.response} language={language} />}
  </section>;
}
