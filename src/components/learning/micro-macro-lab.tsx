'use client';
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { useSiteMotion } from '@/components/motion/motion-provider';
import { useSceneActivity } from '@/components/motion/use-scene-activity';
import { microMacroGuides, microMacroPresets, microMacroPresetInput, microMacroSavedValidators, microMacroTitles, validMicroMacroSavedInput, type MicroMacroKind } from '@/lib/learning-micro-macro';
import { microMacroInputKey, snapshotMicroMacroInput, type MicroMacroInput } from '@/lib/learning-micro-macro-protocol';
import { NumberField } from './lab-controls';
import { parseNumberJsonDraft } from '@/lib/learning-json-number-input';
import { useSavedState } from './use-saved-state';
import { useMicroMacroCalculation } from './use-micro-macro-calculation';
import { MicroMacroResultView } from './micro-macro-view';
type Text = (ko: string, en: string) => string;
function Parameters({ input, load, t }: { input: MicroMacroInput; load: (input: MicroMacroInput) => void; t: Text }) {
  if (input.kind === 'binary-shell-multiplicity-block-counts') {
    const p = input.parameters;
    return <div className="ai-two-fields"><NumberField resetKey={input} label="N" min={0} max={4096} integer step={1} value={p.sites} onChange={n => load({ ...input, parameters: { ...p, sites: n, excitations: Math.min(p.excitations, n), blockASites: Math.min(p.blockASites, n) } })} />
      <NumberField resetKey={input} label="K" min={0} max={p.sites} integer step={1} value={p.excitations} onChange={n => load({ ...input, parameters: { ...p, excitations: n } })} />
      <NumberField resetKey={input} label={t('블록 A 자리 수', 'Block A sites')} min={0} max={p.sites} integer step={1} value={p.blockASites} onChange={n => load({ ...input, parameters: { ...p, blockASites: n } })} />
      <NumberField resetKey={input} label={t('에너지 간격 ε (J)', 'Energy gap ε (J)')} min={Number.MIN_VALUE} max={Number.MAX_VALUE} step="any" value={p.energyGap} onChange={n => load({ ...input, parameters: { ...p, energyGap: n } })} /></div>;
  }
  if (input.kind === 'finite-macro-partition-conditional-entropy') {
    const p = input.parameters;
    return <><NumberField resetKey={input} label={t('선언된 군 수', 'Declared group count')} min={1} max={32} integer step={1} value={p.groupCount} onChange={n => load({ ...input, parameters: { ...p, groupCount: n, mapping: p.mapping.map(g => Math.min(g, n - 1)) } })} />
      {p.trialWeights.map((w, i) => <fieldset key={i}><legend>{t('미시상태', 'Microstate')} X={i}</legend><div className="ai-two-fields"><NumberField resetKey={input} label={`w[${i}]`} min={0} max={Number.MAX_VALUE} step="any" value={w} onChange={n => load({ ...input, parameters: { ...p, trialWeights: p.trialWeights.map((v, j) => j === i ? n : v) } })} />
        <NumberField resetKey={input} label={`v[${i}]`} min={0} max={Number.MAX_VALUE} step="any" value={p.targetWeights[i]} onChange={n => load({ ...input, parameters: { ...p, targetWeights: p.targetWeights.map((v, j) => j === i ? n : v) } })} /></div>
        <label className="ai-number-field"><span>g({i})</span><select style={{ width: '100%', background: 'hsl(var(--background))', color: 'inherit', padding: '.5rem', border: '1px solid hsl(var(--input))', borderRadius: 'calc(var(--radius) - 2px)' }} value={p.mapping[i]} onChange={e => load({ ...input, parameters: { ...p, mapping: p.mapping.map((v, j) => j === i ? Number(e.target.value) : v) } })}>{Array.from({ length: p.groupCount }, (_, g) => <option key={g} value={g}>M={g}</option>)}</select></label></fieldset>)}</>;
  }
  const p = input.parameters;
  return <><div className="ai-two-fields">{p.preparationWeights.map((w, i) => <NumberField resetKey={input} key={i} label={`${t('준비 가중치', 'Preparation weight')} w[${i}]`} min={0} max={Number.MAX_VALUE} step="any" value={w} onChange={n => load({ ...input, parameters: { ...p, preparationWeights: p.preparationWeights.map((v, j) => j === i ? n : v) } })} />)}</div>
    <details><summary>{t('장치의 행 가중치 수정', 'Edit instrument row weights')}</summary><p className="ai-small">{t('각 행을 따로 정규화합니다. 준비 확률0인 입력에도 양의 총합을 가진 행이 필요합니다.', 'Each row is normalized separately. Even a preparation-zero input needs a positive-total row.')}</p>
      <div className="ai-discrete-table" role="region" tabIndex={0} aria-label={t('채널 가중치 입력', 'Channel weight inputs')}><table><thead><tr><th scope="col">X</th>{p.channelWeights[0].map((_, j) => <th scope="col" key={j}>Y={j}</th>)}</tr></thead><tbody>{p.channelWeights.map((row, i) => <tr key={i}><th scope="row">{i}</th>{row.map((w, j) => <td key={j} style={{ minWidth: '8rem' }}><NumberField resetKey={input} label={`a[${i},${j}]`} min={0} max={Number.MAX_VALUE} step="any" value={w} onChange={n => load({ ...input, parameters: { ...p, channelWeights: p.channelWeights.map((r, a) => a === i ? r.map((v, b) => b === j ? n : v) : r) } })} /></td>)}</tr>)}</tbody></table></div>
    </details></>;
}
function Panel({ kind }: { kind: MicroMacroKind }) {
  const { language } = useLanguage(), motion = useSiteMotion(), id = useId();
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  const initial = useMemo(() => microMacroPresetInput(kind, 0), [kind]);
  const [input, setInput, ready, storage] = useSavedState(`suanlab-micro-macro-${kind}`, initial, microMacroSavedValidators[kind]);
  const [submitted, setSubmitted] = useState(input), [message, setMessage] = useState(false), [draft, setDraft] = useState(JSON.stringify(input.parameters, null, 2));
  const [stoppedKey, setStoppedKey] = useState<string | null>(null);
  const [guideIndex, setGuideIndex] = useState(0), [frame, setFrame] = useState<number | null>(null), [playing, setPlaying] = useState(false);
  const panel = useRef<HTMLElement>(null), active = useSceneActivity(panel);
  const inputKey = microMacroInputKey(input), submittedKey = microMacroInputKey(submitted), current = inputKey === submittedKey;
  const calculation = useMicroMacroCalculation(submitted, ready && current && stoppedKey !== inputKey);
  useEffect(() => { if (!ready || stoppedKey === inputKey) return; const timer = setTimeout(() => setSubmitted(input), 120); return () => clearTimeout(timer); }, [input, inputKey, ready, stoppedKey]);
  useEffect(() => setDraft(JSON.stringify(input.parameters, null, 2)), [input]);
  useEffect(() => { if (!active) setPlaying(false); }, [active]);
  useEffect(() => {
    if (!playing || !active || !ready || !current || calculation.status !== 'ready') return;
    if (frame === null || frame >= microMacroPresets[kind].length - 1) { setPlaying(false); return; }
    const timer = setTimeout(() => { const next = frame + 1; setFrame(next); setStoppedKey(null); setInput(microMacroPresetInput(kind, next)); }, 450);
    return () => clearTimeout(timer);
  }, [playing, active, ready, current, calculation.status, frame, kind, setInput]);
  const load = (value: MicroMacroInput) => {
    setPlaying(false); setFrame(null);
    if (validMicroMacroSavedInput(kind, value)) { setStoppedKey(null); setInput(snapshotMicroMacroInput(value)); setMessage(false); } else setMessage(true);
  };
  const show = (index: number) => { setPlaying(false); setFrame(index); setStoppedKey(null); setInput(microMacroPresetInput(kind, index)); setMessage(false); };
  const guides = microMacroGuides[kind], guide = guides[guideIndex], pending = ready && stoppedKey !== inputKey && (!current || calculation.status === 'pending');
  return <section className="ai-lab" ref={panel} id="experiment" aria-labelledby={`${id}-title`} data-micro-macro-lab={kind} data-micro-macro-state={!ready ? 'loading' : pending ? 'pending' : calculation.status} data-micro-macro-current={ready && current && calculation.response ? inputKey : ''} data-micro-macro-playing={playing ? 'true' : 'false'} data-micro-macro-frame={frame ?? -1} data-micro-macro-ready={ready ? 'true' : 'false'}>
    <div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id={`${id}-title`}>{microMacroTitles[kind][language]}</h2></div><span className="ai-live-dot">{t('상태 비교', 'State comparison')}</span></div>
    <details className="ai-guided-lab" open><summary>{t('안내를 따라 실험하기', 'Follow a guided experiment')}</summary><div className="ai-guided-choices" role="group" aria-label={t('안내 실험 선택', 'Choose a guided experiment')}>{guides.map((g, i) => <Button key={i} variant="outline" type="button" aria-pressed={guideIndex === i} onClick={() => setGuideIndex(i)}>{i + 1}. {g.title[language]}</Button>)}</div>
      <h3>{guide.title[language]}</h3><p><strong>{t('먼저 예측', 'Predict first')} · </strong>{guide.predict[language]}</p><p><strong>{t('조작 방법', 'Try it')} · </strong>{guide.action[language]}</p>
      <Button variant="outline" type="button" disabled={!ready} onClick={() => load(microMacroPresetInput(kind, guide.preset))}>{t('이 예제 열기', 'Load this example')}</Button>
      <details key={guideIndex}><summary>{t('결과 해석 보기', 'Reveal the explanation')}</summary><p>{guide.explain[language]}</p></details>
    </details>
    <fieldset className="ai-presets" disabled={!ready}><legend>{t('비교할 예제', 'Examples to compare')}</legend>{microMacroPresets[kind].map((p, i) => <Button key={i} variant="outline" size="sm" type="button" onClick={() => show(i)}>{p.label[language]}</Button>)}</fieldset>
    <fieldset className="ai-parameters" disabled={!ready}><legend>{t('준비·관측 설정', 'Preparation and observation settings')}</legend><Parameters input={input} load={load} t={t} /></fieldset>
    <details><summary>{t('확장 설정: 상태 수·행·분할', 'Extended settings: states, rows and partitions')}</summary><p className="ai-small">{t('현재 매개변수를 JSON으로 편집합니다. 이진4096자리,가중32상태·32군,채널16입력·16출력까지 지원합니다. 가중치는 비음수이며 각 분포/행의 합은 양수여야 합니다.', 'Edit the current parameters as JSON. Supports 4096 binary sites, 32 weighted states/groups, and 16 channel inputs/outputs. Weights must be nonnegative with positive totals for each distribution/row.')}</p>
      <label className="ai-number-field"><span>{t('확장 매개변수 JSON', 'Extended parameter JSON')}</span><textarea rows={10} value={draft} onChange={e => setDraft(e.target.value)} style={{ width: '100%', background: 'hsl(var(--background))', color: 'inherit', fontFamily: 'monospace' }} /></label>
      <Button variant="outline" type="button" disabled={!ready} onClick={() => { try { const next = { kind, parameters: parseNumberJsonDraft(draft, ['sites', 'excitations', 'blockASites', 'groupCount', 'mapping']) }; if (!validMicroMacroSavedInput(kind, next)) { setMessage(true); return; } load(next); } catch { setMessage(true); } }}>{t('확장 설정 적용', 'Apply extended settings')}</Button>
    </details>
    {message && <p role="alert">{t('설정을 적용하지 못했습니다. 범위·정수·양의 가중치 합을 확인하세요. 매우 작은 비영 값을 0으로 바꾸어 적용하지 않습니다. 이전의 유효 설정은 유지합니다.', 'Settings were not applied. Check ranges, integers and positive weight totals. A nonzero underflow is not applied as zero. The previous valid settings are retained.')}</p>}
    <div className="ai-lab-actions"><Button variant="outline" type="button" disabled={!ready || !active} aria-pressed={playing} onClick={() => { if (playing) setPlaying(false); else { setStoppedKey(null); if (frame === null || frame === microMacroPresets[kind].length - 1) { setFrame(0); setInput(microMacroPresetInput(kind, 0)); } setPlaying(true); } }}>{playing ? t('예제 순회 일시정지', 'Pause example scan') : t('예제 비교 순회', 'Scan comparison examples')}</Button>
      <Button variant="outline" type="button" disabled={!ready || frame === microMacroPresets[kind].length - 1} onClick={() => show(frame === null ? 0 : frame + 1)}>{t('비교 예제 한 단계', 'Step comparison example')}</Button>
      <Button variant="outline" type="button" disabled={!ready} onClick={() => load(initial)}>{t('모든 실험 설정 초기화', 'Reset all experiment settings')}</Button></div>
    <p className="ai-small">{t('순회는 서로 다른 준비·제약·장치를 비교합니다. 물리 시간·완화 과정이 아닙니다. 각 계산이 끝난 뒤 다음 예제로 갑니다.', 'Scanning compares different preparations, constraints and instruments. It is not physical time or relaxation. Each calculation finishes before the next example starts.')} {frame !== null && <output>{frame + 1}/{microMacroPresets[kind].length}</output>}</p>
    {!motion.canAnimate && <p className="ai-small">{t('모션 설정으로 자동 순회를 멈췄습니다. 수동 단계는 사용할 수 있습니다.', 'Motion preferences pause automatic scanning. Manual steps remain available.')}</p>}
    <div role="status" aria-live="polite">{!ready ? t('저장된 설정을 확인하고 있습니다.', 'Checking saved settings.') : pending ? t('계산 중입니다. 다른 설정을 선택하거나 중단할 수 있습니다.', 'Calculating. You can choose another setting or stop.') : calculation.status === 'error' ? t('계산을 완료하지 못했습니다. 다시 시도하거나 다른 예제를 선택하세요.', 'The calculation could not finish. Retry or choose another example.') : calculation.status === 'ready' ? t('현재 설정의 계산을 완료했습니다.', 'The current settings have been calculated.') : t('계산을 중단했습니다. 다시 시도할 수 있습니다.', 'Calculation stopped. You can retry.')}</div>
    <div className="ai-lab-actions">{pending && <Button variant="outline" type="button" onClick={() => { setPlaying(false); setStoppedKey(inputKey); calculation.cancel(); }}>{t('계산 중단', 'Stop calculation')}</Button>}{ready && (stoppedKey === inputKey || current && calculation.status !== 'ready') && !pending && <Button variant="outline" type="button" onClick={() => { setStoppedKey(null); calculation.retry(); }}>{t('다시 계산', 'Retry calculation')}</Button>}</div>
    {ready && current && calculation.response && <MicroMacroResultView key={calculation.requestId} response={calculation.response} language={language} />}
    <p className="ai-small">{storage ? t('설정을 이 브라우저에 저장합니다.', 'Settings are saved in this browser.') : t('브라우저 저장을 사용할 수 없습니다.', 'Browser storage is unavailable.')}</p>
  </section>;
}
export default function MicroMacroLab({ kind }: { kind: MicroMacroKind }) { return <Panel key={kind} kind={kind} />; }
