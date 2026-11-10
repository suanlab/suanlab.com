'use client';
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { useSiteMotion } from '@/components/motion/motion-provider';
import { useSceneActivity } from '@/components/motion/use-scene-activity';
import { ensembleGuides, ensemblePresets, ensemblePresetInput, ensembleSavedValidators, ensembleTitles, validEnsembleSavedInput, type EnsembleKind } from '@/lib/learning-ensembles';
import { ensembleInputKey, snapshotEnsembleInput, type EnsembleInput } from '@/lib/learning-ensemble-protocol';
import { NumberField } from './lab-controls';
import { parseNumberJsonDraft } from '@/lib/learning-json-number-input';
import { useSavedState } from './use-saved-state';
import { useEnsembleCalculation } from './use-ensemble-calculation';
import { EnsembleResultView } from './ensemble-view';
type Text = (ko: string, en: string) => string;
function Parameters({ input, load, t }: { input: EnsembleInput; load: (input: EnsembleInput) => void; t: Text }) {
  if (input.kind === 'finite-binary-reservoir-canonical-comparison') {
    const p = input.parameters;
    const resize = (n: number, b: number) => load({ ...input, parameters: { subsystemSites: n, reservoirSites: b, totalExcitations: Math.min(p.totalExcitations, n + b) } });
    return <div className="ai-two-fields">
      <NumberField resetKey={input} label={t('부분계 자리 수 n', 'Subsystem sites n')} min={0} max={4096 - p.reservoirSites} integer step={1} value={p.subsystemSites} onChange={n => resize(n, p.reservoirSites)} />
      <NumberField resetKey={input} label={t('저장고 자리 수 b', 'Reservoir sites b')} min={0} max={4096 - p.subsystemSites} integer step={1} value={p.reservoirSites} onChange={b => resize(p.subsystemSites, b)} />
      <NumberField resetKey={input} label={t('전체 여기 수 K', 'Total excitations K')} min={0} max={p.subsystemSites + p.reservoirSites} integer step={1} value={p.totalExcitations} onChange={n => load({ ...input, parameters: { ...p, totalExcitations: n } })} />
    </div>;
  }
  const p = input.parameters;
  const change = (key: 'sites' | 'energyGap' | 'temperature', value: number) => {
    if (input.kind === 'canonical-binary-energy-fluctuations') load({ ...input, parameters: { ...input.parameters, [key]: value } });
    else load({ ...input, parameters: { ...input.parameters, [key]: value } });
  };
  return <><div className="ai-two-fields">
    <NumberField resetKey={input} label={t('이름 붙은 자리 수', 'Labeled site count')} min={0} max={4096} integer step={1} value={p.sites} onChange={n => change('sites', n)} />
    <NumberField resetKey={input} label={t('에너지 간격 ε (J)', 'Energy gap ε (J)')} min={0} max={Number.MAX_VALUE} step="any" value={p.energyGap} onChange={n => change('energyGap', n)} />
    <NumberField resetKey={input} label={t('온도 T (K)', 'Temperature T (K)')} min={-Number.MAX_VALUE} max={Number.MAX_VALUE} step="any" value={p.temperature} onChange={n => change('temperature', n)} />
    {input.kind === 'grand-canonical-finite-occupancy-fluctuations' && <NumberField resetKey={input} label={t('화학 퍼텐셜 μ (J)', 'Chemical potential μ (J)')} min={-Number.MAX_VALUE} max={Number.MAX_VALUE} step="any" value={input.parameters.chemicalPotential} onChange={n => load({ ...input, parameters: { ...input.parameters, chemicalPotential: n } })} />}
  </div><p className="ai-small">{t('T=0은 고정된 에너지·화학 퍼텐셜에서 T→0+ 극한입니다. 음의 온도는 이 유한한 유계 모형의 인구역전 준비입니다.', 'T=0 means the T→0+ limit at fixed energy and chemical potential. Negative temperature describes population inversion in this finite bounded model.')}</p></>;
}
function Panel({ kind }: { kind: EnsembleKind }) {
  const { language } = useLanguage(), motion = useSiteMotion(), id = useId();
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  const initial = useMemo(() => ensemblePresetInput(kind, 0), [kind]);
  const [input, setInput, ready, storage] = useSavedState(`suanlab-ensemble-${kind}`, initial, ensembleSavedValidators[kind]);
  const [submitted, setSubmitted] = useState(input), [message, setMessage] = useState(false), [draft, setDraft] = useState(JSON.stringify(input.parameters, null, 2));
  const [resetEpoch, setResetEpoch] = useState(0);
  const [stoppedKey, setStoppedKey] = useState<string | null>(null);
  const [guideIndex, setGuideIndex] = useState(0), [frame, setFrame] = useState<number | null>(null), [playing, setPlaying] = useState(false);
  const panel = useRef<HTMLElement>(null), active = useSceneActivity(panel);
  const inputKey = ensembleInputKey(input), submittedKey = ensembleInputKey(submitted), current = inputKey === submittedKey;
  const calculation = useEnsembleCalculation(submitted, ready && current && stoppedKey !== inputKey);
  useEffect(() => { if (!ready || stoppedKey === inputKey) return; const timer = setTimeout(() => setSubmitted(input), 120); return () => clearTimeout(timer); }, [input, inputKey, ready, stoppedKey]);
  useEffect(() => setDraft(JSON.stringify(input.parameters, null, 2)), [input]);
  useEffect(() => { if (!active) setPlaying(false); }, [active]);
  useEffect(() => {
    if (!playing || !active || !ready || !current || calculation.status !== 'ready') return;
    if (frame === null || frame >= ensemblePresets[kind].length - 1) { setPlaying(false); return; }
    const timer = setTimeout(() => { const next = frame + 1; setFrame(next); setStoppedKey(null); setInput(ensemblePresetInput(kind, next)); }, 450);
    return () => clearTimeout(timer);
  }, [playing, active, ready, current, calculation.status, frame, kind, setInput]);
  const load = (value: EnsembleInput) => {
    setResetEpoch(n => n + 1);
    setPlaying(false); setFrame(null);
    if (validEnsembleSavedInput(kind, value)) { setStoppedKey(null); setInput(snapshotEnsembleInput(value)); setMessage(false); } else setMessage(true);
  };
  const show = (index: number) => { setResetEpoch(n => n + 1); setPlaying(false); setFrame(index); setStoppedKey(null); setInput(ensemblePresetInput(kind, index)); setMessage(false); };
  const guides = ensembleGuides[kind], guide = guides[guideIndex], pending = ready && stoppedKey !== inputKey && (!current || calculation.status === 'pending');
  return <section className="ai-lab" ref={panel} id="experiment" aria-labelledby={`${id}-title`} data-ensemble-lab={kind} data-ensemble-state={!ready ? 'loading' : pending ? 'pending' : calculation.status} data-ensemble-current={ready && current && calculation.response ? inputKey : ''} data-ensemble-playing={playing ? 'true' : 'false'} data-ensemble-frame={frame ?? -1} data-ensemble-ready={ready ? 'true' : 'false'}>
    <div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id={`${id}-title`}>{ensembleTitles[kind][language]}</h2></div><span className="ai-live-dot">{t('상태 비교', 'State comparison')}</span></div>
    <details className="ai-guided-lab" open><summary>{t('안내를 따라 실험하기', 'Follow a guided experiment')}</summary><div className="ai-guided-choices" role="group" aria-label={t('안내 실험 선택', 'Choose a guided experiment')}>{guides.map((g, i) => <Button key={i} variant="outline" type="button" aria-pressed={guideIndex === i} onClick={() => setGuideIndex(i)}>{i + 1}. {g.title[language]}</Button>)}</div>
      <h3>{guide.title[language]}</h3><p><strong>{t('먼저 예측', 'Predict first')} · </strong>{guide.predict[language]}</p><p><strong>{t('조작 방법', 'Try it')} · </strong>{guide.action[language]}</p>
      <Button variant="outline" type="button" disabled={!ready} onClick={() => load(ensemblePresetInput(kind, guide.preset))}>{t('이 예제 열기', 'Load this example')}</Button>
      <details key={guideIndex}><summary>{t('결과 해석 보기', 'Reveal the explanation')}</summary><p>{guide.explain[language]}</p></details>
    </details>
    <fieldset className="ai-presets" disabled={!ready}><legend>{t('비교할 예제', 'Examples to compare')}</legend>{ensemblePresets[kind].map((p, i) => <Button key={i} variant="outline" size="sm" type="button" onClick={() => show(i)}>{p.label[language]}</Button>)}</fieldset>
    <fieldset className="ai-parameters" disabled={!ready}><legend>{t('앙상블 준비 설정', 'Ensemble preparation settings')}</legend><Parameters input={input} load={load} t={t} /></fieldset>
    <details><summary>{t('확장 매개변수', 'Extended parameters')}</summary><p className="ai-small">{t('현재 매개변수를 JSON으로 편집합니다. 전체 또는 개별 모형의 자리 수는 최대4096입니다. 에너지 간격은 비음수이고 온도와 화학 퍼텐셜은 유한해야 합니다.', 'Edit current parameters as JSON. Total or individual model site counts are limited to 4096. Energy gaps must be nonnegative; temperature and chemical potential must be finite.')}</p>
      <label className="ai-number-field"><span>{t('확장 매개변수 JSON', 'Extended parameter JSON')}</span><textarea rows={10} value={draft} onChange={e => setDraft(e.target.value)} style={{ width: '100%', background: 'hsl(var(--background))', color: 'inherit', fontFamily: 'monospace' }} /></label>
      <Button variant="outline" type="button" disabled={!ready} onClick={() => { try { const next = { kind, parameters: parseNumberJsonDraft(draft, ['sites', 'subsystemSites', 'reservoirSites', 'totalExcitations']) }; if (!validEnsembleSavedInput(kind, next)) { setMessage(true); return; } load(next); } catch { setMessage(true); } }}>{t('확장 설정 적용', 'Apply extended settings')}</Button>
    </details>
    {message && <p role="alert">{t('설정을 적용하지 못했습니다. 범위·정수·전체 자리 수를 확인하세요. 매우 작은 비영 값을 0으로 바꾸어 적용하지 않습니다. 이전의 유효 설정은 유지합니다.', 'Settings were not applied. Check ranges, integers and total site counts. A nonzero underflow is not applied as zero. The previous valid settings are retained.')}</p>}
    <div className="ai-lab-actions"><Button variant="outline" type="button" disabled={!ready || !active} aria-pressed={playing} onClick={() => { if (playing) setPlaying(false); else { setStoppedKey(null); if (frame === null || frame === ensemblePresets[kind].length - 1) { setFrame(0); setInput(ensemblePresetInput(kind, 0)); } setPlaying(true); } }}>{playing ? t('예제 순회 일시정지', 'Pause example scan') : t('예제 비교 순회', 'Scan comparison examples')}</Button>
      <Button variant="outline" type="button" disabled={!ready || frame === ensemblePresets[kind].length - 1} onClick={() => show(frame === null ? 0 : frame + 1)}>{t('비교 예제 한 단계', 'Step comparison example')}</Button>
      <Button variant="outline" type="button" disabled={!ready} onClick={() => load(initial)}>{t('모든 실험 설정 초기화', 'Reset all experiment settings')}</Button></div>
    <p className="ai-small">{t('순회는 서로 다른 준비·제약을 비교합니다. 물리 시간·완화 과정이 아닙니다. 각 계산이 끝난 뒤 다음 예제로 갑니다.', 'Scanning compares different preparations and constraints. It is not physical time or relaxation. Each calculation finishes before the next example starts.')} {frame !== null && <output>{frame + 1}/{ensemblePresets[kind].length}</output>}</p>
    {!motion.canAnimate && <p className="ai-small">{t('모션 설정으로 자동 순회를 멈췄습니다. 수동 단계는 사용할 수 있습니다.', 'Motion preferences pause automatic scanning. Manual steps remain available.')}</p>}
    <div role="status" aria-live="polite">{!ready ? t('저장된 설정을 확인하고 있습니다.', 'Checking saved settings.') : pending ? t('계산 중입니다. 다른 설정을 선택하거나 중단할 수 있습니다.', 'Calculating. You can choose another setting or stop.') : calculation.status === 'error' ? t('계산을 완료하지 못했습니다. 다시 시도하거나 다른 예제를 선택하세요.', 'The calculation could not finish. Retry or choose another example.') : calculation.status === 'ready' ? t('현재 설정의 계산을 완료했습니다.', 'The current settings have been calculated.') : t('계산을 중단했습니다. 다시 시도할 수 있습니다.', 'Calculation stopped. You can retry.')}</div>
    <div className="ai-lab-actions">{pending && <Button variant="outline" type="button" onClick={() => { setPlaying(false); setStoppedKey(inputKey); calculation.cancel(); }}>{t('계산 중단', 'Stop calculation')}</Button>}{ready && (stoppedKey === inputKey || current && calculation.status !== 'ready') && !pending && <Button variant="outline" type="button" onClick={() => { setStoppedKey(null); calculation.retry(); }}>{t('다시 계산', 'Retry calculation')}</Button>}</div>
    {ready && current && calculation.response && <EnsembleResultView key={`${calculation.requestId}:${resetEpoch}`} response={calculation.response} language={language} />}
    <p className="ai-small">{storage ? t('설정을 이 브라우저에 저장합니다.', 'Settings are saved in this browser.') : t('브라우저 저장을 사용할 수 없습니다.', 'Browser storage is unavailable.')}</p>
  </section>;
}
export default function EnsembleLab({ kind }: { kind: EnsembleKind }) { return <Panel key={kind} kind={kind} />; }
