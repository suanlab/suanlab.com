'use client';
import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { useSiteMotion } from '@/components/motion/motion-provider';
import { useSceneActivity } from '@/components/motion/use-scene-activity';
import { freeEnergyTeaching } from '@/data/interactive-book';
import { canonicalFiniteSpectrum } from '@/lib/learning-canonical-spectrum';
import { evaluateHelmholtzValues, evaluateGasFreeEnergyValues, freeEnergyControls, freeEnergyPresets,
  freeEnergyPlotValue, freeEnergyComparisonValues, validFreeEnergyValues, validHelmholtzValues, validGasFreeEnergyValues, type FreeEnergyKind } from '@/lib/learning-free-energy';
import type { BinaryObservation } from '@/lib/learning-binary-products';
import { NumberField } from './lab-controls';
import { useSavedState } from './use-saved-state';
import { WavePlot, waveNumber as fmt, type Series } from './waves-plot';
type Locale = 'ko' | 'en';
type Text = (ko: string, en: string) => string;
type Row = { label: string; q: BinaryObservation | null; unit: string; additional?: BinaryObservation | null };
function Readout({ rows, t, label }: { rows: Row[]; t: Text; label: string }) {
  const status = (q: BinaryObservation | null) => q === null ? t('계산 미해결', 'Calculation unresolved')
    : 'zeroStatus' in q && q.zeroStatus === 'unresolved-transcendental-cancellation' ? t('근사 상쇄 · 0 판정 미해결', 'Approximate cancellation; zero unresolved')
      : q.sign === 0 ? t('구조적 또는 대수적 0', 'Structural or algebraic zero')
        : q.value === null ? t('비영 · 선형 범위 밖', 'Nonzero; outside linear range') : t('반올림 관측', 'Rounded observation');
  return <div className="ai-readout"><h3>{t('수치와 표현 상태', 'Values and representation status')}</h3>
    <p className="ai-small">{t('—의 의미는 상태 열에서 확인하세요. 값의 null은 비영 값의 표시 범위 문제일 수 있고, 계산 자체의 미해결과 다릅니다. 로그는 표의 단위로 쓴 크기의 자연로그입니다. 추가 로그는 별도 계산이 있는 행의 ln|ln|값||입니다.', 'Read the status beside each dash. A missing linear value can be a nonzero quantity outside the display range, distinct from an unresolved calculation. Logs use magnitudes in the stated unit. The additional log is ln|ln|value|| on rows with a separate logarithmic calculation.')}</p>
    <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={label}><table><thead><tr>{[t('관측', 'Observation'),t('값', 'Value'),t('단위', 'Unit'),t('부호', 'Sign'),t('상태', 'Status'),t('로그 크기', 'Log magnitude'),t('로그 상태', 'Log status'),t('추가 로그', 'Additional log')].map(x => <th key={x} scope="col">{x}</th>)}</tr></thead><tbody>{rows.map(({ label, q, unit, additional }) => <tr key={label}><th scope="row">{label}</th><td>{fmt(q?.value)}</td><td>{unit}</td><td>{q?.sign ?? '—'}</td><td>{status(q)}</td><td>{fmt(q?.logMagnitude)}</td><td>{q === null ? t('미해결', 'Unresolved') : q.sign === 0 ? t('해당 없음', 'Not applicable') : q.logMagnitude === null ? t('범위 밖', 'Out of range') : t('근사', 'Approximate')}</td><td>{fmt(additional?.logMagnitude)}</td></tr>)}</tbody></table></div>
  </div>;
}
function chart(name: string, points: [number, BinaryObservation | null][], color: string): Series {
  const segments: [number, number][][] = []; let segment: [number, number][] = [];
  for (const [x, q] of points) {
    const value = freeEnergyPlotValue(q);
    if (value === null) { if (segment.length) segments.push(segment); segment = []; }
    else segment.push([x, value]);
  }
  if (segment.length) segments.push(segment);
  return { name, color, segments };
}
/** Rescale actual computed ordinates to keep tiny joule values visible and
 * prevent overflowing SVG coordinate arithmetic for very large temperatures. */
function EnergyPlot({ marker, title, series, query, caption, xLabel, domain, t }: { marker: string; title: string; series: Series[];
  query: [number, BinaryObservation | null]; caption: string; xLabel: string; domain: [number, number]; t: Text }) {
  const y = freeEnergyPlotValue(query[1]);
  const scale = Math.max(0, ...series.flatMap(s => s.segments.flat().map(p => Math.abs(p[1]))), y === null ? 0 : Math.abs(y)) || 1;
  const scaled = series.map(s => ({ ...s, segments: s.segments.map(segment => segment.map(([x, y]) => [x, y / scale] as [number, number])) }));
  return <WavePlot marker={marker} title={title} series={scaled} dots={y === null || !Number.isFinite(query[0]) ? [] : [{ point: [query[0], y / scale], color: 'orange' }]}
    xDomain={domain} xLabel={xLabel} yLabel={`J ÷ ${fmt(scale)}`} caption={`${caption} ${t(`세로축에 ${fmt(scale)}을 곱하면 J입니다. 미해결·범위 밖 점은 잇지 않습니다.`, `Multiply the ordinate by ${fmt(scale)} to recover joules. Unresolved and out-of-range points are not connected.`)}`} />;
}
function HelmholtzView({ values, language }: { values: number[]; language: Locale }) {
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  const r = useMemo(() => evaluateHelmholtzValues(values), [values]);
  const history = useMemo(() => Array.from({ length: 33 }, (_, i) => {
    const x = i / 32, v = [...values]; v[2] = 1 - x; v[3] = 1 - x; v[4] = 2 * x;
    return { x, result: evaluateHelmholtzValues(v) };
  }), [values]);
  const temperatures = useMemo(() => Array.from({ length: 33 }, (_, i) => {
    const x = Math.log10(1200) * i / 32;
    return { x, target: canonicalFiniteSpectrum(r.trial.energies, 10 ** x) };
  }), [r.trial.energies]);
  const trial = r.trial, p = trial.perCopy, eq = r.equilibrium.perCopy;
  const queryX = trial.states[2].probability.value;
  const rows: Row[] = [
    { label: 'U', q: p.meanEnergy, unit: 'J' }, { label: 'H', q: p.informationEntropy, unit: 'nat' }, { label: 'S', q: p.entropy, unit: 'J/K' },
    { label: 'TS', q: p.temperatureTimesEntropy, unit: 'J' }, { label: 'F', q: p.helmholtz, unit: 'J' },
    { label: 'Feq', q: eq.helmholtz, unit: 'J' }, { label: 'D(p∥π)', q: r.divergence, unit: 'nat', additional: r.resolution === 'positive-log-tail-resolved' ? r.logDivergence : undefined },
    { label: 'F−Feq', q: r.perCopyGap, unit: 'J' }, { label: 'N(F−Feq)', q: r.totalGap, unit: 'J' },
    { label: 'N F', q: trial.total.helmholtz, unit: 'J' }, { label: 'N Feq', q: r.equilibrium.total.helmholtz, unit: 'J' },
  ];
  trial.states.forEach((s, i) => {
    const target = trial.canonical.states[i];
    rows.push({ label: `p${i}`, q: s.probability, unit: '1' }, { label: `π${i}`, q: target.probability, unit: '1', additional: target.logProbability },
      { label: `E${i}`, q: { value: s.index < 2 ? trial.energies[0] : trial.energies[2], sign: Math.sign(trial.energies[i]), logMagnitude: trial.energies[i] === 0 ? null : Math.log(Math.abs(trial.energies[i])) }, unit: 'J' });
  });
  const resolved = r.resolution === 'structural-equilibrium' ? t('구조적으로 확인한 평형', 'Structurally verified equilibrium')
    : r.resolution === 'unresolved-at-precision-ceiling' ? t('정밀도 한도에서 차이 계산 미해결', 'Gap unresolved at the precision ceiling')
      : r.resolution === 'positive-log-tail-resolved' ? t('로그 꼬리 구간으로 양의 차이 확인', 'Positive gap established by a logarithmic tail enclosure') : t('적응 정밀도 구간으로 양의 차이 확인', 'Positive gap established by an adaptive enclosure');
  return <div data-free-energy-view="helmholtz" lang={language}>
    <p role="status" className="ai-logic-statement">{resolved} · {r.precisionBits} {t('비트', 'bits')}</p>
    <figure className="ai-foundation-figure"><h3>{t('세 미시상태의 확률', 'Probabilities of three microstates')}</h3><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={t('시도 분포와 정준 분포 비교', 'Trial and canonical probability comparison')}>
      <svg viewBox="0 0 350 270" role="img" aria-label={t('파란색 시도 확률과 주황색 정준 확률', 'Blue trial probabilities and orange canonical probabilities')}>
        <path d="M25 30V210H330" stroke="var(--learning-neutral)" fill="none" />
        {trial.states.map((s, i) => <g key={i}>{[{ q: s.probability, dx: 0, color: 'blue' }, { q: trial.canonical.states[i].probability, dx: 24, color: 'orange' }].map(({ q, dx, color }) => q.value === null ? <text key={color} x={42 + 100 * i + dx} y="200" fill="currentColor">—</text> : <rect key={color} x={42 + 100 * i + dx} y={210 - 180 * q.value} width="19" height={180 * q.value} fill={`var(--learning-${color})`} />)}<text x={62 + i * 100} y="233" textAnchor="middle" fontSize="12" fill="currentColor">{t('상태', 'State')} {i}</text><text x={62 + i * 100} y="251" textAnchor="middle" fontSize="10" fill="currentColor">{fmt(s.probability.value)} / {fmt(trial.canonical.states[i].probability.value)}</text></g>)}
      </svg></div><figcaption>{t('각 막대는 미시상태 하나입니다. 상태 0과 1의 에너지는 같아도 시도 확률은 다를 수 있습니다. 파랑: p, 주황: π. 표시 확률 1은 반올림일 수 있습니다.', 'Each pair represents one microstate. States 0 and 1 have equal energies but can have unequal trial probabilities. Blue: p; orange: π. A displayed probability of one can be rounded.')}</figcaption></figure>
    <div className="ai-matrix-panels">
      <EnergyPlot marker="free-energy-spectrum-functional" title={t('에너지와 엔트로피의 경쟁', 'Energy and entropy compete')} t={t} domain={[0, 1]} xLabel="p₂" query={[queryX ?? NaN, p.helmholtz]}
        series={[chart('U', history.map(s => [s.x, s.result.trial.perCopy.meanEnergy]), 'blue'), chart('F', history.map(s => [s.x, s.result.trial.perCopy.helmholtz]), 'green'), chart('Feq', history.map(s => [s.x, s.result.equilibrium.perCopy.helmholtz]), 'neutral')]}
        caption={t('곡선은 바닥 상태 내부를 균등하게 한 비교 분포입니다. 주황 점은 현재 시도 분포로, 내부 불균형이 있으면 곡선 위에 놓이지 않습니다. 시간 축이 아닙니다.', 'Curves compare distributions with uniform ground conditionals. The orange point is the selected trial and can lie off the curve when its conditionals are imbalanced. The axis is not time.')} />
      <EnergyPlot marker="free-energy-spectrum-gap" title={t('직접 계산한 양의 차이', 'Directly computed positive gap')} t={t} domain={[0, 1]} xLabel="p₂" query={[queryX ?? NaN, r.perCopyGap]}
        series={[chart('F−Feq', history.map(s => [s.x, s.result.perCopyGap]), 'blue')]}
        caption={t('동일한 균등 조건부 비교 곡선입니다. 현재 분포의 양의 차이는 반올림한 F−Feq 뺄셈으로 구하지 않습니다.', 'The same uniform-conditional comparison family. The selected positive gap is not obtained by subtracting rounded free energies.')} />
      <WavePlot marker="free-energy-fixed-spectrum-temperature" title={t('에너지를 고정하고 온도 바꾸기', 'Change temperature at fixed energies')}
        series={[chart('π₂', temperatures.map(s => [s.x, s.target.states[2].probability]), 'blue')]} xDomain={[0, Math.log10(1200)]} yDomain={[0, 1]}
        xLabel="log₁₀(T / K)" yLabel="π₂" caption={t('같은 에너지 스펙트럼에서 1–1200 K의 평형을 비교합니다. 현재 T가 범위 밖이어도 표의 계산은 유지됩니다. 실제 완화 경로가 아닙니다.', 'Compare equilibria at 1–1200 K with the same energy spectrum. Table calculations remain available when the selected T lies outside this plot. This is not a relaxation trajectory.')} />
    </div><Readout rows={rows} t={t} label={t('유한 스펙트럼 자유에너지 관측 표', 'Finite-spectrum free-energy observations')} />
  </div>;
}
function GasView({ values, language }: { values: number[]; language: Locale }) {
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  const r = useMemo(() => evaluateGasFreeEnergyValues(values), [values]);
  const history = useMemo(() => Array.from({ length: 49 }, (_, i) => { const x = -3 + i / 16, v = [...values]; v[3] = 10 ** x; return { x, result: evaluateGasFreeEnergyValues(v) }; }), [values]);
  const x = Math.log10(values[3]), budget = r.conditionalEquilibrationBudget;
  const rows: Row[] = [
    { label: 'V*', q: r.equilibriumVolume, unit: 'm³' }, { label: 'V/V*', q: r.volumeRatioToEquilibrium, unit: '1' },
    { label: 'P−pgas', q: r.externalMinusGasPressure, unit: 'Pa' }, { label: 'F', q: r.helmholtzFreeEnergy, unit: 'J' },
    { label: 'Ggas=F+nRT', q: r.gasGibbsFreeEnergy, unit: 'J' }, { label: 'Φ=F+PV', q: r.constrainedPotential, unit: 'J' },
    { label: 'G*', q: r.gibbsAtExternalPressureEquilibrium, unit: 'J' }, { label: 'Φ−G*', q: r.availability, unit: 'J' },
    { label: '(Φ−G*)/(nRT)', q: r.availabilityPerThermalEnergy, unit: '1' },
    { label: 'ΔU', q: budget.energyChangeGas, unit: 'J' }, { label: 'Qgas', q: budget.heatIntoGas, unit: 'J' },
    { label: 'Wby gas', q: budget.workByGas, unit: 'J' }, { label: 'ΔSgas', q: budget.entropyChangeGas, unit: 'J/K' },
    { label: 'ΔSbath', q: budget.entropyChangeThermalReservoir, unit: 'J/K' }, { label: 'Sgen', q: budget.entropyProduction, unit: 'J/K' },
    { label: 'Wrev', q: budget.maximumReversibleWorkByGas, unit: 'J' }, { label: 'Wrev−W', q: budget.workDeficitFromReversible, unit: 'J' },
  ];
  return <div data-free-energy-view="gas" lang={language}><p role="status" className="ai-logic-statement">{r.equilibriumConditionSatisfied ? t('정확한 압력 평형', 'Exact pressure equilibrium') : t('현재 체적은 압력 평형과 다릅니다. 표시 체적 V*도 반올림일 수 있습니다.', 'The query volume differs from pressure equilibrium. The displayed V* can also be rounded.')}</p>
    <div className="ai-matrix-panels"><EnergyPlot marker="free-energy-gas-potentials" title={t('외부 제약을 포함한 함수', 'Include the external constraint')} t={t} domain={[-3, 0]} xLabel="log₁₀(V / m³)" query={[x, r.constrainedPotential]}
      series={[chart('F', history.map(s => [s.x, s.result.helmholtzFreeEnergy]), 'blue'), chart('Ggas', history.map(s => [s.x, s.result.gasGibbsFreeEnergy]), 'orange'), chart('Φ', history.map(s => [s.x, s.result.constrainedPotential]), 'green'), chart('G*', history.map(s => [s.x, s.result.gibbsAtExternalPressureEquilibrium]), 'neutral')]}
      caption={t('동일한 n,T,P에서 체적별 상태를 비교합니다. Ggas의 내부 압력은 변하지만 Φ에는 고정 외압 P가 들어갑니다. 주황 점은 현재 Φ입니다.', 'Compare states at the same n,T,P. The internal pressure in Ggas varies, while Φ uses the fixed external P. The orange point is the selected Φ.')} />
      <EnergyPlot marker="free-energy-gas-availability" title={t('평형까지의 가용 에너지', 'Availability above equilibrium')} t={t} domain={[-3, 0]} xLabel="log₁₀(V / m³)" query={[x, r.availability]}
        series={[chart('Φ−G*', history.map(s => [s.x, s.result.availability]), 'blue')]} caption={t('양의 차이는 nRT(x−1−ln x)로 직접 계산합니다. x=V/V*. 곡선을 움직이는 피스톤의 시간 궤적으로 읽지 마세요.', 'Compute the positive difference directly as nRT(x−1−ln x), x=V/V*. This curve is not a moving piston trajectory.')} />
    </div><p className="ai-small">{t('아래 수지는 일정 외압의 일, 고정 열저장고, 다른 일과 끝점 운동에너지가 없다는 조건의 끝점 비교입니다. 가역 비교 일은 변하는 준정적 하중을 사용합니다. Fr는 현재 n,T,Vr에 주어진 기준값이므로 온도를 바꾼 실행 간 절대 에너지를 직접 비교하지 않습니다.', 'Budgets below compare endpoints assuming constant external-pressure work, a fixed thermal bath, no other work and no endpoint kinetic store. Reversible comparison work uses a varying quasistatic load. Fr is supplied for the current n,T,Vr; absolute energies across runs with different T are not directly comparable.')}</p>
    <Readout rows={rows} t={t} label={t('기체 가용 함수 관측 표', 'Gas availability observations')} />
  </div>;
}
export function FreeEnergyView({ kind, values, language }: { kind: FreeEnergyKind; values: number[]; language: Locale }) {
  return kind === 'finite-spectrum-helmholtz-kl-variational' ? <HelmholtzView values={values} language={language} /> : <GasView values={values} language={language} />;
}
/** Fixed examples are independent of the editable experiment and its storage. */
export function FreeEnergyConceptView({ kind, step, language }: { kind: FreeEnergyKind; step: number; language: Locale }) {
  if (!Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Invalid free-energy concept step');
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  const values = freeEnergyTeaching[kind].conceptInputs[step], rows: Row[] = [];
  let figure: React.ReactNode;
  if (kind === 'finite-spectrum-helmholtz-kl-variational') {
    const r = evaluateHelmholtzValues(values);
    const points = (target: boolean): Series => ({ name: target ? 'π' : 'p', color: target ? 'orange' : 'blue',
      segments: r.trial.states.flatMap((s, i) => { const q = target ? r.trial.canonical.states[i].probability : s.probability;
        return q.value === null ? [] : [[[i, q.value] as [number, number]]]; }) });
    figure = <WavePlot marker={`free-energy-concept-spectrum-${step}`} title={t('고정 예제의 미시상태 확률', 'Microstate probabilities in a fixed example')} series={[points(false), points(true)]}
      xDomain={[0, 2]} yDomain={[0, 1]} xLabel={t('미시상태 번호', 'Microstate index')} yLabel="p, π"
      caption={t('분리된 점은 각각의 미시상태입니다. 시간 경로가 아닙니다. 범위 밖 확률은0에 찍지 않고 표의 로그에 남깁니다.', 'Separate dots are distinct microstates, not a time path. Out-of-range probabilities are retained logarithmically in the table rather than plotted at zero.')} />;
    rows.push({ label: 'F', q: r.trial.perCopy.helmholtz, unit: 'J' }, { label: 'Feq', q: r.equilibrium.perCopy.helmholtz, unit: 'J' },
      { label: 'D(p∥π)', q: r.divergence, unit: 'nat', additional: r.resolution === 'positive-log-tail-resolved' ? r.logDivergence : undefined },
      { label: 'F−Feq', q: r.perCopyGap, unit: 'J' }, { label: 'π₂', q: r.trial.canonical.states[2].probability, unit: '1', additional: r.trial.canonical.states[2].logProbability });
  } else {
    const r = evaluateGasFreeEnergyValues(values);
    const h = Array.from({ length: 33 }, (_, i) => { const v = [...values], x = -3 + 3 * i / 32; v[3] = 10 ** x; return { x, r: evaluateGasFreeEnergyValues(v) }; });
    figure = <EnergyPlot marker={`free-energy-concept-gas-${step}`} title={t('고정 예제의 체적 비교', 'Volume comparison in a fixed example')} t={t} domain={[-3, 0]} xLabel="log₁₀(V / m³)" query={[Math.log10(values[3]), r.constrainedPotential]}
      series={[chart('F', h.map(s => [s.x, s.r.helmholtzFreeEnergy]), 'blue'), chart('Φ', h.map(s => [s.x, s.r.constrainedPotential]), 'green'), chart('G*', h.map(s => [s.x, s.r.gibbsAtExternalPressureEquilibrium]), 'neutral')]}
      caption={t('동일한 고정 n,T,P에서 비교한 상태들입니다. 이 고정 그림은 아래 실습의 현재 설정과 독립적입니다.', 'States compared at the same fixed n,T,P. This reference figure is independent of the editable experiment below.')} />;
    rows.push({ label: 'F', q: r.helmholtzFreeEnergy, unit: 'J' }, { label: 'Φ', q: r.constrainedPotential, unit: 'J' },
      { label: 'G*', q: r.gibbsAtExternalPressureEquilibrium, unit: 'J' }, { label: 'Φ−G*', q: r.availability, unit: 'J' },
      { label: 'P−pgas', q: r.externalMinusGasPressure, unit: 'Pa' });
  }
  return <div data-free-energy-concept={kind} data-free-energy-step={step} lang={language}><p className="ai-small">{t('좁은 화면에서는 그림을 좌우로 밀어 보세요. 키보드로는 그림 영역에 초점을 두고 좌우 방향키를 사용하세요.', 'On narrow screens, swipe the diagram horizontally. With a keyboard, focus its region and use the left and right arrow keys.')}</p>{figure}
    <Readout rows={rows} t={t} label={t('고정 자유에너지 개념 예제 관측', 'Fixed free-energy concept observations')} /></div>;
}
export default function FreeEnergyLab({ kind }: { kind: FreeEnergyKind }) {
  const { language } = useLanguage(), id = useId(), t: Text = (ko, en) => language === 'ko' ? ko : en;
  const initial = freeEnergyPresets[kind][0].values;
  const validator = kind === 'finite-spectrum-helmholtz-kl-variational' ? validHelmholtzValues : validGasFreeEnergyValues;
  const [values, setValues, ready, storage] = useSavedState(`suanlab-learning:${kind}:lab`, initial, validator);
  const [message, setMessage] = useState(false);
  const [selectedGuide, setSelectedGuide] = useState(0);
  const guides = freeEnergyTeaching[kind].guide.experiments, guide = guides[selectedGuide];
  const [playing, setPlaying] = useState(false), [frame, setFrame] = useState<number | null>(null);
  const panel = useRef<HTMLElement>(null), active = useSceneActivity(panel), motion = useSiteMotion();
  useEffect(() => { if (!active || frame === 32) setPlaying(false); }, [active, frame]);
  useEffect(() => {
    if (!playing || !active || frame === null || frame >= 32) return;
    const timer = setInterval(() => { const next = frame + 1; setValues(freeEnergyComparisonValues(kind, values, next)); setFrame(next); }, 250);
    return () => clearInterval(timer);
  }, [playing, active, frame, kind, values, setValues]);
  const load = (v: number[]) => { setPlaying(false); setFrame(null); if (validFreeEnergyValues(kind, v)) { setValues([...v]); setMessage(false); } else setMessage(true); };
  const showFrame = (index: number) => { setValues(freeEnergyComparisonValues(kind, values, index)); setFrame(index); setMessage(false); };
  return <section className="ai-lab" ref={panel} id="experiment" aria-labelledby={`${id}-title`}>
    <div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id={`${id}-title`}>{t('자유에너지 비교 실험', 'Compare free energies')}</h2></div><span className="ai-live-dot">{t('상태 비교', 'State comparison')}</span></div>
    <p className="ai-small">{t('좁은 화면에서는 그림을 좌우로 밀어 보세요. 키보드로는 그림 영역에 초점을 두고 좌우 방향키를 사용하세요.', 'On narrow screens, swipe the diagram horizontally. With a keyboard, focus its region and use the left and right arrow keys.')}</p>
    <details className="ai-guided-lab" open data-free-energy-guides={kind}><summary>{t('안내를 따라 실험하기', 'Follow a guided experiment')}</summary>
      <div className="ai-guided-choices" role="group" aria-label={t('안내 실험 선택', 'Choose a guided experiment')}>{guides.map((g, i) => <Button key={i} type="button" variant="outline" aria-pressed={selectedGuide === i} onClick={() => setSelectedGuide(i)}>{i + 1}. {g.title[language]}</Button>)}</div>
      <h3>{guide.title[language]}</h3><p><strong>{t('먼저 예측', 'Predict first')} · </strong>{guide.predict[language]}</p>
      <p><strong>{t('조작 방법', 'Try it')} · </strong>{guide.action[language]}</p>
      <Button type="button" variant="outline" disabled={!ready} onClick={() => load(freeEnergyPresets[kind][guide.preset].values)}>{t('이 예제 열기', 'Load this example')}</Button>
      <details key={selectedGuide} className="ai-guided-answer"><summary>{t('결과 해석 보기', 'Reveal the explanation')}</summary><p>{guide.explain[language]}</p></details>
      <p className="ai-small">{t('예제를 열면 현재 설정을 교체합니다. 자유롭게 수정할 수 있습니다.', 'Loading replaces the current settings. You can edit them freely.')}</p>
    </details>
    <fieldset className="ai-presets" disabled={!ready}><legend>{t('비교할 예제', 'Examples to compare')}</legend>{freeEnergyPresets[kind].map((preset, i) => <Button key={i} type="button" variant="outline" size="sm" onClick={() => load(preset.values)}>{preset.label[language]}</Button>)}</fieldset>
    <fieldset className="ai-parameters" disabled={!ready}><legend>{t('모델 설정', 'Model settings')}</legend><div className="ai-two-fields">{freeEnergyControls[kind].map((control, i) => <NumberField resetKey={values} integer={kind==='finite-spectrum-helmholtz-kl-variational'&&i===6} key={i} {...control} label={control.label[language]} value={values[i]} onChange={n => load(values.map((v, j) => j === i ? n : v))} />)}</div></fieldset>
    {message && <p role="status">{t('가중치는 적어도 하나가 양수여야 하며 복제 수는 정수여야 합니다.', 'At least one weight must be positive and the copy count must be an integer.')}</p>}
    <div className="ai-lab-actions"><Button type="button" variant="outline" disabled={!ready || !active} aria-pressed={playing} onClick={() => {
      if (playing) setPlaying(false); else { if (frame === null || frame === 32) showFrame(0); setPlaying(true); }
    }}>{playing ? t('비교 순회 일시정지', 'Pause comparison scan') : t('비교 상태 순회', 'Scan comparison states')}</Button>
    <Button type="button" variant="outline" disabled={!ready || frame === 32} onClick={() => { setPlaying(false); showFrame(frame === null ? 0 : frame + 1); }}>{t('비교 상태 한 단계', 'Step comparison state')}</Button>
    <Button type="button" variant="outline" disabled={!ready} onClick={() => { setPlaying(false); showFrame(0); }}>{t('비교 순회 처음으로', 'Restart comparison scan')}</Button></div>
    <p className="ai-small">{t('순회는 33개 비교 상태를 고르는 표시 기능이며 실제 시간·완화·적분이 아닙니다. 스펙트럼에서는 세 가중치를 균등한 바닥 조건부 분포의 비교값으로 교체하고, 기체에서는 현재 등온선의 체적만 바꿉니다.', 'Scanning selects 33 comparison states; it is not physical time, relaxation or integration. For the spectrum it replaces the three weights with a uniform-ground-conditional comparison family; for the gas it changes only volume on the current isotherm.')} {frame !== null && <output aria-live="off">{frame + 1}/33</output>}</p>
    {!motion.canAnimate && <p className="ai-small">{t('모션 설정으로 자동 순회를 멈췄습니다. 수동 단계는 사용할 수 있습니다.', 'Motion preferences pause automatic scanning. Manual steps remain available.')}</p>}
    <Button type="button" variant="outline" disabled={!ready} onClick={() => load(initial)}>{t('모든 실험 설정 초기화', 'Reset all experiment settings')}</Button>
    <FreeEnergyView kind={kind} values={values} language={language} />
    <p className="ai-small">{storage ? t('설정을 이 브라우저에 저장합니다.', 'Settings are saved in this browser.') : t('브라우저 저장을 사용할 수 없습니다.', 'Browser storage is unavailable.')}</p>
  </section>;
}
