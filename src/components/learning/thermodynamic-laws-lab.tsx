'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { useSiteMotion } from '@/components/motion/motion-provider';
import { useSceneActivity } from '@/components/motion/use-scene-activity';
import { thermodynamicLawLabs, type BookLocale } from '@/data/interactive-book';
import { evaluateCarnotLaw, evaluateReservoirLaw, evaluateSpectrumLaw, lawObservationSegments,
  validCarnotValues, validReservoirValues, validSpectrumValues, thermodynamicLawConceptInputs, type ThermodynamicLawKind } from '@/lib/learning-thermodynamic-laws';
import type { BinaryObservation } from '@/lib/learning-binary-products';
import type { SpectrumQuantity } from '@/lib/learning-gapped-spectrum';
import NumberField from './thermodynamic-number-field';
import GuidedExperiments from './guided-experiments';
import { useSavedState } from './use-saved-state';
import { WavePlot, waveNumber as fmt, type Series, type Mark } from './waves-plot';

type Quantity = BinaryObservation | SpectrumQuantity;
type Row = [string, Quantity | null, string];
type Text = (ko: string, en: string) => string;
type Carnot = ReturnType<typeof evaluateCarnotLaw>;
type Reservoir = ReturnType<typeof evaluateReservoirLaw>;
type Spectrum = ReturnType<typeof evaluateSpectrumLaw>;
const number = (value: number): BinaryObservation => ({ value, sign: Math.sign(value), logMagnitude: value === 0 ? null : Math.log(Math.abs(value)) });
const point = (x: number, q: BinaryObservation | null, color = 'blue'): Mark[] =>
  q?.value == null || !Number.isFinite(x) ? [] : [{ point: [x, q.value], color }];
function curve<T>(name: string, history: readonly T[], x: (s: T) => number, q: (s: T) => BinaryObservation | null,
  color = 'blue', scale: 'value' | 'logMagnitude' = 'value'): Series {
  return { name, color, segments: lawObservationSegments(history, x, q, scale) };
}
function positiveDomain(series: Series[]): [number, number] {
  const maximum = Math.max(0, ...series.flatMap(s => s.segments.flat().map(p => p[1])));
  return [0, maximum > 0 ? maximum * 1.08 : 1];
}

function ObservationTable({ rows, t, label }: { rows: Row[]; t: Text; label?: string }) {
  const status = (q: Quantity | null) => q === null ? t('미정', 'Undefined') : q.sign === 0 ? t('실제 0', 'Physical zero')
    : q.value === null ? t('비영 · 범위 밖', 'Nonzero, out of range') : t('반올림 표시', 'Rounded display');
  const logStatus = (q: Quantity | null, second: boolean) => {
    if (q === null) return t('미정', 'Undefined');
    if ('logMagnitudeStatus' in q) {
      const value = second ? q.logNegativeLogMagnitudeStatus : q.logMagnitudeStatus;
      return value === 'finite' ? t('유한', 'Finite') : value === 'outside-number-range' ? t('범위 밖', 'Out of range') : t('해당 없음', 'Not applicable');
    }
    return second || q.sign === 0 ? t('해당 없음', 'Not applicable') : q.logMagnitude === null ? t('범위 밖', 'Out of range') : t('유한', 'Finite');
  };
  return <div className="ai-readout" data-testid="law-readout"><h3>{t('관측값과 표현 범위', 'Observations and representable ranges')}</h3>
    <p className="ai-small">{t('크기 로그는 표의 단위로 쓴 값의 ln|값|입니다. 부호는 별도로 읽으세요. 추가 로그는 정의되는 경우 ln(−ln|값|)입니다. —는 미정·해당 없음·범위 밖일 수 있으므로 상태를 확인하세요. 표시 확률 1도 반올림일 수 있습니다.', 'Log magnitude is ln|value| in the stated unit; read the sign separately. The additional logarithm is ln(−ln|value|) where defined. A dash can mean undefined, not applicable or out of range; inspect its status. A displayed probability of one can be rounded.')}</p>
    <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={label ?? t('열역학 법칙 관측 표', 'Thermodynamic-law observations')}><table><thead><tr>{[
      t('관측', 'Observation'), t('값', 'Value'), t('단위', 'Unit'), t('부호', 'Sign'), t('값 상태', 'Value status'),
      t('로그 크기', 'Log magnitude'), t('로그 상태', 'Log status'), t('추가 로그', 'Additional log'), t('추가 로그 상태', 'Additional log status'),
    ].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead><tbody>{rows.map(([label, q, unit]) => <tr key={label}>
      <th scope="row">{label}</th><td>{fmt(q?.value)}</td><td>{unit}</td><td>{q?.sign ?? '—'}</td><td>{status(q)}</td>
      <td>{fmt(q?.logMagnitude)}</td><td>{logStatus(q, false)}</td><td>{fmt(q && 'logNegativeLogMagnitude' in q ? q.logNegativeLogMagnitude : null)}</td><td>{logStatus(q, true)}</td>
    </tr>)}</tbody></table></div></div>;
}
function CarnotFigures({ result: r, t }: { result: Carnot; t: Text }) {
  if (!r.query || !r.cycle) return null;
  const h = r.history, q = r.query;
  const path = curve(t('순환 경로', 'Cycle path'), h, s => s.volume, s => number(s.pressure));
  const volumes = h.map(s => s.volume), vmin = Math.min(...volumes), vmax = Math.max(...volumes);
  const pvDomain: [number, number] = vmin === vmax ? [vmin * .9, vmax * 1.1] : [vmin, vmax];
  const ts = curve(t('계 엔트로피 변화', 'System entropy change'), h, s => s.entropyChangeGas.value ?? NaN, s => number(s.temperature));
  const entropy = h.flatMap(s => s.entropyChangeGas.value === null ? [] : [s.entropyChangeGas.value]);
  const slow = Math.min(0, ...entropy), shigh = Math.max(0, ...entropy), margin = Math.max(1e-15, (shigh - slow) * .06);
  return <div className="ai-matrix-panels">
    <WavePlot marker="law-carnot-pv" title={t('압력과 부피의 순환', 'Pressure-volume cycle')} series={[path]}
      dots={[{ point: [q.volume, q.pressure], color: 'orange' }]} xDomain={pvDomain} yDomain={positiveDomain([path])}
      xLabel="V (m³)" yLabel="p (Pa)" caption={t('점은 현재 경로 위치입니다. 경로의 방향과 네 구간은 아래 수지에서 확인하세요.', 'The dot marks the current path position. Read the direction and four legs in the budgets below.')} />
    <WavePlot marker="law-carnot-ts" title={t('온도와 엔트로피의 순환', 'Temperature-entropy cycle')} series={[ts]}
      dots={q.entropyChangeGas.value === null ? [] : [{ point: [q.entropyChangeGas.value, q.temperature], color: 'orange' }]}
      xDomain={[slow - margin, shigh + margin]} yDomain={[r.parameters.coldTemperature * .95, r.parameters.hotTemperature * 1.05]}
      xLabel="ΔS (J/K)" yLabel="T (K)" caption={t('가역 경로에서 열은 ∫T dS입니다. 단열 구간은 수직선이며 면적의 부호는 진행 방향에 따릅니다.', 'For a reversible path heat is ∫T dS. Adiabats are vertical; the signed area follows the traversal direction.')} />
    <WavePlot marker="law-carnot-energy" title={t('현재 위치까지의 제1법칙', 'First-law budget to the current position')}
      series={[curve('Q', h, s => s.progress, s => s.netHeat), curve('W', h, s => s.progress, s => s.workByGas, 'orange'), curve('ΔU', h, s => s.progress, s => s.energyChange, 'green')]}
      dots={point(q.progress, q.workByGas, 'orange')} xDomain={[0, 1]} xLabel={t('경로 진행률 · 시간 아님', 'Path progress, not time')} yLabel="J"
      caption={t('ΔU=Q−W입니다. 역방향 순환도 일부 구간에서 누적 일이 양수일 수 있습니다.', 'ΔU=Q−W. A reverse cycle can have positive accumulated work during part of its path.')} />
    <WavePlot marker="law-carnot-entropy" title={t('계와 두 저장고의 엔트로피', 'Entropy of system and two reservoirs')}
      series={[curve('ΔS', h, s => s.progress, s => s.entropyChangeGas), curve('ΔSₕ', h, s => s.progress, s => s.hotReservoirEntropyChange, 'orange'), curve('ΔS꜀', h, s => s.progress, s => s.coldReservoirEntropyChange, 'green')]}
      xDomain={[0, 1]} xLabel={t('경로 진행률 · 시간 아님', 'Path progress, not time')} yLabel="J/K"
      caption={t('세 변화의 합은 가역 경로에서 0입니다. 반올림한 세 표시값을 더해 생성량을 추정하지 마세요.', 'The three changes sum to zero on this reversible path. Do not estimate production by summing rounded displayed values.')} />
  </div>;
}
function ReservoirFigures({ result: r, t }: { result: Reservoir; t: Text }) {
  if (!r.query) return null;
  const h = r.history, q = r.query, x = (s: typeof q) => s.parameters.relativeEntropyGeneration;
  return <><figure className="ai-foundation-figure" data-law-ports="true"><h3>{t('저장고·장치·일 포트', 'Reservoir, device and work ports')}</h3>
    <div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={t('순환 장치 포트 그림', 'Cyclic device port diagram')}>
      <svg viewBox="0 0 350 280" role="img" aria-label={t('고온에서 입력, 저온으로 방출, 일 포트는 부호에 따라 방향 전환', 'Input from hot reservoir, rejection to cold reservoir, work direction follows its sign')}>
        <g fill="none" stroke="var(--learning-blue)" strokeWidth="2"><rect x="30" y="15" width="130" height="45" rx="8" /><rect x="30" y="105" width="130" height="55" rx="8" /><rect x="30" y="215" width="130" height="45" rx="8" />
          {q.heatHotIntoDevice.sign !== 0 && <path d="M95 60V100m-5-7 5 7 5-7" />}
          {q.heatColdIntoDevice.sign !== 0 && <path d="M95 160V210m-5-7 5 7 5-7" />}
          {q.workByDevice.sign !== 0 && <path d={q.workByDevice.sign > 0 ? 'M160 132H260m-7-5 7 5-7 5' : 'M260 132H160m7-5-7 5 7 5'} />}
        </g><g fill="currentColor" fontSize="12" textAnchor="middle"><text x="95" y="42">Tₕ={fmt(r.parameters.hotTemperature)} K</text><text x="95" y="137">{t('순환 장치', 'Cyclic device')}</text><text x="95" y="242">T꜀={fmt(r.parameters.coldTemperature)} K</text>
          <text x="210" y="85">Qₕ={fmt(q.heatHotIntoDevice.value)} J</text><text x="210" y="190">Q꜀={fmt(q.heatColdIntoDevice.value)} J</text><text x="245" y="113">W={fmt(q.workByDevice.value)} J</text></g>
      </svg></div><figcaption>{t('화살표는 반올림 표시값이 아닌 모형의 부호를 따릅니다. 장치 내부 궤적과 시간은 정하지 않았습니다.', 'Arrows follow the model signs, including nonzero values outside display range. No internal trajectory or elapsed time is specified.')}</figcaption></figure>
    <div className="ai-matrix-panels"><WavePlot marker="law-reservoir-work" title={t('생성에 따른 순환 일', 'Cycle work as generation changes')}
      series={[curve('W', h, x, s => s.workByDevice), curve(t('가역 일', 'Reversible work'), h, x, s => s.reversibleWork, 'green')]}
      dots={point(r.parameters.relativeEntropyGeneration, q.workByDevice)} vertical={[0]} xDomain={[-1, 4]} xLabel="σ" yLabel="J"
      caption={t('σ<0은 실현할 수 없는 비교입니다. 양의 생성은 일을 줄이며, 음의 일은 냉동기임을 뜻하지 않습니다.', 'σ<0 is counterfactual. Positive production reduces work; negative work does not identify a refrigerator.')} />
      <WavePlot marker="law-reservoir-generation" title={t('전체 엔트로피 생성', 'Total entropy generation')}
        series={[curve('Sgen', h, x, s => s.entropyGeneration, 'orange')]} dots={point(r.parameters.relativeEntropyGeneration, q.entropyGeneration, 'orange')}
        vertical={[0]} xDomain={[-1, 4]} xLabel="σ" yLabel="J/K" caption={t('이 값은 한 순환의 생성량이며 생성률이 아닙니다. 극소 값은 관측 표의 부호와 로그로 읽으세요.', 'This is production per cycle, not a rate. Use the table sign and logarithm for tiny values.')} />
    </div></>;
}
function SpectrumFigures({ result: r, t }: { result: Spectrum; t: Text }) {
  const h = r.history, q = r.query, x = (s: typeof q) => Math.log10(s.parameters.temperature);
  const plot = (key: 'entropyPerCopyInKB' | 'heatCapacityPerCopyInKB' | 'meanEnergyPerCopyInKB', title: string, unit: string) => {
    const raw = curve(title, h, x, s => s[key]);
    const maximum = Math.max(0, ...raw.segments.flat().map(p => p[1]));
    const scale = maximum > 0 && maximum < 1e-150 ? maximum : 1;
    const series = [{ ...raw, segments: raw.segments.map(segment => segment.map(([a, b]) => [a, b / scale] as [number, number])) }];
    const dots = r.parameters.temperature > 0 ? point(x(q), q[key]).map(dot => ({ ...dot, point: [dot.point[0], dot.point[1] / scale] as [number, number] })) : [];
    return <WavePlot key={key} marker={`law-spectrum-${key}`} title={title} series={series}
      dots={dots} xDomain={r.logTemperatureDomain} yDomain={positiveDomain(series)}
      xLabel="log₁₀(T / K)" yLabel={scale === 1 ? unit : `${unit} ÷ ${fmt(scale)}`} caption={t('양의 온도의 정준 평형을 비교합니다. T=0은 아래 표의 별도 극한입니다. 미정·범위 밖 구간은 잇지 않습니다.', 'Positive-temperature canonical equilibria. T=0 is a separate limit in the table below. Undefined or out-of-range intervals are not connected.') + (scale === 1 ? '' : t(` 세로축 값에 ${fmt(scale)}을 곱하면 표시 단위의 값입니다.`, ` Multiply the ordinate by ${fmt(scale)} to recover the value in the stated unit.`))} />;
  };
  const logScale = Math.max(1, ...h.flatMap(s => s.upperGroupProbability.logMagnitude === null ? [] : [Math.abs(s.upperGroupProbability.logMagnitude)]));
  const logSeries = curve('ln p₁', h, x, s => s.upperGroupProbability, 'orange', 'logMagnitude');
  logSeries.segments = logSeries.segments.map(segment => segment.map(([a, b]) => [a, b / logScale]));
  const gapZero = q.spectrum === 'exact-zero-gap', upperY = gapZero ? 180 : 60;
  return <><figure className="ai-foundation-figure" data-law-spectrum="true"><h3>{t('한 복제의 두 에너지 상태군', 'Two energy groups of one copy')}</h3>
    <div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={t('상태군 에너지 그림', 'Energy-group diagram')}><svg viewBox="0 0 350 250" role="img" aria-label={gapZero ? t('간격 0: 두 상태군 모두 바닥', 'Zero gap: both groups are ground') : t('바닥 상태군과 양의 간격 위 상태군', 'Ground group and upper group at positive gap')}>
      <path d={`M20 180H150M200 ${upperY}H330`} stroke="var(--learning-blue)" strokeWidth="2" />
      <g fill="var(--learning-orange)">{Array.from({ length: r.parameters.groundDegeneracy }, (_, i) => <circle key={`g${i}`} cx={25 + 8 * i} cy="173" r="3" />)}{Array.from({ length: r.parameters.upperDegeneracy }, (_, i) => <circle key={`u${i}`} cx={205 + 8 * i} cy={upperY - 7} r="3" />)}</g>
      <g fill="currentColor" fontSize="12" textAnchor="middle"><text x="85" y="205">E=0 · g₀={r.parameters.groundDegeneracy}</text><text x="265" y={upperY + 25}>E=Δ · g₁={r.parameters.upperDegeneracy}</text><text x="85" y="228">p₀={fmt(q.groundGroupProbability.value)}</text><text x="265" y={upperY + 48}>p₁={fmt(q.upperGroupProbability.value)}</text></g>
    </svg></div><figcaption>{t('각 점은 미시상태 하나이며, 표기 확률은 상태군 전체의 확률입니다. 세로 간격은 개념도이며 에너지 크기 비례가 아닙니다.', 'Each dot is a microstate; probabilities refer to whole groups. Vertical separation is schematic, not proportional to energy.')}</figcaption></figure>
    <div className="ai-matrix-panels">
      {plot('entropyPerCopyInKB', t('복제당 엔트로피', 'Entropy per copy'), 'S / (N kB)')}
      {plot('heatCapacityPerCopyInKB', t('복제당 열용량', 'Heat capacity per copy'), 'C / (N kB)')}
      {plot('meanEnergyPerCopyInKB', t('복제당 평균 에너지', 'Mean energy per copy'), 'U / (N kB) (K)')}
      <WavePlot marker="law-spectrum-log-probability" title={t('작은 상태군 확률의 로그', 'Logarithm of small group probability')}
        series={[logSeries]} xDomain={r.logTemperatureDomain}
        xLabel="log₁₀(T / K)" yLabel={`ln p₁ ÷ ${fmt(logScale)}`} caption={t(`세로축 값에 ${fmt(logScale)}을 곱하면 ln p₁입니다. 첫 로그도 표현 불가이면 선을 끊습니다. 추가 로그와 상태는 관측 표에 남깁니다.`, `Multiply the ordinate by ${fmt(logScale)} to recover ln p₁. A missing first logarithm breaks the line. The table retains the additional logarithm and its availability status.`)} />
    </div></>;
}

/** Pure view also supports inspection before the units are registered. */
export function ThermodynamicLawView({ kind, values, language }: { kind: ThermodynamicLawKind; values: number[]; language: BookLocale }) {
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  const result = useMemo(() => kind === 'carnot-cycle-engine-refrigerator' ? { carnot: evaluateCarnotLaw(values) }
    : kind === 'two-reservoir-cycle-entropy-admissibility' ? { reservoir: evaluateReservoirLaw(values) }
      : { spectrum: evaluateSpectrumLaw(values) }, [kind, values]);
  const rows: Row[] = [];
  const c = result.carnot, r = result.reservoir, s = result.spectrum;
  if (c?.status === 'invalid-reservoir-order' || r?.status === 'invalid-reservoir-order')
    return <p role="status" className="ai-logic-statement">{t('고온 저장고 온도가 저온 저장고보다 낮습니다. 두 온도를 수정하세요. 역할을 자동 교환하지 않습니다.', 'The hot reservoir is colder than the cold reservoir. Edit the temperatures; roles are not swapped automatically.')}</p>;
  if (c?.query && c.cycle) {
    const q = c.query, cycle = c.cycle;
    rows.push(['T', number(q.temperature), 'K'], ['V', number(q.volume), 'm³'], ['p', number(q.pressure), 'Pa'],
      ['ΔT', q.temperatureChange, 'K'], ['ΔV', q.volumeChange, 'm³'], ['Qₕ', q.heatHotIntoGas, 'J'], ['Q꜀', q.heatColdIntoGas, 'J'],
      ['Q', q.netHeat, 'J'], ['W', q.workByGas, 'J'], ['ΔU', q.energyChange, 'J'], ['ΔS', q.entropyChangeGas, 'J/K'],
      ['ΔSₕ', q.hotReservoirEntropyChange, 'J/K'], ['ΔS꜀', q.coldReservoirEntropyChange, 'J/K'], ['Sgen', number(q.entropyProduction), 'J/K'],
      [t('전체 순환 일', 'Complete-cycle work'), cycle.complete.workByGas, 'J'], [t('온도 카르노 상한', 'Temperature Carnot bound'), cycle.carnotTemperatureBound, '1'],
      [t('형식적 W/Qₕ', 'Formal W/Qₕ'), cycle.workPerHotHeat, '1'], [t('기관 효율', 'Engine efficiency'), cycle.engineEfficiency, '1'],
      [t('냉동 성능계수', 'Refrigerator COP'), cycle.refrigeratorCOP, '1'], [t('난방 성능계수', 'Heating COP'), cycle.heatingCOP, '1']);
    cycle.legs.forEach((leg, i) => rows.push([`${t('구간', 'Leg')} ${i + 1} · Q`, leg.heat, 'J'], [`${t('구간', 'Leg')} ${i + 1} · W`, leg.work, 'J'], [`${t('구간', 'Leg')} ${i + 1} · ΔU`, leg.energyChange, 'J'], [`${t('구간', 'Leg')} ${i + 1} · ΔS`, leg.entropyChange, 'J/K']));
  }
  if (r?.query) {
    const q = r.query;
    rows.push(['Qₕ', q.heatHotIntoDevice, 'J'], ['Q꜀', q.heatColdIntoDevice, 'J'], ['W', q.workByDevice, 'J'], ['Sgen', q.entropyGeneration, 'J/K'],
      ['ΔSₕ', q.hotReservoirEntropyChange, 'J/K'], ['ΔS꜀', q.coldReservoirEntropyChange, 'J/K'], ['ΔUcycle', number(q.cycleEnergyChange), 'J'], ['ΔScycle', number(q.cycleEntropyChange), 'J/K'],
      [t('가역 순환 일', 'Reversible-cycle work'), q.reversibleWork, 'J'], [t('생성에 따른 일 감소', 'Work deficit from generation'), q.workDeficitFromReversible, 'J'],
      [t('온도 카르노 상한', 'Temperature Carnot bound'), q.carnotTemperatureBound, '1'], [t('형식적 W/Qₕ', 'Formal W/Qₕ'), q.workFraction, '1'], [t('기관 효율', 'Engine efficiency'), q.engineEfficiency, '1']);
  }
  if (s) {
    const q = s.query;
    rows.push(['Δ', q.gapEnergy, 'J'], ['Δ/(kB T)', q.gapToTemperatureRatio, '1'], ['p₀', q.groundGroupProbability, '1'], ['p₁', q.upperGroupProbability, '1'], ['ln Z₁', q.logPartitionPerCopy, '1'],
      ['U', q.meanEnergy, 'J'], ['U/(N kB)', q.meanEnergyPerCopyInKB, 'K'], ['Var(E)', q.energyVariance, 'J²'], ['S', q.entropy, 'J/K'], ['S/(N kB)', q.entropyPerCopyInKB, '1'],
      [t('잔류 S', 'Residual S'), q.residualEntropy, 'J/K'], [t('간격 들뜸 보정', 'Gap excitation correction'), q.excitationEntropy, 'J/K'], ['C', q.heatCapacity, 'J/K'], ['C/(N kB)', q.heatCapacityPerCopyInKB, '1'], ['dS/dT', q.entropyTemperatureDerivative, 'J/K²'],
      [t('선택 간격의 T=0 극한 S', 'T=0 limit S at selected gap'), s.zeroTemperatureLimit.entropy, 'J/K'],
      [t('양의 간격 뒤 저온 극한', 'Low-T limit after positive gap'), q.positiveGapLowTemperatureEntropy, 'J/K'],
      [t('영 간격 뒤 저온 극한', 'Low-T limit after zero gap'), q.zeroGapThenLowTemperatureEntropy, 'J/K']);
  }
  const regimes: Record<string, [string, string]> = { 'heat-engine': ['열기관', 'Heat engine'], refrigerator: ['냉동기', 'Refrigerator'], degenerate: ['퇴화 순환', 'Degenerate cycle'],
    counterfactual: ['제2법칙 위반 가정 · 실현 불가', 'Counterfactual second-law violation, not realizable'], 'work-driven-dissipation': ['일 입력 소산 · 냉동기로 분류하지 않음', 'Work-driven dissipation, not classified as a refrigerator'],
    'heat-transfer': ['영 일 열전달', 'Zero-work heat transfer'], 'no-transfer': ['전달 없음', 'No transfer'] };
  const regime = c?.cycle?.regime ?? r?.query?.regime;
  return <div data-law-view={kind} lang={language}>
    {regime && <p className="ai-logic-statement" role="status">{t(...regimes[regime])}</p>}
    {c?.query && <p className="ai-small">{t('현재 구간', 'Current leg')} {c.query.legIndex + 1} · {c.direction === 'forward' ? t('정방향', 'Forward') : t('역방향', 'Reverse')} · {c.query.stage === 'isothermal' ? t('등온', 'Isothermal') : t('단열', 'Adiabatic')}</p>}
    {c && <CarnotFigures result={c} t={t} />}{r && <ReservoirFigures result={r} t={t} />}{s && <SpectrumFigures result={s} t={t} />}
    {s && <p className="ai-logic-statement">{t('정확한 전체 미시상태 수', 'Exact total microstate count')}: <span className="break-all">{s.query.totalStateCount}</span><br />{t('정확한 바닥 상태 수', 'Exact ground-state count')}: <span className="break-all">{s.query.groundStateCount}</span><br />{t('복제당 확률은 한 복제의 상태군 확률이며, N개 중 하나 이상이 위 상태군일 확률과 다릅니다. dS/dT의 단위 J/K²는 시간당 생성률 W/K와 다릅니다.', 'Per-copy probabilities describe one copy, rather than the probability that at least one of N copies occupies the upper group. dS/dT in J/K² is distinct from a time-production rate in W/K.')}</p>}
    <ObservationTable rows={rows} t={t} />
  </div>;
}

export function ThermodynamicLawConceptView({ kind, step, language }: { kind: ThermodynamicLawKind; step: number; language: BookLocale }) {
  if (!Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Invalid thermodynamic concept step');
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  const values = thermodynamicLawConceptInputs[kind][step], rows: Row[] = [];
  let series: Series[] = [], dots: Mark[] = [], xDomain: [number, number] = [0, 1];
  let title = '', xLabel = t('경로 진행률 · 시간 아님', 'Path progress, not time'), yLabel = '', caption = '';
  if (kind === 'carnot-cycle-engine-refrigerator') {
    const r = evaluateCarnotLaw(values); if (!r.query || !r.cycle) throw new Error('Invalid fixed Carnot example');
    const q = r.query, h = r.history;
    if (step === 1) {
      title = t('가역 순환의 T–S 경로', 'T–S path of a reversible cycle'); xLabel = 'ΔS (J/K)'; yLabel = 'T (K)';
      series = [curve('T', h, s => s.entropyChangeGas.value ?? NaN, s => number(s.temperature))];
      xDomain = [0, Math.max(...h.map(s => s.entropyChangeGas.value ?? 0)) * 1.05];
      dots = point(q.entropyChangeGas.value!, number(q.temperature));
      caption = t('고정 정방향 순환입니다. 가역 열은 ∫T dS이고 전체 엔트로피 생성은 0입니다.', 'Fixed forward cycle. Reversible heat is ∫T dS and total entropy production is zero.');
      rows.push(['Sgen', number(q.entropyProduction), 'J/K'], ['η', r.cycle.engineEfficiency, '1']);
    } else {
      title = step === 2 ? t('역방향의 누적 일', 'Accumulated reverse work') : step === 3 ? t('r=1의 퇴화 순환', 'Degenerate cycle at r=1') : t('완결 순환의 제1법칙', 'First law of a complete cycle'); yLabel = 'J';
      series = [curve('Q', h, s => s.progress, s => s.netHeat), curve('W', h, s => s.progress, s => s.workByGas, 'orange'), curve('ΔU', h, s => s.progress, s => s.energyChange, 'green')];
      dots = point(q.progress, q.workByGas, 'orange');
      caption = t('고정 예제입니다. 현재 구간의 일과 완결 순환의 일, 온도 상한과 실제 성능을 구별하세요.', 'Fixed example. Distinguish partial from complete-cycle work and a temperature bound from actual performance.');
      rows.push(['Wcycle', r.cycle.complete.workByGas, 'J'], ['ΔUcycle', r.cycle.totalEnergyChange, 'J']);
      if (step === 2) rows.push(['COPR', r.cycle.refrigeratorCOP, '1'], ['COPH', r.cycle.heatingCOP, '1']);
      if (step === 3) rows.push(['ηC', r.cycle.carnotTemperatureBound, '1'], ['W/Qₕ', r.cycle.workPerHotHeat, '1'], ['η', r.cycle.engineEfficiency, '1']);
    }
  } else if (kind === 'two-reservoir-cycle-entropy-admissibility') {
    const r = evaluateReservoirLaw(values); if (!r.query) throw new Error('Invalid fixed reservoir example');
    const q = r.query, production = step === 1 || step === 2;
    title = production ? t('엔트로피 생성의 허용 범위', 'Admissible entropy production') : t('순환의 일과 가역 한계', 'Cycle work and reversible limit');
    xLabel = 'σ'; yLabel = production ? 'J/K' : 'J'; xDomain = [-1, 4];
    series = [curve(production ? 'Sgen' : 'W', r.history, s => s.parameters.relativeEntropyGeneration, s => production ? s.entropyGeneration : s.workByDevice)];
    dots = point(values[3], production ? q.entropyGeneration : q.workByDevice);
    caption = t('고정 저장고와 열 입력에서 완결 순환들을 비교합니다. σ<0은 반사실적이며, 이 축은 시간이 아닙니다.', 'Complete cycles at fixed reservoirs and heat input. σ<0 is counterfactual; this axis is not time.');
    rows.push(['W', q.workByDevice, 'J'], ['Sgen', q.entropyGeneration, 'J/K'], ['Wrev−W', q.workDeficitFromReversible, 'J'], ['η', q.engineEfficiency, '1']);
    if (step === 3) rows.push(['Q꜀', q.heatColdIntoDevice, 'J']);
  } else {
    const r = evaluateSpectrumLaw(values), q = r.query, x = (s: typeof q) => Math.log10(s.parameters.temperature);
    xDomain = r.logTemperatureDomain; xLabel = 'log₁₀(T / K)';
    if (step === 0) {
      title = t('상태군 확률', 'Group probability'); yLabel = 'p₁'; series = [curve('p₁', r.history, x, s => s.upperGroupProbability)];
      dots = point(x(q), q.upperGroupProbability); rows.push(['p₁', q.upperGroupProbability, '1']);
    } else if (step === 3) {
      title = t('범위 밖 꼬리의 추가 로그', 'Additional logarithm of an out-of-range tail'); yLabel = 'ln(−ln p₁)';
      series = [curve('ln(−ln p₁)', r.history, x, s => s.upperGroupProbability.logNegativeLogMagnitude === null ? null : number(s.upperGroupProbability.logNegativeLogMagnitude))];
      dots = point(x(q), q.upperGroupProbability.logNegativeLogMagnitude === null ? null : number(q.upperGroupProbability.logNegativeLogMagnitude));
      rows.push(['p₁', q.upperGroupProbability, '1']);
    } else {
      title = t('잔류 엔트로피의 두 고정 예제', 'Two fixed residual-entropy examples'); yLabel = 'S/(N kB)';
      series = [curve('S/(N kB)', r.history, x, s => s.entropyPerCopyInKB)];
      rows.push(['S/(N kB)', q.entropyPerCopyInKB, '1']);
    }
    caption = t('양의 온도의 평형 비교입니다. T=0의 선택 관측은 아래 표에 별도로 두고 로그 축에 잇지 않습니다.', 'Positive-temperature equilibrium comparison. A selected T=0 limit is shown separately in the table, never connected to the log axis.');
  }
  return <div data-law-concept={kind} data-law-concept-step={step} lang={language}>
    <WavePlot marker={`law-concept-${kind}-${step}`} title={title} series={series} dots={dots} xDomain={xDomain}
      yDomain={kind === 'carnot-cycle-engine-refrigerator' && step === 1 ? [250, 650] : kind === 'gapped-spectrum-low-temperature-entropy' && step !== 3 ? positiveDomain(series) : undefined}
      xLabel={xLabel} yLabel={yLabel} caption={caption} />
    <ObservationTable rows={rows} t={t} label={t('고정 개념 예제 관측 표', 'Fixed concept example observations')} />
  </div>;
}
export function ThermodynamicLawConcept({ kind, step }: { kind: ThermodynamicLawKind; step: number }) {
  const { language } = useLanguage(); return <ThermodynamicLawConceptView kind={kind} step={step} language={language} />;
}

export default function ThermodynamicLawsLab({ kind }: { kind: ThermodynamicLawKind }) {
  const { language } = useLanguage(), t: Text = (ko, en) => language === 'ko' ? ko : en;
  const spec = thermodynamicLawLabs[kind], initial = useMemo(() => spec.controls.map(c => c.initial), [spec]);
  const isCarnot = kind === 'carnot-cycle-engine-refrigerator';
  const validator = isCarnot ? validCarnotValues : kind === 'two-reservoir-cycle-entropy-admissibility' ? validReservoirValues : validSpectrumValues;
  const [values, setValues, ready, storage] = useSavedState(`suanlab-learning:${kind}:lab`, initial, validator);
  const panel = useRef<HTMLElement>(null), active = useSceneActivity(panel), motion = useSiteMotion();
  const [playing, setPlaying] = useState(false), [formEpoch, setFormEpoch] = useState(0), ended = values[6] >= 1;
  const validOrder = !isCarnot || values[1] >= values[2];
  useEffect(() => { if (!active || ended || !validOrder) setPlaying(false); }, [active, ended, validOrder]);
  useEffect(() => {
    if (!isCarnot || !playing || !active || !validOrder) return;
    const timer = setInterval(() => setValues(v => v.map((x, i) => i === 6 ? Math.min(1, Number((x + .02).toPrecision(14))) : x)), 200);
    return () => clearInterval(timer);
  }, [isCarnot, playing, active, validOrder, setValues]);
  const load = (next: number[]) => { setPlaying(false); setValues([...next]); setFormEpoch(n => n + 1); };
  return <section className="ai-lab" ref={panel} id="experiment" aria-labelledby="ai-lab-title">
    <div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id="ai-lab-title">{t('직접 바꿔보세요', 'Make it your experiment')}</h2></div><span className="ai-live-dot">{t('실시간 계산', 'Live calculation')}</span></div>
    <GuidedExperiments kind={kind} ready={ready} onLoad={guide => load(spec.presets[guide.preset].values)} />
    <fieldset className="ai-presets" disabled={!ready}><legend>{t('비교할 예제', 'Examples to compare')}</legend>{spec.presets.map((p, i) => <Button key={i} type="button" variant="outline" size="sm" onClick={() => load(p.values)}>{p.name[language]}</Button>)}</fieldset>
    <fieldset className="ai-parameters" disabled={!ready} onFocusCapture={() => setPlaying(false)}><legend>{t('모델 설정', 'Model settings')}</legend><div className="ai-two-fields" key={formEpoch}>{spec.controls.map((c, i) => isCarnot && i === 5
      ? <label className="ai-number-field" key={i}><span>{c.label[language]}</span><select className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground" value={values[i]} onChange={e => { const n = Number(e.target.value); if (n === 0 || n === 1) load(values.map((x, j) => j === i ? n : x)); }}><option value={0}>{t('정방향', 'Forward')}</option><option value={1}>{t('역방향', 'Reverse')}</option></select></label>
      : <NumberField key={i} integer={kind === 'gapped-spectrum-low-temperature-entropy' && i < 3} label={c.label[language]} min={c.min} max={c.max} step={c.step} value={values[i]} onChange={n => {
        if (kind === 'gapped-spectrum-low-temperature-entropy' && i < 3 && !Number.isInteger(n)) return;
        load(values.map((x, j) => j === i ? n : x));
      }} />)}</div></fieldset>
    <p className="ai-small">{t('숫자는 Enter 또는 입력란을 떠날 때 적용합니다. 잘못된 입력은 이전 설정을 유지하며 초기화하면 입력 초안도 지웁니다.', 'Apply a number with Enter or by leaving its field. Invalid drafts retain the previous setting; reset also clears drafts.')}</p>
    <p className="ai-logic-statement">{spec.note[language]}</p>
    {kind === 'gapped-spectrum-low-temperature-entropy' && <p className="ai-small">{t('복제 수와 두 축퇴도는 정수만 적용합니다.', 'Copies and both degeneracies accept whole numbers only.')}</p>}
    <div className="ai-lab-actions">{isCarnot && <>
      <Button type="button" variant="outline" size="sm" disabled={!ready || !active || ended || !validOrder} onClick={() => setPlaying(p => !p)}>{playing ? t('일시정지', 'Pause') : t('경로 재생', 'Play path')}</Button>
      <Button type="button" variant="outline" size="sm" disabled={!ready || ended || !validOrder} onClick={() => load(values.map((x, i) => i === 6 ? Math.min(1, x + .02) : x))}>{t('한 단계 진행', 'Step forward')}</Button>
    </>}<Button type="button" variant="outline" size="sm" disabled={!ready} onClick={() => load(initial)}>{t('모든 실험 설정 초기화', 'Reset all experiment settings')}</Button></div>
    {isCarnot && !motion.canAnimate && <p className="ai-small">{t('모션 설정에 따라 자동 재생을 멈췄습니다. 수동 단계는 사용할 수 있습니다.', 'Motion preferences pause automatic playback. Manual steps remain available.')}</p>}
    <ThermodynamicLawView kind={kind} values={values} language={language} />
    <p className="ai-small">{storage ? t('설정을 이 브라우저에 저장합니다.', 'Settings are saved in this browser.') : t('브라우저 저장을 사용할 수 없습니다.', 'Browser storage is unavailable.')}</p>
  </section>;
}
