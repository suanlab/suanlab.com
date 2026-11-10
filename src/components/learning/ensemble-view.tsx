'use client';
import { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { NumberField } from './lab-controls';
import type { BinaryObservation } from '@/lib/learning-binary-products';
import type { SpectrumQuantity } from '@/lib/learning-gapped-spectrum';
import type { EnsembleSuccess } from '@/lib/learning-ensemble-protocol';
import { ensemblePlotValue } from '@/lib/learning-ensembles';
type Text = (ko: string, en: string) => string;
type Q = (BinaryObservation & { status?: 'finite' | 'infinite' }) | null;
export const ensembleNumber = (v: number | null) => v === null ? '—' : v === 0 ? '0' : Math.abs(v) < .001 || Math.abs(v) >= 10000 ? v.toExponential(4) : Number(v.toPrecision(6)).toString();
export function ensembleQuantity(q: Q, t: Text): string {
  if (q === null) return t('미정', 'Undefined');
  if ('status' in q && q.status === 'infinite') return '∞';
  if (q.value !== null) return ensembleNumber(q.value);
  return `${q.sign > 0 ? '+' : '−'} · ${t('선형 범위 밖', 'outside linear range')}`;
}
export function ensembleChartIndices(length: number, selected: number[] = []) {
  if (!Number.isInteger(length) || length < 1) return [];
  const count = Math.min(65, length);
  return [...new Set([...Array.from({ length: count }, (_, i) => count === 1 ? 0 : Math.round((length - 1) * i / (count - 1))), ...selected.slice(0, 3).filter(i => Number.isInteger(i) && i >= 0 && i < length)])].sort((a, b) => a - b);
}
function Table({ title, headers, rows }: { title: string; headers: string[]; rows: (string | number)[][] }) {
  return <div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption style={{ textAlign: 'left', padding: '.5rem' }}>{title}</caption><thead><tr>{headers.map((h, i) => <th scope="col" key={i}>{h}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => j === 0 ? <th scope="row" key={j}>{cell}</th> : <td key={j} style={{ whiteSpace: 'normal', overflowWrap: 'anywhere', maxWidth: '18rem' }}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
function quantityStatus(q: Q, t: Text) {
  if (q === null) return t('미정', 'Undefined');
  if ('status' in q && q.status === 'infinite') return t('수학적 무한대', 'Mathematical infinity');
  return q.sign === 0 ? t('실제0', 'True zero') : q.value === null ? t('비영 · 선형 범위 밖', 'Nonzero; outside linear range') : t('반올림한 값', 'Rounded value');
}
function logStatus(q: Q, t: Text) {
  if (!q) return '—';
  const s = q as Partial<SpectrumQuantity>;
  return q.logMagnitude !== null ? ensembleNumber(q.logMagnitude) : s.logMagnitudeStatus === 'outside-number-range' ? t('로그도 범위 밖', 'Log also outside range') : '—';
}
function Readout({ rows, t, context }: { rows: [string, Q, string][]; t: Text; context?: string }) {
  return <><p className="ai-small">{t('미정·실제0·표시 범위 밖의 비영·무한대는 다릅니다. 로그는 표의 단위로 쓴 크기의 자연로그입니다. 두 번째 로그는 꼬리 크기의 보조 진단이며 정확한 등식의 증명이 아닙니다.', 'Undefined, true zero, nonzero outside the display range and infinity differ. Logs use magnitudes in the stated units. The second logarithm diagnoses tail scale; it does not prove exact identities.')}</p>
    <Table title={`${context ? context + ' · ' : ''}${t('관측값과 표현 상태', 'Observations and representation status')}`} headers={[t('관측', 'Observation'), t('값', 'Value'), t('단위', 'Unit'), t('상태', 'Status'), 'ln|q|', 'ln(−ln|q|)']} rows={rows.map(([label, q, unit]) => {
      const s = q as Partial<SpectrumQuantity> | null;
      const second = s?.logNegativeLogMagnitude;
      return [label, ensembleQuantity(q, t), unit, quantityStatus(q, t), logStatus(q, t), typeof second === 'number' ? ensembleNumber(second) : s?.logNegativeLogMagnitudeStatus === 'outside-number-range' ? t('범위 밖', 'Outside range') : '—'];
    })} /></>;
}
function Bars({ indices, series, t }: { indices: number[]; series: { name: string; color: string; values: Q[] }[]; t: Text }) {
  const maximum = Math.max(.000001, ...series.flatMap(s => s.values.map(ensemblePlotValue)).filter((v): v is number => v !== null));
  const step = 272 / Math.max(1, indices.length), labelStep = Math.max(1, Math.ceil(indices.length / 8));
  const title = t('이산 분포 비교', 'Discrete distribution comparison');
  const scrollHintId = useId();
  return <figure className="ai-foundation-figure" data-ensemble-plot><h3>{title}</h3><p id={scrollHintId} className="ai-small">{t('그림을 가로로 밀어 보세요. 키보드는 그림 영역에 초점을 두고 좌우 방향키를 사용하세요.', 'Swipe horizontally to explore the plot. With keyboard focus in the plot region, use the left and right arrow keys.')}</p><div className="ai-foundation-scroll" role="region" tabIndex={0} aria-label={title} aria-describedby={scrollHintId}>
    <svg viewBox="0 0 350 270" style={{ minWidth: 560 }} role="img" aria-label={t('가로축은 선택한 이산 여기·입자 수, 세로축은 각 수의 확률입니다. 물음표는 표시 범위 밖의 양수입니다.', 'The horizontal axis shows selected excitation or particle counts; the vertical axis gives their probabilities. Question marks indicate positive probabilities outside display range.')}>
      <path d="M54 18V218H330" fill="none" stroke="var(--learning-neutral)" />
      {[0, .5, 1].map(r => <text key={r} x="49" y={218 - 190 * r + 3} textAnchor="end" fill="currentColor" style={{ fontSize: 10 }}>{ensembleNumber(maximum * r)}</text>)}
      {series.map((s, k) => <g key={s.name}>{s.values.map((q, i) => {
        const v = ensemblePlotValue(q), width = Math.max(.5, step / (series.length + 1)), x = 56 + i * step + k * width;
        return <g key={i}><title>{`${s.name}(${indices[i]})=${ensembleQuantity(q, t)}; ln|q|=${logStatus(q, t)}`}</title>{v === null ? <text x={x} y="32" fill="currentColor" style={{ fontSize: 10 }}>?</text> : v === 0 ? <path d={`M${x} 218h${width}`} stroke={`var(--learning-${s.color})`} /> : <rect x={x} y={218 - 190 * v / maximum} width={width} height={190 * v / maximum} fill={`var(--learning-${s.color})`} fillOpacity={k ? .55 : .9} stroke={`var(--learning-${s.color})`} strokeDasharray={k ? '2 2' : undefined} />}</g>;
      })}</g>)}
      {indices.map((n, i) => i % labelStep === 0 || i === indices.length - 1 ? <text key={n} x={56 + i * step} y="236" fill="currentColor" style={{ fontSize: 10 }}>{n}</text> : null)}
      <text x="190" y="258" textAnchor="middle" fill="currentColor" style={{ fontSize: 10 }}>{t('여기·입자 수 · 확률', 'Excitation/particle count · probability')}</text>
    </svg></div><div className="ai-legend">{series.map(s => <span key={s.name} style={{ color: `var(--learning-${s.color})` }}>{s.name}</span>)}</div>
    <figcaption>{t('최대68개 레이블을 선택하며 막대를 재정규화하지 않습니다. 가로 간격은 선택 레이블 간격입니다. ?는 선형 범위 밖의 양수이며 실제0과 다릅니다. 전체 분포는 아래 페이지 표에서 확인하세요.', 'At most 68 labels are selected without renormalizing bars. Horizontal spacing separates selected labels. ? denotes a positive value outside linear range and differs from true zero. The paged table contains the full distribution.')}</figcaption></figure>;
}
function PagedTable<Row>({ title, headers, rows, renderRow, t }: { title: string; headers: string[]; rows: readonly Row[]; renderRow: (row: Row) => (string | number)[]; t: Text }) {
  const [page, setPage] = useState(0), last = Math.max(0, Math.ceil(rows.length / 32) - 1), actual = Math.min(page, last);
  return <><div className="ai-lab-actions"><Button type="button" variant="outline" disabled={actual === 0} onClick={() => setPage(actual - 1)}>{t('이전 분포', 'Previous distribution rows')}</Button><output aria-label={t('분포 페이지', 'Distribution page')}>{actual + 1}/{last + 1}</output><Button type="button" variant="outline" disabled={actual === last} onClick={() => setPage(actual + 1)}>{t('다음 분포', 'Next distribution rows')}</Button></div>
    <Table title={title} headers={headers} rows={rows.slice(actual * 32, actual * 32 + 32).map(renderRow)} /></>;
}
function Reservoir({ r, t, context }: { r: Extract<EnsembleSuccess, { kind: 'finite-binary-reservoir-canonical-comparison' }>['result']; t: Text; context?: string }) {
  const p = r.parameters, indices = ensembleChartIndices(r.distribution.length, [Math.floor(r.meanExcitations.value ?? 0)]);
  return <><p>n={p.subsystemSites} · b={p.reservoirSites} · K={p.totalExcitations} · K/(n+b)={ensembleNumber(p.totalExcitations / p.totalSites)}</p>
    <p>{t('P는 전체 K를 고정한 균등 껍질의 부분계 분포입니다. Q는 같은 여기 밀도의 독립 정준 준비입니다. 평균이 같아도 분산과 지지집합은 다를 수 있습니다.', 'P is the subsystem law in a uniform shell at fixed total K. Q is an independent canonical preparation with the same excitation density. Equal means need not imply equal variance or support.')}</p>
    <Bars indices={indices} series={[{ name: 'P(A|K)', color: 'blue', values: indices.map(i => r.distribution[i].microcanonical) }, { name: 'Q(A)', color: 'orange', values: indices.map(i => r.distribution[i].canonical) }]} t={t} />
    <Readout t={t} context={context} rows={[[t('평균 여기 수', 'Mean excitation count'), r.meanExcitations, '1'], ['Var P', r.varianceExcitations, '1'], ['Var Q', r.canonicalVariance, '1'], ['TV(P,Q)', r.totalVariation, '1'], ['H(P)', r.entropyNats, 'nat'], ['D(P||Q)', r.microToCanonicalKlNats, 'nat'], ['D(Q||P)', r.canonicalToMicroKlNats, 'nat']]} />
    <p>{t('정확한 분포의 일치', 'Exact equality of the distributions')}: {r.exactlyEqual ? t('예', 'Yes') : t('아니오', 'No')}. {t('전체 계수와 공통 분모는 반올림하지 않은 정수입니다.', 'Counts and common denominators are unrounded integers.')}</p>
    <p style={{ overflowWrap: 'anywhere' }}>P: W={r.wholeShellCount}<br />Q: D={r.canonicalCommonDenominator}</p>
    <PagedTable title={`${context ? context + ' · ' : ''}${t('분포와 정확한 계수 · 32행씩', 'Distribution and exact counts · 32 rows per page')}`} headers={['A', 'W(A)', 'P(A|K)', 'Q numerator', 'Q(A)']} rows={r.distribution} renderRow={row => [row.excitations, row.compatibleMicrostates, ensembleQuantity(row.microcanonical, t), row.canonicalNumerator, ensembleQuantity(row.canonical, t)]} t={t} />
  </>;
}
function Canonical({ r, t, context }: { r: Extract<EnsembleSuccess, { kind: 'canonical-binary-energy-fluctuations' }>['result']; t: Text; context?: string }) {
  const p = r.parameters, indices = ensembleChartIndices(r.distribution.length, [Math.floor(r.meanExcitations.value ?? 0)]);
  return <><p>n={p.sites} · ε={ensembleNumber(p.energyGap)} J · T={ensembleNumber(p.temperature)} K</p>
    <p>{t('각 자리는 독립이고 E=εK입니다. C=Var(E)/(kBT²)는 유한한 비영 온도에서의 응답이며, Var(K)와 단위가 다릅니다. 에너지 간격0에서는 에너지 요동이0이어도 배치 불확실성이 남습니다.', 'Sites are independent and E=epsilon K. C=Var(E)/(kB T²) is a response at finite nonzero temperature, with different units from Var(K). At zero gap, energy fluctuations vanish while pattern uncertainty remains.')}</p>
    <Bars indices={indices} series={[{ name: 'P(K)', color: 'blue', values: indices.map(i => r.distribution[i].probability) }]} t={t} />
    <Readout t={t} context={context} rows={[[t('한 자리 여기 확률', 'One-site excitation probability'), r.oneProbability, '1'], [t('한 자리 바닥 확률', 'One-site ground probability'), r.zeroProbability, '1'], ['t=ε/(kBT)', r.scaledEnergyGap, '1'], ['E[K]', r.meanExcitations, '1'], ['Var(K)', r.excitationVariance, '1'], ['U', r.meanEnergy, 'J'], ['Var(E)', r.energyVariance, 'J²'], ['S', r.entropy, 'J/K'], ['S/kB', r.entropyInKB, '1'], ['C', r.heatCapacity, 'J/K'], ['C/kB', r.heatCapacityInKB, '1'], ['ln Z', r.logPartition, '1'], ['σK/E[K]', r.relativeCountFluctuation, '1'], ['σE/U', r.relativeEnergyFluctuation, '1']]} />
    <p style={{ overflowWrap: 'anywhere' }}>{t('이름 붙은 전체 배치 수', 'Total labeled patterns')}: {r.totalMicrostates}. {t('T=0의 값은 양의 쪽 극한이며, ε=0이면 모든 배치가 같은 에너지입니다. 평균이0인 상대 요동은 미정입니다.', 'T=0 uses the positive-side limit. At epsilon=0 every pattern has the same energy. Relative fluctuations with zero mean are undefined.')}</p>
    <PagedTable title={`${context ? context + ' · ' : ''}${t('정준 분포 · 32행씩', 'Canonical distribution · 32 rows per page')}`} headers={['K', 'C(n,K)', 'P(K)', 'ln P(K)']} rows={r.distribution} renderRow={row => [row.excitations, row.multiplicity, ensembleQuantity(row.probability, t), logStatus(row.probability, t)]} t={t} />
  </>;
}
function PatternInspection({ r, t }: { r: Extract<EnsembleSuccess, { kind: 'grand-canonical-finite-occupancy-fluctuations' }>['result']; t: Text }) {
  const [chosen, setChosen] = useState(Math.floor(r.meanParticles.value ?? 0));
  const count = Math.min(r.parameters.sites, chosen), row = r.distribution[count];
  const patterns = r.parameters.sites <= 6 ? Array.from({ length: 2 ** r.parameters.sites }, (_, i) => r.parameters.sites === 0 ? '' : i.toString(2).padStart(r.parameters.sites, '0')).filter(bits => [...bits].filter(bit => bit === '1').length === count) : null;
  return <section aria-label={t('입자 수를 고정한 배치 검사', 'Inspect patterns at a fixed particle count')}>
    <h3>{t('입자 수와 조건부 배치를 구분하기', 'Distinguish counts from conditional patterns')}</h3>
    <NumberField resetKey={r} label={t('검사할 입자 수 N', 'Particle count N to inspect')} value={count} min={0} max={r.parameters.sites} integer step={1} onChange={n => { if (Number.isInteger(n)) setChosen(n); }} />
    <p style={{ overflowWrap: 'anywhere' }}>N={count} · P(N)={ensembleQuantity(row.probability, t)} · C(M,N)={row.multiplicity} · P({t('배치', 'pattern')}|N)={ensembleQuantity(row.conditionalPatternProbability, t)}</p>
    {row.conditionalPatternProbability === null ? <p>{t('선택한 수의 실제 확률이0이므로 그 사건을 조건으로 한 배치 확률은 미정입니다.', 'The chosen count has true zero probability, so its conditional pattern probability is undefined.')}</p> : <p>{t('같은 수의 이름 붙은 배치들은 조건부로 균등합니다. 이 조건부 확률을 P(N)와 혼동하지 마세요.', 'Labeled patterns with the same count are conditionally uniform. This conditional probability differs from P(N).')}</p>}
    {patterns ? <p style={{ overflowWrap: 'anywhere' }}>{t('배치 레이블', 'Pattern labels')}: {patterns.map(bits => bits || t('(빈 문자열)', '(empty string)')).join(' · ')}</p> : <p className="ai-small">{t('배치는 여섯 자리까지 열거합니다. 더 큰 경우에는 위의 정확한 다중도를 사용하세요.', 'Patterns are enumerated for up to six sites. For larger systems use the exact multiplicity above.')}</p>}
  </section>;
}
function Grand({ r, t, context }: { r: Extract<EnsembleSuccess, { kind: 'grand-canonical-finite-occupancy-fluctuations' }>['result']; t: Text; context?: string }) {
  const p = r.parameters, indices = ensembleChartIndices(r.distribution.length, [Math.floor(r.meanParticles.value ?? 0)]);
  const infinite = r.logGrandPartitionStatus === 'infinite-zero-temperature-limit';
  return <><p>M={p.sites} · ε={ensembleNumber(p.energyGap)} J · μ={ensembleNumber(p.chemicalPotential)} J · T={ensembleNumber(p.temperature)} K</p>
    <p>{t('이름 붙은 각 자리에 입자가0개 또는1개 들어가는 고전적 배제 모형입니다. P(N)는 입자 수 확률이고 P(배치|N)=1/C(M,N)는 그 수를 조건으로 한 배치 확률입니다. 페르미 기체 전체를 모형화하지 않습니다.', 'This classical exclusion model allows zero or one particle at each labeled site. P(N) is a count probability; P(pattern|N)=1/C(M,N) is conditional on that count. This is not a full Fermi-gas model.')}</p>
    <Bars indices={indices} series={[{ name: 'P(N)', color: 'green', values: indices.map(i => r.distribution[i].probability) }]} t={t} />
    {infinite && <p data-ensemble-infinity>{t('온도0의 가득 찬 극한: ln Ξ=+∞이지만 Ω는 유한할 수 있습니다. 이는 선형 표시 범위 밖의 유한값과 다릅니다.', 'Fully occupied zero-temperature limit: ln Xi=+∞ while Omega may remain finite. This differs from a finite value outside linear display range.')}</p>}
    {r.susceptibilityStatus === 'undefined-at-zero-temperature-threshold' && <p data-ensemble-threshold>{t('고정 μ=ε, T→0+ 경로에는 요동이 남습니다. 온도0 임계점의 유한한 μ 미분은 정의되지 않으며0이 아닙니다.', 'Fluctuations remain on the fixed mu=epsilon, T→0+ path. A finite chemical derivative at the zero-temperature threshold is undefined, rather than zero.')}</p>}
    <Readout t={t} context={context} rows={[[t('점유 확률', 'Occupancy probability'), r.oneProbability, '1'], [t('빈 자리 확률', 'Hole probability'), r.zeroProbability, '1'], ['ε−μ', r.occupationCostMinusChemicalPotential, 'J'], ['y=(μ−ε)/(kBT)', r.scaledChemicalBias, '1'], ['E[N]', r.meanParticles, '1'], [t('평균 빈 자리', 'Mean holes'), r.meanHoles, '1'], ['Var(N)', r.numberVariance, '1'], ['U', r.meanEnergy, 'J'], ['Var(E)', r.energyVariance, 'J²'], ['Cov(E,N)', r.energyNumberCovariance, 'J'], ['S', r.entropy, 'J/K'], ['S/kB', r.entropyInKB, '1'], ['∂E[N]/∂μ', r.numberSusceptibility, '1/J'], ['σN/E[N]', r.relativeNumberFluctuation, '1'], ['Ω', r.grandPotential, 'J'], ['ln Ξ', infinite ? { value: null, sign: 1, logMagnitude: null, status: 'infinite' } : r.logGrandPartition, '1']]} />
    <PatternInspection r={r} t={t} />
    <p>{t('응답은 유한한 비영 온도에서 Var(N)/(kBT)입니다. 음의 온도에서는 부호도 음수입니다. T=0은 μ를 고정한 양의 쪽 극한이며 μ가 온도와 함께 변하는 경로는 다른 극한을 줄 수 있습니다.', 'At finite nonzero temperature the response is Var(N)/(kBT), including its negative sign at negative temperature. T=0 holds mu fixed in the positive-side limit. Varying mu with temperature can give a different joint limit.')}</p>
    <PagedTable title={`${context ? context + ' · ' : ''}${t('입자 수와 조건부 배치 · 32행씩', 'Particle counts and conditional patterns · 32 rows per page')}`} headers={['N', 'C(M,N)', 'P(N)', 'ln P(N)', 'P(pattern|N)']} rows={r.distribution} renderRow={row => [row.particles, row.multiplicity, ensembleQuantity(row.probability, t), logStatus(row.probability, t), ensembleQuantity(row.conditionalPatternProbability, t)]} t={t} />
  </>;
}
export function EnsembleResultView({ response, language, context }: { response: EnsembleSuccess; language: 'ko' | 'en'; context?: string }) {
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  return <div data-ensemble-result={response.kind}>{response.kind === 'finite-binary-reservoir-canonical-comparison' ? <Reservoir r={response.result} t={t} context={context} /> : response.kind === 'canonical-binary-energy-fluctuations' ? <Canonical r={response.result} t={t} context={context} /> : <Grand r={response.result} t={t} context={context} />}</div>;
}
