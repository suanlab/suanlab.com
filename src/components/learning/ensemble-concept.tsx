'use client';
import { useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { ensembleConceptInputs } from '@/lib/learning-ensemble-concepts';
import type { EnsembleKind } from '@/lib/learning-ensembles';
import type { EnsembleSuccess } from '@/lib/learning-ensemble-protocol';
import type { BinaryObservation } from '@/lib/learning-binary-products';
import { ensembleNumber, ensembleQuantity } from './ensemble-view';
import { useEnsembleSeries } from './use-ensemble-series';
type Text = (ko: string, en: string) => string;
export function EnsembleConceptResult({ responses, step, language }: { responses: readonly EnsembleSuccess[]; step: number; language: 'ko' | 'en' }) {
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  if (!responses.length) return null;
  const first = responses[0], title = t('고정 개념 예제 · 비교 관측', 'Fixed concept example · comparison observations');
  const rows: (string | number)[][] = [];
  const bars: { label: string; q: BinaryObservation | null }[] = [];
  let headers: string[], axis: string;
  if (first.kind === 'finite-binary-reservoir-canonical-comparison') {
    headers = ['b', 'K', 'E[A]', 'Var P', 'Var Q', 'TV', 'D(P||Q)', 'D(Q||P)']; axis = 'TV(P,Q)';
    for (const response of responses) if (response.kind === first.kind) {
      const r = response.result;
      rows.push([r.parameters.reservoirSites, r.parameters.totalExcitations, ...[r.meanExcitations, r.varianceExcitations, r.canonicalVariance, r.totalVariation, r.microToCanonicalKlNats, r.canonicalToMicroKlNats].map(q => ensembleQuantity(q, t))]);
      bars.push({ label: `b=${r.parameters.reservoirSites}`, q: r.totalVariation });
    }
    if (step < 3) return <div data-ensemble-concept-result={first.kind}>
      <svg viewBox="0 0 360 235" role="img" aria-label={t('여섯 전체 배치: 두 자리 부분계와 두 자리 저장고. 여기 수별 계수는1·4·1입니다.', 'Six full patterns: a two-site subsystem and a two-site reservoir. Count multiplicities are 1,4,1.')}>
        {['1100','1010','1001','0110','0101','0011'].map((bits, i) => <g key={bits}>{[...bits].map((bit, j) => <g key={j}><rect x={34 + j * 48 + (j >= 2 ? 24 : 0)} y={8 + i * 36} width="36" height="27" rx="4" fill={j < 2 ? 'var(--learning-blue)' : 'var(--learning-orange)'} fillOpacity={bit === '1' ? '.55' : '.08'} stroke={j < 2 ? 'var(--learning-blue)' : 'var(--learning-orange)'} /><text x={52 + j * 48 + (j >= 2 ? 24 : 0)} y={27 + i * 36} textAnchor="middle" fill="currentColor" fontSize="14">{bit}</text></g>)}<text x="310" y={27 + i * 36} fill="currentColor" fontSize="13">A={[...bits.slice(0, 2)].filter(v => v === '1').length}</text></g>)}
      </svg><p>{t('전체 배치마다1/6입니다. 부분계10은 두 저장고 배치와 짝지어져2/6, 여기 수 A=1은 네 전체 배치를 모아4/6입니다.', 'Each full pattern has probability 1/6. Subsystem pattern 10 pairs with two reservoir patterns, giving 2/6; count A=1 collects four full patterns, giving 4/6.')}</p>
      <ConceptTable title={title} headers={['A', 'W(A)', 'P(A|K)', 'Q(A)']} rows={first.result.distribution.map(r => [r.excitations, r.compatibleMicrostates, ensembleQuantity(r.microcanonical, t), ensembleQuantity(r.canonical, t)])} />
      <ConceptTable title={`${title} · ${t('모멘트와 거리', 'moments and distances')}`} headers={headers} rows={rows} />
    </div>;
  } else if (first.kind === 'canonical-binary-energy-fluctuations') {
    headers = ['n', 'T (K)', 'r', 'E[K]', 'Var(K)', 'C/kB', 'σK/E[K]']; axis = step === 3 ? 'σK/E[K]' : 'C/kB';
    for (const response of responses) if (response.kind === first.kind) {
      const r = response.result;
      rows.push([r.parameters.sites, ensembleNumber(r.parameters.temperature), ...[r.oneProbability, r.meanExcitations, r.excitationVariance, r.heatCapacityInKB, r.relativeCountFluctuation].map(q => ensembleQuantity(q, t))]);
      bars.push({ label: step === 3 ? `n=${r.parameters.sites}` : `T=${r.parameters.temperature}`, q: step === 3 ? r.relativeCountFluctuation : r.heatCapacityInKB });
    }
  } else {
    headers = ['μ (J)', 'T (K)', 'r', 'E[N]', 'Var(N)', 'χ (1/J)', 'Ω (J)', 'ln Ξ']; axis = step === 2 ? 'χ (1/J)' : 'E[N]';
    for (const response of responses) if (response.kind === first.kind) {
      const r = response.result;
      rows.push([ensembleNumber(r.parameters.chemicalPotential), ensembleNumber(r.parameters.temperature), ...[r.oneProbability, r.meanParticles, r.numberVariance, r.numberSusceptibility, r.grandPotential].map(q => ensembleQuantity(q, t)), r.logGrandPartitionStatus === 'infinite-zero-temperature-limit' ? '∞' : ensembleQuantity(r.logGrandPartition, t)]);
      bars.push({ label: `μ=${ensembleNumber(r.parameters.chemicalPotential)}`, q: step === 2 ? r.numberSusceptibility : r.meanParticles });
    }
  }
  if (first.kind === 'canonical-binary-energy-fluctuations' && step < 2) {
    bars.length = 0;
    if (step === 0) { axis = 'p(site)'; bars.push({ label: '0', q: first.result.zeroProbability }, { label: '1', q: first.result.oneProbability }); }
    else { axis = 'P(K)'; for (const row of first.result.distribution) bars.push({ label: `K=${row.excitations}`, q: row.probability }); }
  }
  if (first.kind === 'grand-canonical-finite-occupancy-fluctuations' && step < 2) {
    bars.length = 0; axis = 'P(N)'; for (const row of first.result.distribution) bars.push({ label: `N=${row.particles}`, q: row.probability });
  }
  const max = Math.max(.000001, ...bars.map(b => b.q?.value ?? 0));
  return <div data-ensemble-concept-result={first.kind}>
    <svg viewBox="0 0 360 240" role="img" aria-label={`${title} · ${axis}`}>
      <path d="M74 20V192H346" fill="none" stroke="var(--learning-neutral)" />
      {[0, .5, 1].map(v => <text key={v} x="70" y={195 - 160 * v} textAnchor="end" fill="currentColor" fontSize="10">{ensembleNumber(v * max)}</text>)}
      {bars.map((b, i) => { const x = 82 + i * (253 / bars.length), v = b.q?.value; return <g key={i}><title>{`${b.label}: ${ensembleQuantity(b.q, t)}`}</title>{v === null || v === undefined ? <text x={x} y="45" fill="currentColor">?</text> : <rect x={x} y={192 - 160 * v / max} width={Math.min(45, 180 / bars.length)} height={160 * v / max} fill="var(--learning-blue)" />}
        <text x={x} y="211" fill="currentColor" fontSize="9">{bars.length > 4 ? i + 1 : b.label}</text></g>; })}
      <text x="195" y="234" textAnchor="middle" fill="currentColor" fontSize="12">{axis}</text>
    </svg>
    <p className="ai-small">{step < 2 ? t('같은 고정 준비의 자리 상태 또는 여기·입자 수 확률입니다. 아래 분포 표와 함께 읽으세요.', 'These are site-state or excitation/particle-count probabilities in the same fixed preparation. Read them with the distribution table below.') : t('막대는 아래 표와 같은 순서의 고정 준비를 비교합니다. 점 사이를 보간한 응답 곡선이나 시간 변화가 아닙니다. ?는 미정 또는 표시 범위 밖이며0으로 대체하지 않습니다.', 'Bars compare fixed preparations in table order. They are not an interpolated response curve or time evolution. ? marks undefined or out-of-range values and is not replaced by zero.')}</p>
    <ConceptTable title={title} headers={headers} rows={rows} />
    {first.kind === 'canonical-binary-energy-fluctuations' && step < 2 && <ConceptTable title={t('고정 개념 예제 · 정규화한 이항 분포', 'Fixed concept example · normalized binomial law')} headers={['K', 'C(n,K)', 'P(K)']} rows={first.result.distribution.map(r => [r.excitations, r.multiplicity, ensembleQuantity(r.probability, t)])} />}
    {first.kind === 'grand-canonical-finite-occupancy-fluctuations' && step < 2 && <ConceptTable title={t('고정 개념 예제 · 수와 조건부 배치', 'Fixed concept example · counts and conditional patterns')} headers={['N', 'C(M,N)', 'P(N)', 'P(pattern|N)']} rows={first.result.distribution.map(r => [r.particles, r.multiplicity, ensembleQuantity(r.probability, t), ensembleQuantity(r.conditionalPatternProbability, t)])} />}
  </div>;
}
function ConceptTable({ title, headers, rows }: { title: string; headers: string[]; rows: (string | number)[][] }) {
  return <div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption style={{ textAlign: 'left', padding: '.5rem' }}>{title}</caption><thead><tr>{headers.map((h, i) => <th scope="col" key={i}>{h}</th>)}</tr></thead><tbody>{rows.map((r, i) => <tr key={i}>{r.map((v, j) => j === 0 ? <th scope="row" key={j}>{v}</th> : <td key={j}>{v}</td>)}</tr>)}</tbody></table></div>;
}
export function EnsembleConceptView({ kind, step, language }: { kind: EnsembleKind; step: number; language: 'ko' | 'en' }) {
  const inputs = useMemo(() => ensembleConceptInputs(kind, step), [kind, step]), calculation = useEnsembleSeries(inputs), ko = language === 'ko';
  return <div data-ensemble-concept={kind} data-ensemble-step={step} data-ensemble-concept-state={calculation.status}>
    {calculation.responses.length ? <EnsembleConceptResult responses={calculation.responses} step={step} language={language} /> : <p role="status">{calculation.status === 'error' ? (ko ? '고정 예제를 계산하지 못했습니다. 다시 시도할 수 있습니다.' : 'The fixed example could not be calculated. You can retry.') : (ko ? '고정 예제를 계산하고 있습니다.' : 'Calculating the fixed example.')}</p>}
    {calculation.status === 'error' && <Button variant="outline" type="button" onClick={calculation.retry}>{ko ? '다시 계산' : 'Retry calculation'}</Button>}
  </div>;
}
