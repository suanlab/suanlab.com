'use client';
import { useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { partitionConceptInputs } from '@/lib/learning-partition-concepts';
import type { PartitionKind } from '@/lib/learning-partition';
import type { PartitionSuccess } from '@/lib/learning-partition-protocol';
import type { PartitionObservation } from '@/lib/learning-partition-exact-core';
import { partitionNumber, partitionQuantity, partitionQuantityStatus } from './partition-view';
import { usePartitionSeries } from './use-partition-series';
type Text = (ko: string, en: string) => string;
function ConceptTable({ title, headers, rows }: { title: string; headers: string[]; rows: (string | number)[][] }) {
  return <div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption style={{ textAlign: 'left', padding: '.5rem' }}>{title}</caption><thead><tr>{headers.map((h, i) => <th scope="col" key={i}>{h}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((v, j) => j === 0 ? <th scope="row" key={j}>{v}</th> : <td key={j} style={{ whiteSpace: typeof v === 'number' || /^[+−-]?(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?|∞)$/i.test(v) ? 'nowrap' : 'normal', overflowWrap: 'normal', maxWidth: '18rem' }}>{v}</td>)}</tr>)}</tbody></table></div>;
}
export function PartitionConceptResult({ responses, step, language }: { responses: readonly PartitionSuccess[]; step: number; language: 'ko' | 'en' }) {
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  if (!responses.length) return null;
  const kind = responses[0].kind, title = t('고정 개념 예제의 비교', 'Comparison of fixed concept preparations');
  const rows: (string | number)[][] = [], bars: { label: string; q: PartitionObservation | null; color: string }[] = [];
  const q = (value: PartitionObservation | null) => partitionQuantity(value, t);
  let headers: string[] = [], axis = t('확률', 'Probability');
  responses.forEach((response, i) => {
    const number = i + 1;
    if (response.kind === 'finite-spectrum-partition-cumulant-generators') {
      const r = response.result; headers = [t('준비', 'Case'), 'c (J)', 'T (K)', 'U (J)', 'Var(E) (J²)', 'κ₃(E) (J³)', 'S/kB', 'ln Z'];
      rows.push([number, partitionNumber(r.parameters.energyOffset), partitionNumber(r.parameters.temperature), ...[r.meanEnergy, r.energyVariance, r.thirdEnergyCumulant, r.entropyInKB, r.logPartition].map(q)]);
      r.states.forEach(s => bars.push({ label: `${number}:${s.index}`, q: s.probability, color: i % 2 ? 'orange' : 'blue' }));
    } else if (response.kind === 'coupled-binary-sites-partition-factorization') {
      const r = response.result; headers = [t('준비', 'Case'), 'J (J)', 'h (J)', 'T (K)', t('지원', 'Support'), 'E[A]', 'E[M]', 'Cov(σ₁,σ₂)', 'Δ', 'I (nat)', 'Cov(A,M)', '∂h E[A] (1/J)'];
      rows.push([number, partitionNumber(r.parameters.couplingEnergy), partitionNumber(r.parameters.fieldEnergy), partitionNumber(r.parameters.temperature), r.parameters.support === 'full-product' ? t('전체', 'Full') : t('합0', 'Zero sum'), ...[r.meanAlignment, r.meanMagnetization, r.spinCovariance, r.logFactorizationDefect, r.mutualInformation, r.sourceHessian[0][1], r.responses.alignmentToField.observation].map(q)]);
      r.states.forEach(s => bars.push({ label: `${number}:${s.spins.join(',')}`, q: s.probability, color: i % 2 ? 'orange' : 'blue' }));
    } else {
      const r = response.result, infinite = r.infinite.quantities;
      headers = [t('준비', 'Case'), 'ε (J)', 'T (K)', 'L', t('무한 준비', 'Infinite preparation'), 'δ', 'E[n]L', 'E[n]∞', 'C_L/kB', 'C_∞/kB', 'D(pL||p∞)', 'D(p∞||pL)'];
      rows.push([number, partitionNumber(r.parameters.gapEnergy), partitionNumber(r.parameters.temperature), r.parameters.cutoff, r.infinite.status === 'not-normalizable' ? t('정규화 불가', 'Not normalizable') : r.infinite.status === 'positive-side-zero-temperature-limit' ? t('바닥 극한', 'Ground limit') : t('정규화 가능', 'Normalizable'), ...[r.comparison?.missingMass ?? null, r.finite.meanIndex, infinite?.meanIndex ?? null, r.finite.heatCapacityInKB, infinite?.heatCapacityInKB ?? null, r.comparison?.forwardKL ?? null, r.comparison?.reverseKL ?? null].map(q)]);
      if (step === 2) {
        axis = 'C/kB'; bars.push({ label: `${number}:L`, q: r.finite.heatCapacityInKB, color: 'blue' }, { label: `${number}:∞`, q: infinite?.heatCapacityInKB ?? null, color: 'orange' });
      } else r.samples.filter(s => s.n <= 3).forEach(s => {
        bars.push({ label: `${number}:L:${s.n}`, q: s.finiteProbability, color: 'blue' });
        if (infinite) bars.push({ label: `${number}:∞:${s.n}`, q: s.infiniteProbability, color: 'orange' });
      });
    }
  });
  return <div data-partition-concept-result={kind}>
    <svg viewBox="0 0 360 250" role="img" aria-label={`${title} · ${axis}`}>
      <path d="M52 20V200H344" fill="none" stroke="var(--learning-neutral)" />
      {[0, .5, 1].map(v => <text key={v} x="47" y={203 - 170 * v} textAnchor="end" fontSize="10" fill="currentColor">{v}</text>)}
      {bars.map((bar, i) => {
        const x = 60 + i * 277 / bars.length, width = Math.min(35, 210 / bars.length), value = bar.q?.value;
        const height = typeof value === 'number' && value > 0 ? Math.max(2, 170 * Math.min(1, value)) : 0;
        return <g key={i}><title>{`${bar.label}: ${q(bar.q)}; ${partitionQuantityStatus(bar.q, t)}`}</title>{bar.q === null || bar.q.resolution === 'unresolved-at-precision-ceiling' ? <text x={x} y="190" fontSize="10" fill="currentColor">?</text> : value === null ? <circle cx={x + width / 2} cy="197" r="2" fill="none" stroke={`var(--learning-${bar.color})`} /> : value === 0 ? <path d={`M${x} 200h${width}`} stroke={`var(--learning-${bar.color})`} /> : <rect x={x} y={200 - height} width={width} height={height} fill={`var(--learning-${bar.color})`} />}
          <text x={x + width / 2} y="219" fontSize="9" textAnchor="middle" fill="currentColor">{i + 1}</text></g>;
      })}<text x="195" y="242" textAnchor="middle" fontSize="11" fill="currentColor">{axis}</text>
    </svg>
    <p className="ai-small">{t('번호는 아래 막대 표 순서입니다. 작은 양수의 최소2px 표식은 실제 확률·열용량 크기가 아닙니다. 빈 원은 표시 범위 밖의 양수,?는 미정·미해결입니다. 보간한 시간 변화나 동역학이 아닙니다.', 'Numbers follow the bar table below. A 2px marker for a tiny positive is not its actual probability or capacity. Open circles indicate positives outside display range; ? means undefined or unresolved. This is not interpolated time evolution or dynamics.')}</p>
    <ConceptTable title={title} headers={headers} rows={rows} />
    <ConceptTable title={t('막대와 정확한 표현 상태', 'Bars and numerical representation status')} headers={['#', t('준비:상태', 'Case:state'), axis, t('상태', 'Status')]} rows={bars.map((b, i) => [i + 1, b.label, q(b.q), partitionQuantityStatus(b.q, t)])} />
    {kind === 'finite-spectrum-partition-cumulant-generators' && step === 1 && <p>{t('표의 세 번째 cumulant는 양의 온도에서도 음수일 수 있습니다. 평균·분산·세 번째 cumulant는 ε·ε²·ε³의 다른 척도를 가집니다.', 'The third cumulant can be negative at positive temperature. Mean, variance and third cumulant carry different scales epsilon, epsilon² and epsilon³.')}</p>}
    {kind === 'finite-spectrum-partition-cumulant-generators' && step === 2 && <p>{t('준비1·2는 간격과 확률이 같지만 공통 기준이 다릅니다. 준비3은 입력 전에 간격이 소실되어 실제로 다른 평평한 스펙트럼입니다.', 'Cases 1 and 2 share gaps and probabilities but use different offsets. Case 3 lost its gaps before input and is a different, flat spectrum.')}</p>}
    {kind === 'coupled-binary-sites-partition-factorization' && step === 2 && <p>{t('Δ는 비교 분배함수의 로그 차이,상호정보량은 실제 주변분포 곱에 대한 KL입니다. 약0.22314355와0.19274476은 같지 않습니다.', 'Delta is a log difference of comparison normalizers; mutual information is KL against the actual marginal product. About 0.22314355 and 0.19274476 are different values.')}</p>}
    {kind === 'coupled-binary-sites-partition-factorization' && step === 3 && <p>{t('준비1의 Cov(A,M)=0에도 온도0의 정렬 미분은 없습니다. 준비2는 세 상태 바닥 교차입니다. source 공분산의 유한 극한을 물리적 미분으로 대체하지 마세요.', 'Case 1 has Cov(A,M)=0 but no zero-temperature alignment derivative. Case 2 is a three-ground-state crossing. A finite source covariance limit is not a physical derivative.')}</p>}
    {kind === 'infinite-energy-ladder-normalization-cutoff' && step === 2 && <p>{t('같은 x=0.001에서 절단2와4096을 비교합니다. 막대는 같은 단위의 C/kB이며,고정 절단의 높은 온도 극한0과 무한 사다리의1을 구별합니다.', 'Compare cutoffs 2 and 4096 at the same x=0.001. Bars share units C/kB; distinguish the fixed-cutoff high-temperature limit zero from the infinite-ladder limit one.')}</p>}
  </div>;
}
export function PartitionConceptView({ kind, step, language }: { kind: PartitionKind; step: number; language: 'ko' | 'en' }) {
  const inputs = useMemo(() => partitionConceptInputs(kind, step), [kind, step]), calculation = usePartitionSeries(inputs), ko = language === 'ko';
  return <div data-partition-concept={kind} data-partition-step={step} data-partition-concept-state={calculation.status}>
    {calculation.responses.length ? <PartitionConceptResult responses={calculation.responses} step={step} language={language} /> : <p role="status">{calculation.status === 'error' ? ko ? '고정 예제를 계산하지 못했습니다. 다시 시도할 수 있습니다.' : 'The fixed example could not be calculated. You can retry.' : ko ? '고정 예제를 계산하고 있습니다.' : 'Calculating the fixed example.'}</p>}
    {calculation.status === 'error' && <Button variant="outline" type="button" onClick={calculation.retry}>{ko ? '다시 계산' : 'Retry calculation'}</Button>}
  </div>;
}
