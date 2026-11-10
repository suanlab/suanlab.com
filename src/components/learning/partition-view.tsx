'use client';
import { useId, useState, type ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import type { PartitionSuccess } from '@/lib/learning-partition-protocol';
import type { PartitionObservation } from '@/lib/learning-partition-exact-core';
type Text = (ko: string, en: string) => string;
type Q = PartitionObservation | null;
export const partitionNumber = (n: number | null): string => n === null ? '—' : n === 0 ? '0' : Math.abs(n) < .001 || Math.abs(n) >= 10000 ? n.toExponential(4) : Number(n.toPrecision(6)).toString();
export function partitionQuantityStatus(q: Q, t: Text): string {
  if (q === null) return t('정의되지 않음', 'Undefined');
  if (q.resolution === 'mathematical-infinity') return t('수학적 무한대', 'Mathematical infinity');
  if (q.resolution === 'unresolved-at-precision-ceiling') return `${t('정밀도 상한 미해결', 'Unresolved at precision ceiling')}${q.sign === 1 ? t(' · 양수 확인', ' · positive sign established') : q.sign === -1 ? t(' · 음수 확인', ' · negative sign established') : ''}`;
  if (q.resolution === 'structural-zero') return t('정확한0', 'Exact zero');
  const size = q.value === null ? t('비영 · 표시 범위 밖', 'Nonzero · outside display range') : t('반올림한 값', 'Rounded value');
  return q.resolution === 'log-tail-approximation' ? `${size} · ${t('로그 꼬리 근사', 'log-tail approximation')}` : size;
}
export function partitionQuantity(q: Q, t: Text): string {
  if (q === null) return t('미정', 'Undefined');
  if (q.resolution === 'mathematical-infinity') return q.sign === -1 ? '−∞' : '+∞';
  if (q.resolution === 'unresolved-at-precision-ceiling') return `${q.sign === 1 ? '+' : q.sign === -1 ? '−' : '?'} · ${t('미해결', 'unresolved')}`;
  if (q.value !== null) return partitionNumber(q.value);
  return `${q.sign === 1 ? '+' : q.sign === -1 ? '−' : '?'} · ${t('표시 범위 밖', 'outside display range')}`;
}
const logarithm = (q: Q, t: Text) => q?.logMagnitude !== null && q?.logMagnitude !== undefined ? partitionNumber(q.logMagnitude)
  : q?.logResolution === 'outside-number-range' ? t('로그도 범위 밖', 'Log also outside range') : '—';
function Table({ title, headers, rows }: { title: string; headers: string[]; rows: (number | string)[][] }) {
  return <div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption style={{ textAlign: 'left', padding: '.5rem' }}>{title}</caption><thead><tr>{headers.map((s, i) => <th scope="col" key={i}>{s}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => j === 0 ? <th scope="row" key={j}>{cell}</th> : <td key={j} style={{ whiteSpace: typeof cell === 'number' || /^[+−-]?(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?|∞)$/i.test(cell) ? 'nowrap' : 'normal', overflowWrap: 'normal', maxWidth: '18rem' }}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
function Readout({ title, rows, t }: { title: string; rows: [string, Q, string][]; t: Text }) {
  return <Table title={title} headers={[t('관측량', 'Observation'), t('값', 'Value'), t('단위', 'Unit'), t('표현 상태', 'Representation status'), 'ln|q|']} rows={rows.map(([label, q, unit]) => [label, partitionQuantity(q, t), unit, partitionQuantityStatus(q, t), logarithm(q, t)])} />;
}
function PagedTable({ title, headers, rows, t }: { title: string; headers: string[]; rows: (string | number)[][]; t: Text }) {
  const [page, setPage] = useState(0), last = Math.max(0, Math.ceil(rows.length / 32) - 1), actual = Math.min(page, last);
  return <><div className="ai-lab-actions"><Button type="button" variant="outline" disabled={actual === 0} onClick={() => setPage(actual - 1)}>{t('이전 상태 행', 'Previous state rows')}</Button><output aria-label={t('상태 표 페이지', 'State-table page')}>{actual + 1}/{last + 1}</output><Button type="button" variant="outline" disabled={actual === last} onClick={() => setPage(actual + 1)}>{t('다음 상태 행', 'Next state rows')}</Button></div><Table title={title} headers={headers} rows={rows.slice(actual * 32, actual * 32 + 32)} /></>;
}
function PlotScroll({ title, t, children }: { title: string; t: Text; children: ReactNode }) {
  const hintId = useId();
  return <><p id={hintId} className="ai-small">{t('그림을 가로로 밀어 보세요. 키보드는 그림 영역에 초점을 두고 좌우 방향키를 사용하세요.', 'Swipe horizontally to explore the plot. With keyboard focus in the plot region, use the left and right arrow keys.')}</p><div className="ai-foundation-scroll" role="region" tabIndex={0} aria-label={title} aria-describedby={hintId}>{children}</div></>;
}
function Bars({ indices, series, t }: { indices: number[]; series: { name: string; color: string; values: Q[] }[]; t: Text }) {
  const step = 270 / Math.max(1, indices.length), labelStep = Math.max(1, Math.ceil(indices.length / 8)), title = t('원래 확률 비교', 'Original probability comparison');
  return <figure className="ai-foundation-figure" data-partition-figure data-partition-plot><h3>{title}</h3><PlotScroll title={title} t={t}><svg viewBox="0 0 350 270" style={{ minWidth: 560 }} role="img" aria-label={t('가로축은 선택 상태 지수,세로축은0부터1의 원래 확률입니다.', 'The horizontal axis gives selected state indices; the vertical axis is original probability from zero to one.')}>
    <path d="M54 18V218H330" fill="none" stroke="var(--learning-neutral)" />
    {[0, .5, 1].map(v => <text key={v} x="48" y={218 - 190 * v + 3} textAnchor="end" fill="currentColor" style={{ fontSize: 10 }}>{v}</text>)}
    {series.map((s, k) => <g key={s.name}>{s.values.map((q, i) => {
      const width = Math.max(.5, step / (series.length + 1)), x = 56 + i * step + k * width;
      const v = q?.value, unresolved = q?.resolution === 'unresolved-at-precision-ceiling';
      const height = typeof v === 'number' && Number.isFinite(v) && v > 0 ? Math.max(2, 190 * Math.min(1, v)) : 0;
      return <g key={i}><title>{`${s.name}(${indices[i]})=${partitionQuantity(q, t)}; ${partitionQuantityStatus(q, t)}; ln|p|=${logarithm(q, t)}`}</title>{unresolved || q === null ? <text x={x} y="205" style={{ fontSize: 10 }} fill="currentColor">?</text>
        : v === null ? <circle cx={x + width / 2} cy="215" r="2" fill="none" stroke={`var(--learning-${s.color})`} />
          : v === 0 ? <path d={`M${x} 218h${width}`} stroke={`var(--learning-${s.color})`} />
            : <rect x={x} y={218 - height} width={width} height={height} fill={`var(--learning-${s.color})`} fillOpacity={k ? .5 : .9} stroke={`var(--learning-${s.color})`} strokeDasharray={k ? '2 2' : undefined} />}</g>;
    })}</g>)}
    {indices.map((n, i) => i % labelStep === 0 || i === indices.length - 1 ? <text key={n} x={56 + i * step} y="236" style={{ fontSize: 10 }} fill="currentColor">{n}</text> : null)}
    <text x="190" y="258" textAnchor="middle" style={{ fontSize: 10 }} fill="currentColor">{t('선택 상태 지수', 'Selected state index')}</text>
  </svg></PlotScroll><div className="ai-legend">{series.map(s => <span key={s.name} style={{ color: `var(--learning-${s.color})` }}>{s.name}</span>)}</div><figcaption>{t('선택 행의 확률을 재정규화하지 않습니다. 가로 간격은 선택 레이블 간격입니다. 작은 양수는 가시성을 위한 최소2px 표식,표시 범위 밖 양수는 빈 원,미해결은 ?입니다. 실제 수치는 표에서 확인하세요.', 'Selected probabilities are not renormalized. Horizontal spacing separates selected labels. Tiny positives use a 2px visibility marker; positives outside display range use an open circle; unresolved results use ?. Read the actual numbers in the table.')}</figcaption></figure>;
}
function EnergyLevels({ levels, t }: { levels: readonly number[]; t: Text }) {
  const scale = Math.max(...levels.map(Math.abs)) || 1, title = t('상대 에너지 준위', 'Relative energy levels');
  return <figure className="ai-foundation-figure" data-partition-figure><h3>{title}</h3><PlotScroll title={title} t={t}><svg viewBox="0 0 350 270" style={{ minWidth: 560 }} role="img" aria-label={t('입력한 상대 에너지 준위입니다. 공통 기준은 이 그림에서 제외하며 실제 에너지는 표에 표시합니다.', 'Input relative energy levels. The common offset is excluded here; actual energies appear in the table.')}>
    <path d="M35 25V240M35 140H330" fill="none" stroke="var(--learning-neutral)" strokeDasharray="3 3" />
    {levels.map((v, i) => { const x = 45 + 275 * (i + .5) / levels.length, y = 140 - 100 * (v / scale), width = Math.min(40, 245 / levels.length);
      return <g key={i}><title>{`i=${i}, e=${partitionNumber(v)} J`}</title><line x1={x - width / 2} x2={x + width / 2} y1={y} y2={y} stroke="var(--learning-blue)" strokeWidth="3" />{levels.length <= 8 && <text x={x} y="257" textAnchor="middle" style={{ fontSize: 10 }} fill="currentColor">{i}</text>}</g>;
    })}<text x="39" y="20" style={{ fontSize: 10 }} fill="currentColor">{t('상대 척도 · 에너지(J)', 'Relative scale · energy (J)')}</text>
  </svg></PlotScroll><figcaption>{t('세로 위치는 최대 절댓값으로 나눈 상대 에너지입니다. 같은 높이여도 다른 레이블의 상태는 각각 셉니다.', 'Vertical positions are relative energies divided by their maximum absolute value. Distinct labeled states are counted separately even at equal height.')}</figcaption></figure>;
}
function Spectrum({ r, t }: { r: Extract<PartitionSuccess, { kind: 'finite-spectrum-partition-cumulant-generators' }>['result']; t: Text }) {
  return <><p>{t('고정된 명시적 미시상태를 정규화합니다. 공통 기준은 확률·S·C를 보존하고 U·F를 이동시킵니다. T=0은 양의 쪽 바닥 극한입니다.', 'Normalize fixed explicit microstates. A common offset preserves probabilities, S and C while shifting U and F. T=0 is the positive-side ground limit.')}</p>
    <p>c={partitionNumber(r.parameters.energyOffset)} J · T={partitionNumber(r.parameters.temperature)} K</p><EnergyLevels levels={r.parameters.energyLevels} t={t} />
    <Bars indices={r.states.map(s => s.index)} series={[{ name: 'p(i)', color: 'blue', values: r.states.map(s => s.probability) }]} t={t} />
    <Readout title={t('분배함수·모멘트 관측값', 'Partition and moment observations')} t={t} rows={[
      ['U', r.meanEnergy, 'J'], ['Var(E)', r.energyVariance, 'J²'], [t('세 번째 에너지 cumulant', 'Third energy cumulant'), r.thirdEnergyCumulant, 'J³'], ['S/kB', r.entropyInKB, '1'], ['S', r.entropy, 'J/K'], ['C/kB', r.heatCapacityInKB, '1'], ['C', r.heatCapacity, 'J/K'], ['F', r.freeEnergy, 'J'], ['ln Z', r.logPartition, '1'], ['Z', r.partitionFunction, '1'], ['∂β ln Z = −U', r.logZBetaDerivatives.first, 'J'], ['∂β² ln Z', r.logZBetaDerivatives.second, 'J²'], ['∂β³ ln Z', r.logZBetaDerivatives.third, 'J³']]} />
    <PagedTable title={t('이름 붙은 미시상태 · 32행씩', 'Labeled microstates · 32 rows per page')} headers={['i', t('상대 에너지(J)', 'Relative energy (J)'), t('실제 에너지(J)', 'Actual energy (J)'), 'p(i)', 'ln p(i)', t('확률 상태', 'Probability status')]} rows={r.states.map(s => [s.index, partitionNumber(s.relativeEnergy), partitionQuantity(s.energy, t), partitionQuantity(s.probability, t), partitionQuantity(s.logProbability, t), partitionQuantityStatus(s.probability, t)])} t={t} />
    <p className="ai-small">{t('이미 반올림된 입력에서 소실된 간격은 복구할 수 없습니다. 같은 에너지의 서로 다른 미시상태를 퇴화도와 함께 이중 계수하지 마세요.', 'Gaps lost before input cannot be recovered. Do not count both repeated microstates and a second degeneracy factor.')}</p></>;
}
function Pair({ r, t }: { r: Extract<PartitionSuccess, { kind: 'coupled-binary-sites-partition-factorization' }>['result']; t: Text }) {
  const title = t('두 스핀의 결합확률', 'Joint probabilities of two spins');
  return <><p>E=−Jσ₁σ₂−h(σ₁+σ₂) · J={partitionNumber(r.parameters.couplingEnergy)} J · h={partitionNumber(r.parameters.fieldEnergy)} J · T={partitionNumber(r.parameters.temperature)} K</p>
    <p>{t('결합 J와 허용 상태 제약은 다른 준비 조건입니다. 개별 평균0은 독립을 의미하지 않으며,Δ는 상호정보량과 다른 비교입니다.', 'Coupling J and a support constraint are different preparation conditions. Zero individual means do not imply independence; delta is a different comparison from mutual information.')}</p>
    <figure className="ai-foundation-figure" data-partition-figure><h3>{title}</h3><PlotScroll title={title} t={t}><svg viewBox="0 0 350 280" style={{ minWidth: 560 }} role="img" aria-label={t('행은 첫 스핀,열은 둘째 스핀입니다. 금지된 배치는 대각선으로 표시합니다.', 'Rows give the first spin; columns give the second. Forbidden configurations are crossed out.')}>
      <text x="195" y="20" textAnchor="middle" fill="currentColor" style={{ fontSize: 11 }}>σ₂</text>{[-1, 1].map((s, i) => <g key={s}><text x={124 + 125 * i} y="40" textAnchor="middle" fill="currentColor" style={{ fontSize: 12 }}>{s > 0 ? '+1' : '−1'}</text><text x="35" y={105 + 100 * i} fill="currentColor" style={{ fontSize: 12 }}>{s > 0 ? '+1' : '−1'}</text></g>)}
      {r.states.map((s, i) => { const x = 65 + 125 * (i % 2), y = 52 + 100 * Math.floor(i / 2), p = s.probability.value;
        return <g key={i}><title>{`${s.spins.join(',')}: ${partitionQuantity(s.probability, t)}; ${partitionQuantityStatus(s.probability, t)}`}</title><rect x={x} y={y} width="116" height="90" rx="8" fill="var(--learning-blue)" fillOpacity={p !== null ? .08 + .75 * p : .08} stroke="var(--learning-blue)" />{!s.allowed && <path d={`M${x + 6} ${y + 6}l104 78`} stroke="currentColor" strokeDasharray="4 4" />}<text x={x + 58} y={y + 37} textAnchor="middle" fill="currentColor" style={{ fontSize: 12 }}>{s.probability.resolution === 'unresolved-at-precision-ceiling' ? '?' : p === null ? '+' : partitionNumber(p)}</text><text x={x + 58} y={y + 62} textAnchor="middle" fill="currentColor" style={{ fontSize: 10 }}>{s.allowed ? t('허용', 'Allowed') : t('금지', 'Forbidden')}</text></g>;
      })}<text x="16" y="158" fill="currentColor" style={{ fontSize: 11 }}>σ₁</text>
    </svg></PlotScroll><figcaption>{t('사각형 색은 시각 보조이며 숫자가 확률입니다. +는 표시 범위 밖 양수,?는 미해결입니다. 금지 상태와 드문 양수는 다릅니다.', 'Cell color is a visual aid; numbers give probabilities. + means a positive outside display range, ? means unresolved. Forbidden states differ from rare positives.')}</figcaption></figure>
    <Readout title={t('상관·정보량과 곱 분해', 'Correlation, information and factorization')} t={t} rows={[
      ['E[A]', r.meanAlignment, '1'], ['E[M]', r.meanMagnetization, '1'], ['E[σ₁]', r.meanSpins[0], '1'], ['E[σ₂]', r.meanSpins[1], '1'], ['Cov(σ₁,σ₂)', r.spinCovariance, '1'], ['I(σ₁;σ₂)', r.mutualInformation, 'nat'], ['Δ=ln Z−2ln Z₁', r.logFactorizationDefect, '1'], ['F−2F₁', r.freeEnergyDefect, 'J'], ['U', r.joint.meanEnergy, 'J'], ['ln Z', r.joint.logPartition, '1'], ['Z', r.joint.partitionFunction, '1']]} />
    <Readout title={t('무차원 source Hessian', 'Dimensionless source Hessian')} t={t} rows={[
      ['H₁₁=Var(A)', r.sourceHessian[0][0], '1'], ['H₁₂=Cov(A,M)', r.sourceHessian[0][1], '1'], ['H₂₁=Cov(M,A)', r.sourceHessian[1][0], '1'], ['H₂₂=Var(M)', r.sourceHessian[1][1], '1']]} />
    <p>{t('source는 (βJ,βh)입니다. 물리적 응답은 고정 온도에서 β를 곱합니다. T=0에서는 바닥집합 선택을 따로 검사하며 미분이 없을 수 있습니다.', 'Sources are (beta J,beta h). Physical responses multiply by beta at fixed temperature. T=0 requires a separate ground-set check and may have no derivative.')}</p>
    <Readout title={t('고정 온도의 물리적 응답', 'Physical responses at fixed temperature')} t={t} rows={[
      ['∂J E[A]', r.responses.alignmentToCoupling.observation, '1/J'], ['∂h E[M]', r.responses.magnetizationToField.observation, '1/J'], ['∂h E[A]', r.responses.alignmentToField.observation, '1/J']]} />
    {Object.values(r.responses).some(v => v.status === 'undefined-at-zero-temperature') && <p>{t('이 바닥상태 교차에서는 표시한 응답 중 일부의 유한 미분이 존재하지 않습니다. 공분산0에 무한 β를 곱해 정의하지 않습니다.', 'Some displayed responses have no finite derivative at this ground-state crossing. Zero covariance times infinite beta does not define them.')}</p>}
    <PagedTable title={t('네 배치와 지지집합', 'Four configurations and support')} headers={[t('배치', 'Configuration'), t('허용', 'Allowed'), 'E (J)', 'p', 'ln p']} rows={r.states.map(s => [s.spins.join(','), s.allowed ? t('예', 'Yes') : t('아니오', 'No'), partitionQuantity(s.energy, t), partitionQuantity(s.probability, t), partitionQuantity(s.logProbability, t)])} t={t} /></>;
}
function Ladder({ r, t }: { r: Extract<PartitionSuccess, { kind: 'infinite-energy-ladder-normalization-cutoff' }>['result']; t: Text }) {
  const infinite = r.infinite.quantities, c = r.comparison;
  const thermodynamics = (q: typeof r.finite): [string, Q, string][] => [['E[n]', q.meanIndex, '1'], ['Var(n)', q.indexVariance, '1'], ['U', q.meanEnergy, 'J'], ['Var(E)', q.energyVariance, 'J²'], ['C/kB', q.heatCapacityInKB, '1'], ['C', q.heatCapacity, 'J/K'], ['S/kB', q.entropyInKB, '1'], ['S', q.entropy, 'J/K'], ['F', q.freeEnergy, 'J'], ['ln Z', q.logPartition, '1'], ['Z', q.partitionFunction, '1']];
  return <><p>Eₙ=ε(n+1/2) · ε={partitionNumber(r.parameters.gapEnergy)} J · T={partitionNumber(r.parameters.temperature)} K · L={r.parameters.cutoff}</p>
    <p>{t('유한 분포는0..L에서 정규화합니다. 무한 준비가 존재할 때 이는 n≤L 조건부 분포이며,그 합이1이라는 사실만으로 누락 꼬리가0은 아닙니다.', 'The finite law normalizes on 0..L. When the infinite law exists this is conditioning on n≤L; a finite sum of one does not make the omitted tail zero.')}</p>
    {r.infinite.status === 'not-normalizable' ? <p role="note" data-partition-normalization="absent">{t('무한 분포는 정규화할 수 없습니다. 간격0에서는 같은 가중치가 무한히 많고,음의 온도에서는 높은 준위의 가중치가 증가합니다. 아래에는 유한 준비만 표시합니다.', 'The infinite law cannot be normalized: zero gap gives infinitely many equal weights, and negative temperature gives increasing high-level weights. Only the finite preparation is displayed below.')}</p> : !infinite ? <p data-partition-normalization="unresolved">{t('무한 분포는 수학적으로 정규화 가능하지만 현재 정밀도에서 관측값을 해결하지 못했습니다.', 'The infinite law is mathematically normalizable, but its observations remain unresolved at the current precision.')}</p> : <p data-partition-normalization="present">{r.infinite.status === 'positive-side-zero-temperature-limit' ? t('무한 준비의 양의 쪽 온도0 극한: 바닥상태의 점질량입니다.', 'Positive-side zero-temperature limit of the infinite preparation: a ground-state point mass.') : t('무한 분포는 양의 간격·양의 온도에서 정규화됩니다.', 'The infinite law is normalizable at positive gap and temperature.')}</p>}
    <Bars indices={r.samples.map(s => s.n)} series={[{ name: t('유한 pL(n)', 'Finite pL(n)'), color: 'blue', values: r.samples.map(s => s.finiteProbability) }, ...(infinite ? [{ name: t('무한 p∞(n)', 'Infinite p∞(n)'), color: 'orange', values: r.samples.map(s => s.infiniteProbability) }] : [])]} t={t} />
    <Readout title={t('유한 절단의 열역학', 'Finite-cutoff thermodynamics')} t={t} rows={thermodynamics(r.finite)} />
    {infinite && <Readout title={t('무한 사다리의 열역학', 'Infinite-ladder thermodynamics')} t={t} rows={thermodynamics(infinite)} />}
    <Readout title={t('입력 비와 계산 기준', 'Input ratio and calculation anchor')} t={t} rows={[
      ['x=ε/(kBT)', r.x, '1'], ['q=exp(−x)', r.q, '1'], [t('기준점에서 감소하는 비', 'Decay ratio from the anchor'), r.anchoredDecayRatio, '1']]} />
    {c && <Readout title={t('절단 오차와 비교 방향', 'Cutoff error and comparison direction')} t={t} rows={[
      ['δ=P∞(n>L)', c.missingMass, '1'], ['1−δ', c.retainedMass, '1'], ['ln(1−δ)', c.logRetainedMass, '1'], ['TV(pL,p∞)', c.totalVariation, '1'], ['D(pL||p∞)', c.forwardKL, 'nat'], ['D(p∞||pL)', c.reverseKL, 'nat'], [t('평균 지수 편향', 'Mean-index bias'), c.meanIndexBias, '1'], [t('지수 분산 편향', 'Index-variance bias'), c.indexVarianceBias, '1'], [t('평균 에너지 편향', 'Mean-energy bias'), c.meanEnergyBias, 'J']]} />}
    <p className="ai-small">{t('무한 준비가 없을 때 절단 오차도 정의하지 않습니다. T=0에서는 두 분포가 같은 점질량이므로 두 KL이0입니다. 유한 양의 온도의 역방향 KL은 지지집합 차이로 무한대입니다.', 'No infinite preparation means no defined truncation error. At T=0 both laws are the same point mass and both KLs vanish. At finite positive temperature reverse KL is infinite because the supports differ.')}</p>
    <PagedTable title={r.sampledAllFiniteStates ? t('모든 유한 상태와 첫 바깥 상태', 'All finite states and the first outside state') : t('선택한 상태 · 재정규화하지 않음', 'Selected states · not renormalized')} headers={['n', 'Eₙ (J)', 'pL(n)', 'p∞(n)', t('유한 확률 상태', 'Finite probability status')]} rows={r.samples.map(s => [s.n, partitionQuantity(s.energy, t), partitionQuantity(s.finiteProbability, t), partitionQuantity(s.infiniteProbability, t), partitionQuantityStatus(s.finiteProbability, t)])} t={t} />
    <p className="ai-small">{t('준위 절단 제거는 입자 수·부피의 열역학 극한과 다릅니다. 고정 L의 높은 온도에서 C/kB→0,무한 사다리에서는1입니다.', 'Removing a level cutoff differs from the particle-number/volume thermodynamic limit. At high temperature C/kB tends to zero for fixed L, and one for the infinite ladder.')}</p></>;
}
export function PartitionResultView({ response, language }: { response: PartitionSuccess; language: 'ko' | 'en' }) {
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  return <div data-partition-result={response.kind} data-partition-resolved={response.result.fullyResolved ? 'true' : 'false'}>
    <p className="ai-small">{t('정확한0·미정·미해결·표시 범위 밖 비영·수학적 무한대는 다릅니다. 로그는 표의 단위로 쓴 크기의 자연로그입니다. 로그 꼬리 근사는 구간 인증과 구별합니다.', 'Exact zero, undefined, unresolved, nonzero outside display range and mathematical infinity differ. Logarithms use magnitudes in the stated units. Log-tail approximations differ from interval certification.')}</p>
    {!response.result.fullyResolved && <p role="note">{t('정밀도 상한에서 일부 관측값이 미해결입니다. 확인한 부호를 유지하며 미해결을0으로 바꾸지 않습니다.', 'Some observations remain unresolved at the precision ceiling. Established signs are retained; unresolved values are not replaced by zero.')}</p>}
    {response.kind === 'finite-spectrum-partition-cumulant-generators' ? <Spectrum r={response.result} t={t} /> : response.kind === 'coupled-binary-sites-partition-factorization' ? <Pair r={response.result} t={t} /> : <Ladder r={response.result} t={t} />}
    <p className="ai-small">{t('시도한 정밀도(bit)', 'Attempted precision (bits)')}: {response.result.attempts.join(' → ')}. {t('확률은 평형 준비를 설명하며 전이율이나 혼합 시간을 지정하지 않습니다.', 'Probabilities describe equilibrium preparations, not transition rates or mixing times.')}</p>
  </div>;
}
