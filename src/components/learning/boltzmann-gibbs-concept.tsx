'use client';
import { useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { boltzmannConceptInputs } from '@/lib/learning-boltzmann-gibbs-concepts';
import type { BoltzmannGibbsKind } from '@/lib/learning-boltzmann-gibbs';
import type { BoltzmannGibbsSuccess } from '@/lib/learning-boltzmann-gibbs-protocol';
import type { GibbsKernelName } from '@/lib/learning-finite-gibbs';
import { partitionNumber as number, partitionQuantity as quantity } from './partition-view';
import { BoltzmannTable, BoltzmannBars, BoltzmannGibbsGraph, BoltzmannGaussianFigure, BoltzmannGibbsResultView, boltzmannControlStyle, type BoltzmannText } from './boltzmann-gibbs-view';
import { useBoltzmannGibbsSeries } from './use-boltzmann-gibbs-series';
export function BoltzmannConceptResult({ kind, step, responses, language }: { kind: BoltzmannGibbsKind; step: number; responses: readonly BoltzmannGibbsSuccess[]; language: 'ko' | 'en' }) {
  const t: BoltzmannText = (ko, en) => language === 'ko' ? ko : en;
  const [focus, setFocus] = useState(0), [mode, setMode] = useState<GibbsKernelName>(step === 0 ? 'first' : step === 2 ? 'random' : 'systematic'), [frame, setFrame] = useState<'x' | 'z'>('x');
  const selected = responses[Math.min(focus, responses.length - 1)];
  if (!selected) return null;
  const q = (v: Parameters<typeof quantity>[0]) => quantity(v, t);
  const rows: (string | number)[][] = [];
  let headers: string[] = [];
  responses.forEach((response, i) => {
    if (kind === 'reference-measure-boltzmann-softmax-identifiability' && response.kind === kind) {
      const r = response.result; headers = [t('준비', 'Case'), t('레이블 수', 'Labels'), 'T (K)', 'c (J)', 'H(label)', 'E[ln m]', 'Sₘ/kB', 'ln Zₘ', 'U (J)', 'Var(E) (J²)'];
      rows.push([i + 1, r.states.length, number(response.input.parameters.temperature), number(response.input.parameters.energyOffset), ...[r.labelEntropyNats, r.meanLogReferenceWeight, r.relativeEntropyInKB, r.logPartition, r.meanEnergy, r.energyVariance].map(q)]);
    } else if (kind === 'finite-gibbs-conditionals-scan-support' && response.kind === kind) {
      const r = response.result, d = r.diagnostics[mode]; headers = [t('준비', 'Case'), 'J (J)', 'T (K)', t('지원', 'Support'), t('클래스 수', 'Classes'), t('선택 kernel 상세평형', 'Selected kernel detailed balance'), 'Δ flow(systematic)', 'λ(systematic)', 'TV(q₄,π)', t('Q 연결성', 'Connectivity of Q')];
      rows.push([i + 1, number(response.input.parameters.couplingEnergy), number(response.input.parameters.temperature), response.input.parameters.support === 'full-product' ? t('전체', 'Full') : t('합0', 'Zero sum'), d.stronglyConnectedComponents.length, d.detailedBalance ? t('예', 'Yes') : t('아니오', 'No'), q(r.systematicDetailedBalanceWitness.difference), q(r.systematicContraction), q(r.totalVariationToTarget[mode][2]), response.path?.status === 'ready' ? response.path.quantizedIrreducible ? t('한 클래스', 'One class') : t('여러 클래스', 'Multiple classes') : '—']);
    } else if (kind === 'quadratic-gaussian-gibbs-normalizability') {
      if (step === 3) {
        headers = [t('준비', 'Case'), t('법칙과 측도', 'Law and measure'), 'T (K)', 'C/kB', t('밀도의 H (nat)', 'Density H (nat)'), 'H(label) (nat)'];
        if (response.kind === kind && response.result.normalizable) rows.push([i + 1, t('R² 면적 Gaussian', 'R² area Gaussian'), number(response.input.parameters.temperature), q(response.result.heatCapacityInKB), q(response.result.differentialEntropyNats), '—']);
        else if (response.kind === 'reference-measure-boltzmann-softmax-identifiability') rows.push([i + 1, t('유한 레이블 바닥 법칙', 'Finite-label ground law'), '0', q(response.result.heatCapacityInKB), '—', q(response.result.labelEntropyNats)]);
      } else if (response.kind === kind) {
        const r = response.result;
        if (step === 1) {
          headers = [t('준비', 'Case'), 'A₁₁ (J)', 'A₁₂ (J)', 'A₂₂ (J)', 'det A (J²)', t('전체 평면 정규화', 'Full-plane normalization'), 'ln Z'];
          rows.push([i + 1, number(response.input.parameters.firstStiffness), number(response.input.parameters.crossStiffness), number(response.input.parameters.secondStiffness), q(r.determinant), r.normalizable ? t('가능', 'Proper') : t('불가', 'Not normalizable'), q(r.logPartition)]);
        } else if (step === 2 && r.normalizable) {
          headers = [t('준비', 'Case'), 'pₓ(max)', 'p_z(max)', 'Hₓ (nat)', 'H_z (nat)', 'ln|det C|']; rows.push([i + 1, ...[r.densityAtMean, r.transformedCoordinates.densityAtMean, r.differentialEntropyNats, r.transformedCoordinates.differentialEntropyNats, r.transformedCoordinates.logAbsoluteJacobian].map(q)]);
        } else if (r.normalizable) {
          headers = [t('준비', 'Case'), 'μ₁', 'μ₂', 'Σ₁₁', 'Σ₁₂', 'Σ₂₂', 'Corr(x₁,x₂)']; rows.push([i + 1, ...[...r.mean, r.covariance[0][0], r.covariance[0][1], r.covariance[1][1], r.correlation].map(q)]);
        }
      }
    }
  });
  return <div data-boltzmann-concept-result={kind}>
    <BoltzmannTable title={t('고정 개념 준비의 비교', 'Comparison of fixed concept preparations')} headers={headers} rows={rows} />
    <div className="ai-lab-actions" role="group" aria-label={t('그림으로 볼 고정 준비', 'Fixed preparation to visualize')}>{responses.map((_, i) => <Button key={i} type="button" variant="outline" aria-pressed={focus === i} onClick={() => setFocus(i)}>{t('준비', 'Case')} {i + 1}</Button>)}</div>
    {selected.kind === 'reference-measure-boltzmann-softmax-identifiability' && <BoltzmannBars title={t('선택한 준비의 레이블 확률', 'Label probabilities in the selected preparation')} labels={selected.result.states.map(s => String(s.index))} series={[{ label: 'p(i)', color: 'blue', values: selected.result.states.map(s => s.probability) }]} t={t} />}
    {selected.kind === 'finite-gibbs-conditionals-scan-support' && (step === 3 ? <BoltzmannGibbsResultView scope="concept" key={focus} response={selected} language={language} /> : <>
      <label className="ai-number-field"><span>{t('고정 준비에서 볼 갱신 규칙', 'Update rule in the fixed preparation')}</span><select aria-label={t('고정 준비에서 볼 갱신 규칙', 'Update rule in the fixed preparation')} style={boltzmannControlStyle} value={mode} onChange={e => setMode(e.target.value as GibbsKernelName)}>{[['first', t('첫째 자리', 'First coordinate')], ['second', t('둘째 자리', 'Second coordinate')], ['random', 'Random scan'], ['systematic', 'Systematic sweep'], ['blocked', 'Joint block']].map(([v, text]) => <option value={v} key={v}>{text}</option>)}</select></label>
      <BoltzmannGibbsGraph response={selected} mode={mode} t={t} />
      <BoltzmannTable title={t('고정 준비의 전이행렬', 'Transition matrix of the fixed preparation')} headers={['i→j', ...selected.result.states.map(s => s.spins.map(v => v < 0 ? '−' : '+').join(''))]} rows={selected.result.kernels[mode].map((row, i) => [i, ...row.map(q)])} />
    </>)}
    {selected.kind === 'quadratic-gaussian-gibbs-normalizability' && <><div className="ai-lab-actions"><Button type="button" variant="outline" aria-pressed={frame === 'x'} onClick={() => setFrame('x')}>{t('원래 좌표 x', 'Original coordinates x')}</Button><Button type="button" variant="outline" aria-pressed={frame === 'z'} onClick={() => setFrame('z')}>{t('변환 좌표 z', 'Transformed coordinates z')}</Button></div><BoltzmannGaussianFigure result={selected.result} frame={frame} t={t} /></>}
    {kind === 'reference-measure-boltzmann-softmax-identifiability' && step === 0 && <p>{t('준비1의 세 준위와 준비2의 네 미시상태는 서로 다른 레이블입니다. 두 가운데 미시상태 확률을 합하면 원래 가운데 준위 질량이 됩니다. 레이블 엔트로피는 달라도 기준 계수 엔트로피는 같습니다.', 'Case 1 has three level labels; case 2 has four microstate labels. Summing the two middle microstate probabilities recovers the middle level mass. Label entropies differ while reference-counting entropy agrees.')}</p>}
    {kind === 'quadratic-gaussian-gibbs-normalizability' && step === 3 && <p>{t('마지막 준비는 다른 기준 측도의 유한 간격 목록입니다. 미분 엔트로피와 레이블 엔트로피를 같은 열에 섞지 않습니다. 작은 양의 온도의 C/kB=1은 정확한0 K 면적 밀도를 뜻하지 않습니다.', 'The last case is a finite gapped list in a different measure. Differential and label entropy are kept in separate columns. C/kB=1 at tiny positive temperature does not define a zero-K area density.')}</p>}
  </div>;
}
export function BoltzmannConceptView({ kind, step, language }: { kind: BoltzmannGibbsKind; step: number; language: 'ko' | 'en' }) {
  const inputs = useMemo(() => boltzmannConceptInputs(kind, step), [kind, step]), calculation = useBoltzmannGibbsSeries(inputs), ko = language === 'ko';
  return <div data-boltzmann-concept={kind} data-boltzmann-step={step} data-boltzmann-concept-state={calculation.status}>
    {calculation.responses.length ? <BoltzmannConceptResult key={`${kind}-${step}`} kind={kind} step={step} responses={calculation.responses} language={language} /> : <p role="status">{calculation.status === 'error' ? ko ? '고정 준비를 계산하지 못했습니다. 다시 시도할 수 있습니다.' : 'The fixed preparation could not be calculated. You can retry.' : ko ? '고정 준비를 계산하고 있습니다.' : 'Calculating the fixed preparation.'}</p>}
    {calculation.status === 'error' && <Button type="button" variant="outline" onClick={calculation.retry}>{ko ? '다시 계산' : 'Retry calculation'}</Button>}
  </div>;
}
