'use client';
import { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { BinaryObservation } from '@/lib/learning-binary-products';
import type { MicroMacroSuccess } from '@/lib/learning-micro-macro-protocol';
import { binaryPatternExamples, microMacroPlotValue } from '@/lib/learning-micro-macro';
type Locale = 'ko' | 'en';
type Text = (ko: string, en: string) => string;
type Q = BinaryObservation | null;
export const microMacroNumber = (v: number | null) => v === null ? '—' : v === 0 ? '0' : Math.abs(v) < .001 || Math.abs(v) >= 10000 ? v.toExponential(3) : Number(v.toPrecision(5)).toString();
function value(q: Q, t: Text) {
  if (q === null) return t('미정', 'Undefined');
  if ('status' in q && q.status === 'infinite') return '∞';
  return q.value === null ? `${q.sign > 0 ? '+' : '−'} · ln|q|=${microMacroNumber(q.logMagnitude)}` : microMacroNumber(q.value);
}
function status(q: Q, t: Text) { return q === null ? t('조건부 미정', 'Undefined conditional') : 'status' in q && q.status === 'infinite' ? t('수학적 무한대', 'Mathematical infinity') : q.sign === 0 ? t('실제0', 'True zero') : q.value === null ? t('비영 · 선형 범위 밖', 'Nonzero; outside linear range') : t('반올림한 값', 'Rounded value'); }
function Table({ title, headers, rows }: { title: string; headers: string[]; rows: (string | number)[][] }) {
  return <div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption style={{ textAlign: 'left', padding: '.5rem' }}>{title}</caption><thead><tr>{headers.map((h, i) => <th scope="col" key={i}>{h}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => j === 0 ? <th scope="row" key={j}>{cell}</th> : <td key={j} style={typeof cell === 'string' && cell.length > 80 ? { whiteSpace: 'normal', overflowWrap: 'anywhere', wordBreak: 'break-all', maxWidth: '16rem' } : undefined}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
function Readout({ rows, t, context }: { rows: [string, Q, string][]; t: Text; context?: string }) {
  return <><p className="ai-small">{t('표는 좌우로 스크롤할 수 있습니다. 미정·실제0·선형 범위 밖·무한대는 서로 다릅니다. 로그는 표의 단위로 쓴 크기의 자연로그입니다.', 'Tables can be scrolled horizontally. Undefined, true zero, outside linear range and infinity differ. Logs use magnitudes in the stated unit.')}</p>
    <Table title={`${context ? context + ' · ' : ''}${t('정보량과 표현 상태', 'Information and representation status')}`} headers={[t('관측', 'Observation'), t('값', 'Value'), t('단위', 'Unit'), t('상태', 'Status'), 'ln|q|']} rows={rows.map(([label, q, unit]) => [label, value(q, t), unit, status(q, t), microMacroNumber(q?.logMagnitude ?? null)])} /></>;
}
function Bars({ title, labels, series, caption, t }: { title: string; labels: string[]; series: { name: string; color: string; values: Q[] }[]; caption: string; t: Text }) {
  const hint = useId();
  const finite = series.flatMap(s => s.values.map(microMacroPlotValue)).filter((v): v is number => v !== null);
  const maximum = Math.max(.000001, ...finite), count = Math.max(1, labels.length), step = 272 / count, labelStep = Math.max(1, Math.ceil(count / 8));
  return <figure className="ai-foundation-figure" data-micro-macro-plot={title}><h3>{title}</h3><p id={hint} className="ai-small">{t('좁은 화면에서는 그림을 좌우로 밀어 보세요. 그림 영역에 초점을 두고 좌우 방향키로도 이동할 수 있습니다.', 'On narrow screens, swipe the diagram horizontally. Focus its region to move with the left and right arrow keys.')}</p><div className="ai-foundation-scroll" role="region" tabIndex={0} aria-label={title} aria-describedby={hint}>
    <svg viewBox="0 0 350 270" style={{ minWidth: 560 }} role="img" aria-label={`${title}. ${caption}`}><path d="M54 18V218H330" fill="none" stroke="var(--learning-neutral)" />
      {[0, .5, 1].map(r => <text key={r} x="49" y={218 - 190 * r + 3} textAnchor="end" fill="currentColor" style={{ fontSize: 10 }}>{microMacroNumber(maximum * r)}</text>)}
      {series.map((s, k) => <g key={s.name}>{s.values.map((q, i) => {
        const v = microMacroPlotValue(q), width = Math.max(.5, step / (series.length + 1)), x = 56 + i * step + k * width;
        return v === null ? <text key={i} x={x} y="30" fill="currentColor" style={{ fontSize: 10 }}>?</text> : <rect key={i} x={x} y={218 - 190 * v / maximum} width={width} height={190 * v / maximum} fill={`var(--learning-${s.color})`} fillOpacity={k ? .55 : .9} stroke={`var(--learning-${s.color})`} strokeDasharray={k ? '2 2' : undefined} />;
      })}</g>)}
      {labels.map((label, i) => i % labelStep === 0 || i === labels.length - 1 ? <text key={i} x={56 + i * step} y="236" fill="currentColor" style={{ fontSize: 10 }}>{label}</text> : null)}
      <text x="190" y="258" textAnchor="middle" fill="currentColor" style={{ fontSize: 10 }}>{t('이산 레이블 · 확률', 'Discrete label · probability')}</text>
    </svg></div><div className="ai-legend">{series.map(s => <span key={s.name} style={{ color: `var(--learning-${s.color})` }}>{s.name}</span>)}</div><figcaption>{caption} {t('?는 표시할 수 없는 값이며0 막대가 아닙니다.', '? marks an unavailable value, not a zero bar.')}</figcaption></figure>;
}
export function microMacroChartIndices(length: number, selected: number[] = []) {
  if (!Number.isInteger(length) || length < 1) return [];
  const count = Math.min(65, length);
  return [...new Set([...Array.from({ length: count }, (_, i) => count === 1 ? 0 : Math.round((length - 1) * i / (count - 1))), ...selected.filter(i => Number.isInteger(i) && i >= 0 && i < length)])].sort((a, b) => a - b);
}
function BinaryView({ r, t }: { r: Extract<MicroMacroSuccess, { kind: 'binary-shell-multiplicity-block-counts' }>['result']; t: Text }) {
  const [page, setPage] = useState(0), [blockPage, setBlockPage] = useState(0), maxPage = Math.floor((r.shells.length - 1) / 32), actualPage = Math.min(page, maxPage);
  const blockMaxPage = Math.floor((r.blockDistribution.length - 1) / 32), actualBlockPage = Math.min(blockPage, blockMaxPage);
  const indices = microMacroChartIndices(r.shells.length, [r.parameters.excitations, ...r.macroModes]), shell = r.selectedShell;
  const patterns = binaryPatternExamples(r.parameters.sites, r.parameters.excitations);
  return <><p>N={r.parameters.sites} · K={r.parameters.excitations} · N_A={r.parameters.blockASites} · ε={microMacroNumber(r.parameters.energyGap)} J</p><h3>{t('이름 붙은 배치와 선택 껍질', 'Labeled patterns and selected shell')}</h3>
    <p style={{ overflowWrap: 'anywhere' }}>{t('다중도', 'Multiplicity')} Ω={shell.multiplicity}. {t('가장 흔한 여기 수', 'Modal excitation counts')}: {r.macroModes.join(', ')}.</p>
    {patterns && <p style={{ overflowWrap: 'anywhere' }}>{t('배치 예시', 'Example patterns')}: {patterns.slice(0, 12).join(' · ')}{patterns.length > 12 ? ' …' : ''}</p>}
    <Bars title={t('전체 준비에서 껍질 확률', 'Shell probabilities under full-space preparation')} labels={indices.map(String)} series={[{ name: 'P(K)', color: 'blue', values: indices.map(i => r.shells[i].probabilityUnderUniformFullSpace) }]} caption={t('큰N의 그림은 최대68개 선택 레이블을 보여줍니다. 정확한 전체 계수는 아래 페이지 표에 있습니다.', 'Large-N plots show at most 68 selected labels. Exact counts are available in the paged table below.')} t={t} />
    <Bars title={t('고정 K의 블록 A 조건부', 'Block A conditional at fixed K')} labels={microMacroChartIndices(r.blockDistribution.length).map(i => String(r.blockDistribution[i].excitationsA))} series={[{ name: 'P(A|K)', color: 'orange', values: microMacroChartIndices(r.blockDistribution.length).map(i => r.blockDistribution[i].probabilityGivenFixedTotal) }]} caption={t('조건은 균등한 선택 껍질입니다. 큰 분포는 레이블을 골라 표시하며 각 막대의 확률을 재정규화하지 않습니다.', 'Condition on the uniform selected shell. Large plots sample labels without renormalizing the displayed bars.')} t={t} />
    <Readout t={t} rows={[[t('전체 준비의 배치 하나', 'One full-space pattern'), r.uniformFullSpaceMicrostateProbability, '1'], [t('선택 껍질', 'Selected shell'), shell.probabilityUnderUniformFullSpace, '1'], [t('선택 껍질 내부 배치 하나', 'One pattern within shell'), r.uniformWithinSelectedShellProbability, '1'], ['H(X)', r.fullSpaceInformationEntropy, 'nat'], ['H(K)', r.macroInformationEntropy, 'nat'], ['H(X|K)', r.meanWithinShellInformationEntropy, 'nat'], ['SB(K)', shell.boltzmannEntropy, 'J/K'], ['−P(K) ln P(K)', shell.macroEntropyContribution, 'nat'], ['E=εK', r.selectedEnergy, 'J'], ['E[A|K]', r.conditionalBlockMean, '1'], ['Var(A|K)', r.conditionalBlockVariance, '1']]} />
    <div className="ai-lab-actions"><Button type="button" variant="outline" disabled={actualPage === 0} onClick={() => setPage(actualPage - 1)}>{t('이전 계수', 'Previous counts')}</Button><span>{actualPage + 1}/{maxPage + 1}</span><Button type="button" variant="outline" disabled={actualPage === maxPage} onClick={() => setPage(actualPage + 1)}>{t('다음 계수', 'Next counts')}</Button></div>
    <Table title={t('정확한 껍질 계수 · 페이지당32행', 'Exact shell counts · 32 rows per page')} headers={['K', 'Ω(N,K)', 'P(K)']} rows={r.shells.slice(actualPage * 32, actualPage * 32 + 32).map(s => [s.excitations, s.multiplicity, value(s.probabilityUnderUniformFullSpace, t)])} />
    <div className="ai-lab-actions"><Button type="button" variant="outline" disabled={actualBlockPage === 0} onClick={() => setBlockPage(actualBlockPage - 1)}>{t('이전 블록 계수', 'Previous block counts')}</Button><span>{actualBlockPage + 1}/{blockMaxPage + 1}</span><Button type="button" variant="outline" disabled={actualBlockPage === blockMaxPage} onClick={() => setBlockPage(actualBlockPage + 1)}>{t('다음 블록 계수', 'Next block counts')}</Button></div>
    <Table title={t('정확한 고정 K 블록 계수 · 페이지당32행', 'Exact fixed-K block counts · 32 rows per page')} headers={['A', 'K−A', 'W(A|K)', 'P(A|K)']} rows={r.blockDistribution.slice(actualBlockPage * 32, actualBlockPage * 32 + 32).map(b => [b.excitationsA, b.excitationsB, b.compatibleMicrostates, value(b.probabilityGivenFixedTotal, t)])} />
  </>;
}
function PartitionView({ r, t }: { r: Extract<MicroMacroSuccess, { kind: 'finite-macro-partition-conditional-entropy' }>['result']; t: Text }) {
  const [chosen, setChosen] = useState(0), selected = Math.min(chosen, r.groups.length - 1), group = r.groups[selected];
  return <><p>n={r.probabilities.length} · G={r.groups.length} · p=w/W · q=v/V</p><Bars title={t('미시 준비와 균등 재구성', 'Microstate preparation and uniform reconstruction')} labels={r.probabilities.map((_, i) => String(i))} series={[{ name: 'p', color: 'blue', values: r.probabilities }, { name: 'pbar', color: 'orange', values: r.reconstructedProbabilities }]} caption={t('재구성은 각 군 내부에 균등하게 질량을 채우는 추가 규칙입니다. 시간 진화가 아닙니다.', 'Reconstruction adds a uniform-within-group rule. It is not time evolution.')} t={t} />
    <Table title={t('미시 확률과 상태군', 'Microstate probabilities and groups')} headers={['i', 'g(i)', 'p(i)', 'q(i)', 'pbar(i)']} rows={r.probabilities.map((p, i) => [i, r.parameters.mapping[i], value(p, t), value(r.targetProbabilities[i], t), value(r.reconstructedProbabilities[i], t)])} />
    <Table title={t('관측 군의 확률과 크기', 'Observed group probability and size')} headers={['M', 'Ω', 'P(M)', 'Q(M)']} rows={r.groups.map(g => [g.group, g.count, value(g.probability, t), value(g.targetProbability, t)])} />
    <label className="ai-number-field"><span>{t('조건부를 볼 군', 'Group to condition on')}</span><select style={{ width: '100%', background: 'hsl(var(--background))', color: 'inherit', padding: '.5rem', border: '1px solid hsl(var(--input))', borderRadius: 'calc(var(--radius) - 2px)' }} value={selected} onChange={e => setChosen(Number(e.target.value))}>{r.groups.map(g => <option key={g.group} value={g.group}>M={g.group}</option>)}</select></label>
    <Table title={t('선택 군의 실제 조건부', 'Actual conditionals in selected group')} headers={['i', 'p(i|M)', 'q(i|M)']} rows={group.members.map((i, j) => [i, value(group.conditionalProbabilities?.[j] ?? null, t), value(group.targetConditionalProbabilities?.[j] ?? null, t)])} />
    {!group.count && <p>{t('이 군에는 상태가 없습니다.', 'This group contains no states.')}</p>}
    <Readout t={t} rows={[[t('선택 군 내부 엔트로피', 'Selected within-group entropy'), group.conditionalEntropy, 'nat'], [t('선택 군의 재구성 기여', 'Selected group reconstruction contribution'), group.reconstructionGainContribution, 'nat'], [t('선택 군의 가중 KL', 'Selected weighted group KL'), group.conditionalDivergenceContribution, 'nat'], ['H(X)', r.microEntropy, 'nat'], ['H(M)', r.macroEntropy, 'nat'], ['H(X|M)', r.conditionalEntropy, 'nat'], ['H(pbar)', r.reconstructedEntropy, 'nat'], [t('재구성 증가량', 'Reconstruction gain'), r.reconstructionGain, 'nat'], ['D(p||q)', r.microDivergence, 'nat'], ['D(P||Q)', r.macroDivergence, 'nat'], [t('가중 숨은 KL', 'Weighted hidden KL'), r.hiddenDivergence, 'nat']]} />
  </>;
}
function Matrix({ title, rows, t }: { title: string; rows: BinaryObservation[][]; t: Text }) { return <Table title={title} headers={['X', ...rows[0].map((_, j) => `Y=${j}`)]} rows={rows.map((row, i) => [i, ...row.map(q => value(q, t))])} />; }
function NoisyView({ r, t }: { r: Extract<MicroMacroSuccess, { kind: 'finite-noisy-macro-observation-channel' }>['result']; t: Text }) {
  const [chosen, setChosen] = useState(0), selected = Math.min(chosen, r.outputProbabilities.length - 1), posterior = r.posteriorProbabilities[selected];
  return <><p>{t('입력 상태 수', 'Input states')}: {r.inputProbabilities.length} · {t('출력 레이블 수', 'Output labels')}: {r.outputProbabilities.length}</p><Bars title={t('관측 전 입력', 'Input before observation')} labels={r.inputProbabilities.map((_, i) => String(i))} series={[{ name: 'p(X)', color: 'blue', values: r.inputProbabilities }]} caption={t('입력 준비를 장치의 행 조건부와 구별하세요.', 'Distinguish preparation from the instrument conditional rows.')} t={t} />
    <Matrix title={t('장치 채널 K(Y|X) · 각 행의 합1', 'Instrument channel K(Y|X) · each row sums to 1')} rows={r.channelProbabilities} t={t} />
    <Table title={t('채널 행 엔트로피와 가중 기여', 'Channel row entropy and weighted contribution')} headers={['X', 'p(X)', 'H(row) (nat)', 'pH(row) (nat)']} rows={r.rowEntropies.map((h, i) => [i, value(r.inputProbabilities[i], t), value(h, t), value(r.noiseEntropyContributions[i], t)])} />
    <Matrix title={t('결합 J(X,Y) · 전체 합1', 'Joint J(X,Y) · whole table sums to 1')} rows={r.jointProbabilities} t={t} />
    <Table title={t('출력 확률 · 결합 열의 합', 'Output probability · sum of joint column')} headers={['Y', 'R(Y)']} rows={r.outputProbabilities.map((q, j) => [j, value(q, t)])} />
    <label className="ai-number-field"><span>{t('사후를 볼 출력', 'Output to condition on')}</span><select style={{ width: '100%', background: 'hsl(var(--background))', color: 'inherit', padding: '.5rem', border: '1px solid hsl(var(--input))', borderRadius: 'calc(var(--radius) - 2px)' }} value={selected} onChange={e => setChosen(Number(e.target.value))}>{r.outputProbabilities.map((_, j) => <option key={j} value={j}>Y={j}</option>)}</select></label>
    {posterior ? <Bars title={t('선택 눈금 뒤의 사후', 'Posterior after selected reading')} labels={posterior.map((_, i) => String(i))} series={[{ name: 'P(X|Y)', color: 'green', values: posterior }]} caption={t('선택 출력의 결합 열을 R(Y)로 나눕니다. 지워짐 예제는 Y=2를?로 정의합니다.', 'Divide the selected joint column by R(Y). The erasure example defines Y=2 as ?.')} t={t} /> : <p>{t('이 출력의 확률은 실제0이므로 사후가 미정입니다.', 'This output has true zero probability, so its posterior is undefined.')}</p>}
    <Table title={t('선택 출력의 사후 확률', 'Posterior probabilities for selected output')} headers={['X', 'P(X|Y)']} rows={r.inputProbabilities.map((_, i) => [i, value(posterior?.[i] ?? null, t)])} />
    <Readout t={t} rows={[[t('선택 출력 확률', 'Selected output probability'), r.outputProbabilities[selected], '1'], [t('선택 출력의 숨은 정보', 'Hidden information at selected output'), r.posteriorEntropies[selected], 'nat'], ['H(X)', r.inputEntropy, 'nat'], ['H(Y)', r.outputEntropy, 'nat'], ['H(X,Y)', r.jointEntropy, 'nat'], ['H(Y|X)', r.noiseConditionalEntropy, 'nat'], ['H(X|Y)', r.hiddenConditionalEntropy, 'nat'], ['I(X;Y)', r.mutualInformation, 'nat']]} />
    <p>{t('독립 여부', 'Independence')}: {r.independentExactly ? t('정확히 독립', 'Exactly independent') : t('독립이 아님', 'Not independent')}. {t('준비 지지집합 위의 결정론적 관측', 'Deterministic on preparation support')}: {r.deterministicOnPreparationSupport ? t('예', 'Yes') : t('아니오', 'No')}.</p>
  </>;
}
export function MicroMacroResultView({ response, language }: { response: MicroMacroSuccess; language: Locale }) {
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  return <div data-micro-macro-result={response.kind}>{response.kind === 'binary-shell-multiplicity-block-counts' ? <BinaryView r={response.result} t={t} /> : response.kind === 'finite-macro-partition-conditional-entropy' ? <PartitionView r={response.result} t={t} /> : <NoisyView r={response.result} t={t} />}</div>;
}

/** A single compact calculated figure per concept; no saved experiment state. */
export function MicroMacroConceptResult({ response, step, language }: { response: MicroMacroSuccess; step: number; language: Locale }) {
  const t: Text = (ko, en) => language === 'ko' ? ko : en;
  if (response.kind === 'binary-shell-multiplicity-block-counts') {
    const r = response.result, block = step === 2, rare = step === 3;
    const labels = rare ? ['P(K=0)', '1/Ω'] : block ? r.blockDistribution.map(b => String(b.excitationsA)) : r.shells.map(s => String(s.excitations));
    const values = rare ? [r.selectedShell.probabilityUnderUniformFullSpace, r.uniformWithinSelectedShellProbability] : block ? r.blockDistribution.map(b => b.probabilityGivenFixedTotal) : r.shells.map(s => s.probabilityUnderUniformFullSpace);
    return <div><p>N={r.parameters.sites} · K={r.parameters.excitations} · N_A={r.parameters.blockASites} · Ω={r.selectedShell.multiplicity}</p>
      {step === 0 && <p>{binaryPatternExamples(4, 2)!.join(' · ')}</p>}
      <Bars title={rare ? t('희귀 껍질과 조건부 확률', 'Rare shell and conditional probability') : block ? t('제약 아래 블록 확률', 'Block probability under a constraint') : t('배치 수를 모은 거시 확률', 'Macro probability from collected patterns')} labels={labels} series={[{ name: block ? 'P(A|K)' : 'P', color: 'blue', values }]} caption={t('고정된 유한 예제입니다. 가로축은 이산 레이블이며 시간 궤적이 아닙니다.', 'A fixed finite example. The horizontal axis lists discrete labels, not a time trajectory.')} t={t} />
      <Readout t={t} context={t('고정 개념 예제', 'Fixed concept example')} rows={block ? [['E[A|K]', r.conditionalBlockMean, '1'], ['Var(A|K)', r.conditionalBlockVariance, '1']] : rare ? [['P(K=0)', r.selectedShell.probabilityUnderUniformFullSpace, '1'], ['−P ln P', r.selectedShell.macroEntropyContribution, 'nat'], ['SB(0)', r.selectedShell.boltzmannEntropy, 'J/K']] : [[t('전체 공간의 배치 하나', 'One full-space pattern'), r.uniformFullSpaceMicrostateProbability, '1'], ['P(K=2)', r.selectedShell.probabilityUnderUniformFullSpace, '1'], [t('껍질 내부 배치 하나', 'One within-shell pattern'), r.uniformWithinSelectedShellProbability, '1']]} /></div>;
  }
  if (response.kind === 'finite-macro-partition-conditional-entropy') {
    const r = response.result, grouped = step === 1;
    const labels = (grouped ? r.groups : r.probabilities).map((_, i) => String(i));
    const series = grouped ? [{ name: 'P(M)', color: 'blue', values: r.groups.map(g => g.probability) }, { name: 'Q(M)', color: 'orange', values: r.groups.map(g => g.targetProbability) }]
      : [{ name: 'p', color: 'blue', values: r.probabilities }, ...(step === 2 ? [{ name: 'pbar', color: 'orange', values: r.reconstructedProbabilities }] : step === 3 ? [{ name: 'q', color: 'orange', values: r.targetProbabilities }] : [])];
    return <div><Bars title={grouped ? t('같은 관측 아래 다른 숨은 정보', 'Different hidden information under the same readings') : step === 2 ? t('추가 규칙으로 만든 재구성', 'Reconstruction from an added rule') : t('가중 미시 준비', 'Weighted microstate preparation')} labels={labels} series={series} caption={t('미시 준비·거시 관측·재구성은 서로 다른 확률 법칙입니다.', 'Microstate preparation, macro observation and reconstruction are different probability laws.')} t={t} />
      <Readout t={t} context={t('고정 개념 예제', 'Fixed concept example')} rows={step === 3 ? [['D(p||q)', r.microDivergence, 'nat'], ['D(P||Q)', r.macroDivergence, 'nat'], [t('숨은 KL', 'Hidden KL'), r.hiddenDivergence, 'nat'], ['H(X|M)', r.conditionalEntropy, 'nat']] : [['H(X)', r.microEntropy, 'nat'], ['H(M)', r.macroEntropy, 'nat'], ['H(X|M)', r.conditionalEntropy, 'nat'], ['H(pbar)−H(p)', r.reconstructionGain, 'nat']]} /></div>;
  }
  const r = response.result, selected = step === 1 ? 1 : step === 2 ? 2 : 0, independent = step === 3;
  const values = independent ? r.outputProbabilities : step === 0 ? r.jointProbabilities.flat() : r.posteriorProbabilities[selected]!;
  const labels = step === 0 ? r.jointProbabilities.flatMap((row, i) => row.map((_, j) => `${i},${j}`)) : values.map((_, i) => String(i));
  return <div><p>{independent ? t('입력이 확실해도 램프는 무작위일 수 있습니다.', 'The lamp can be random even with a certain input.') : `${t('조건으로 읽은 출력', 'Conditioned reading')}: Y=${selected}`}</p>
    <Bars title={independent ? t('다양한 출력·정보0', 'Diverse outputs, zero information') : step === 0 ? t('입력과 눈금의 결합 질량', 'Joint input and reading masses') : t('관측 뒤 남는 사후', 'Posterior remaining after observation')} labels={labels} series={[{ name: independent ? 'P(Y)' : step === 0 ? 'J(X,Y)' : 'P(X|Y)', color: 'green', values }]} caption={t('관측의 행 조건부와 거꾸로 읽은 사후를 구분하세요.', 'Distinguish the forward channel conditional from the posterior.')} t={t} />
    <Readout t={t} context={t('고정 개념 예제', 'Fixed concept example')} rows={[[t('선택 출력 확률', 'Selected output probability'), r.outputProbabilities[selected], '1'], ['H(X)', r.inputEntropy, 'nat'], ['H(Y)', r.outputEntropy, 'nat'], ['H(Y|X)', r.noiseConditionalEntropy, 'nat'], ['H(X|Y)', r.hiddenConditionalEntropy, 'nat'], ['I(X;Y)', r.mutualInformation, 'nat']]} /></div>;
}
