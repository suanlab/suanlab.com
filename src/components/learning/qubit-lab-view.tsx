'use client';
import { useId, useState, type ReactNode } from 'react';
import type { QubitReportScalar, QubitReportMatrix, QubitCoordinateTree, QubitObservationState } from '@/lib/learning-qubit-observations-schema';
import type { QubitDisplayComplex } from '@/lib/learning-qubit-display';
import QubitBloch from './qubit-bloch';
export type QubitTranslate = (ko: string, en: string) => string;
export function qubitScalarText(value: QubitReportScalar): string {
  const quadratic = (pair: readonly string[]) => pair[1] === '0' ? pair[0] : `(${pair[0]}) + (${pair[1]})√2`;
  const real = quadratic(value.real), imaginary = quadratic(value.imaginary);
  return imaginary === '0' ? real : `${real} + i(${imaginary})`;
}
/** Only access coordinates already admitted by the trusted controller. */
export function qubitCoordinateAt(tree: QubitCoordinateTree, path: readonly (string | number)[]): QubitDisplayComplex {
  let value: unknown = tree;
  for (const part of path) value = (value as Record<string | number, unknown>)[part];
  return value as QubitDisplayComplex;
}
export function QubitTable({ title, headers, rows }: { title: string; headers: string[]; rows: ReactNode[][] }) {
  const id = useId();
  return <div className="qubit-table" role="region" tabIndex={0} aria-labelledby={id}>
    <table><caption id={id}>{title}</caption><thead><tr>{headers.map((h, i) => <th key={i} scope="col">{h}</th>)}</tr></thead>
      <tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => j === 0 ? <th scope="row" key={j}>{cell}</th> : <td key={j}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}
export function QubitValue({ exact, display, t }: { exact: QubitReportScalar; display?: QubitDisplayComplex; t: QubitTranslate }) {
  return <span className="qubit-value"><code>{qubitScalarText(exact)}</code>{display && <details><summary>{t('표시 구간', 'Display interval')}</summary>
    {(['real', 'imaginary'] as const).map((part, i) => { const v = display[part]; return <span key={part} className="qubit-certificate">
      {i === 0 ? t('실수', 'Real') : t('허수', 'Imaginary')}: {v.status === 'outside' ? t('±1 범위 밖 · 좌표 미제공', 'Outside ±1; no plot coordinate') : `[${v.lower}, ${v.upper}]`}
      {' · '}{v.sign === 0 ? t('정확한0', 'Exact zero') : v.sign > 0 ? t('정확히 양수', 'Strictly positive') : t('정확히 음수', 'Strictly negative')}
    </span>; })}</details>}</span>;
}
export function QubitMatrix({ matrix, title, t, coordinates }: { matrix: QubitReportMatrix; title: string; t: QubitTranslate; coordinates?: QubitDisplayComplex[][] }) {
  return <QubitTable title={title} headers={[t('행 / 열', 'Row / column'), ...matrix[0].map((_, j) => String(j))]}
    rows={matrix.map((row, i) => [String(i), ...row.map((z, j) => <QubitValue key={j} exact={z} display={coordinates?.[i][j]} t={t} />)])} />;
}
const roles: Record<string, [string, string]> = {
  input: ['입력 상태', 'Input state'], output: ['선형 사상의 출력', 'Linear map output'], bellOutput: ['Bell 참조 출력', 'Bell reference output'], referenceMarginal: ['참조계 주변 상태', 'Reference marginal'],
  joint: ['계·환경 결합 상태', 'Joint system–environment state'], system: ['축약 계 상태', 'Reduced system state'], environment: ['축약 환경 상태', 'Reduced environment state'], referenceOutput: ['참조계와 출력', 'Reference and output'],
  forgottenBeforeFeedback: ['기록을 잊은 보정 전 상태', 'Forgotten record before feedback'], systemAfterFeedback: ['보정 후 계 상태', 'System after feedback'],
  classicalQuantumBeforeFeedback: ['보정 전 고전–양자 상태', 'Classical–quantum state before feedback'], classicalQuantumAfterFeedback: ['보정 후 고전–양자 상태', 'Classical–quantum state after feedback'],
  conditionalBeforeFeedback0: ['결과0 · 보정 전 조건부 상태', 'Outcome0 · conditional state before feedback'], conditionalBeforeFeedback1: ['결과1 · 보정 전 조건부 상태', 'Outcome1 · conditional state before feedback'],
  conditionalAfterFeedback0: ['결과0 · 보정 후 조건부 상태', 'Outcome0 · conditional state after feedback'], conditionalAfterFeedback1: ['결과1 · 보정 후 조건부 상태', 'Outcome1 · conditional state after feedback'],
  coherentJoint: ['결맞는 결합 상태', 'Coherent joint state'], dephasedJoint: ['탈위상 결합 상태', 'Dephased joint state'], coherentSystem: ['결맞는 기록의 계 상태', 'System with coherent record'], coherentRecord: ['결맞는 기록 상태', 'Coherent record state'],
  dephasedSystem: ['탈위상 후 계 상태', 'System after dephasing'], dephasedRecord: ['탈위상 후 기록 상태', 'Record after dephasing'], recordSystem: ['기록–계 순서의 상태', 'State in record–system order'], classicalCorrectedSystem: ['고전 측정·보정 후 계', 'System after classical measurement and correction'],
};
export const qubitRoleLabel = (role: string, t: QubitTranslate) => roles[role] ? t(...roles[role]) : role;
export function QubitStatePanel({ states, language, t }: { states: Record<string, QubitObservationState | null>; language: 'ko' | 'en'; t: QubitTranslate }) {
  const keys = Object.keys(states), [selected, setSelected] = useState(() => ['output', 'system', 'systemAfterFeedback', 'coherentRecord'].find(role => keys.includes(role)) ?? keys[0]), id = useId();
  const role = keys.includes(selected) ? selected : keys[0], entry = states[role];
  const vector = entry?.observation.bloch?.map(v => v.display.real.coordinate);
  const usableVector = vector && vector.every(v => v !== null && Number.isFinite(v));
  return <section aria-label={t('상태 비교', 'Compare states')}>
    <QubitTable title={t('역할별 순도 비교', 'Purity by state role')} headers={[t('상태 역할', 'State role'), t('밀도행렬 조건', 'Density conditions'), t('순도', 'Purity')]} rows={keys.map(key => {
      const item = states[key]; return [qubitRoleLabel(key, t), item === null ? t('확률0 · 조건부 상태 없음', 'Zero probability; no conditional state') : item.observation.isDensity ? t('유효', 'Valid') : t('불만족', 'Not satisfied'), item?.observation.purity ? <QubitValue key="value-2" exact={item.observation.purity.exact} display={item.observation.purity.display} t={t} /> : '—'];
    })} />
    <label className="qubit-field" htmlFor={id}>{t('자세히 볼 상태', 'State to inspect')}<select id={id} aria-label={t('자세히 볼 상태', 'State to inspect')} value={role} onChange={e => setSelected(e.target.value)}>{keys.map(key => <option key={key} value={key}>{qubitRoleLabel(key, t)}</option>)}</select></label>
    {entry === null ? <p role="status">{t('확률0인 결과에서는 정규화한 조건부 상태가 정의되지 않습니다. 영행렬을 혼합 상태로 해석하지 마세요.', 'A normalized conditional state is undefined for a zero-probability outcome. The zero matrix is not a mixed state.')}</p> : <>
      {!entry.observation.isDensity && <p>{t('이 출력은 유효한 밀도행렬이 아닙니다. 자동으로 정규화하거나 Bloch 상태로 그리지 않습니다. 아래 정확 행렬과 trace·양성 조건을 비교하세요.', 'This output is not a valid density matrix. It is not automatically normalized or drawn as a Bloch state. Compare the exact matrix with the trace and positivity conditions.')}</p>}
      <QubitTable title={t('선택한 행렬의 조건', 'Selected matrix conditions')} headers={[t('조건', 'Condition'), t('결과', 'Result')]} rows={[
        ['Hermitian', entry.observation.hermitian ? t('예', 'Yes') : t('아니오', 'No')], ['PSD', entry.observation.positiveSemidefinite ? t('예', 'Yes') : t('아니오', 'No')], ['Tr ρ = 1', entry.observation.unitTrace ? t('예', 'Yes') : t('아니오', 'No')],
      ]} />
      {entry.observation.bloch && <QubitTable title={t('Bloch 성분과 정확 값', 'Bloch components and exact values')} headers={[t('축', 'Axis'), t('기댓값', 'Expectation')]} rows={entry.observation.bloch.map((z, i) => [['X', 'Y', 'Z'][i], <QubitValue key="value-3" exact={z.exact} display={z.display} t={t} />])} />}
      {usableVector && <QubitBloch vector={vector as number[]} scope={qubitRoleLabel(role, t)} language={language} />}
      <QubitMatrix title={qubitRoleLabel(role, t)} matrix={entry.matrix} coordinates={entry.matrixCoordinates} t={t} />
    </>}
  </section>;
}
