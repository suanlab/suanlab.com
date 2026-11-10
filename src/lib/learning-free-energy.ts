/** Reader-facing parameterizations; numerical models retain their wider domain. */
import { finiteCanonicalHelmholtzGap } from './learning-canonical-gap';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
import { gasVolumeFreeEnergy } from './learning-gas-free-energy';
import type { BinaryObservation } from './learning-binary-products';
export type FreeEnergyKind = 'finite-spectrum-helmholtz-kl-variational' | 'ideal-gas-helmholtz-gibbs-volume-availability';
export const isFreeEnergyKind = (kind: string): kind is FreeEnergyKind => kind === 'finite-spectrum-helmholtz-kl-variational' || kind === 'ideal-gas-helmholtz-gibbs-volume-availability';
export type FreeEnergyText = { ko: string; en: string };
const text = (ko: string, en: string): FreeEnergyText => ({ ko, en });
export const freeEnergyControls = {
  'finite-spectrum-helmholtz-kl-variational': [
    { label: text('저장고 온도 T (K)', 'Reservoir temperature T (K)'), min: Number.MIN_VALUE, max: Number.MAX_VALUE, step: 'any' },
    { label: text('에너지 간격 / kB (K)', 'Energy gap / kB (K)'), min: 0, max: 1200, step: 1 },
    { label: text('미시상태 0 가중치', 'Microstate 0 weight'), min: 0, max: Number.MAX_VALUE, step: 'any' },
    { label: text('미시상태 1 가중치', 'Microstate 1 weight'), min: 0, max: Number.MAX_VALUE, step: 'any' },
    { label: text('미시상태 2 가중치', 'Microstate 2 weight'), min: 0, max: Number.MAX_VALUE, step: 'any' },
    { label: text('에너지 기준 / kB (K)', 'Energy reference / kB (K)'), min: -1000, max: 1000, step: 1 },
    { label: text('독립 복제 수 N (정수)', 'Independent copies N (integer)'), min: 0, max: Number.MAX_VALUE, step: 1 },
  ],
  'ideal-gas-helmholtz-gibbs-volume-availability': [
    { label: text('물질량 n (mol)', 'Amount n (mol)'), min: .25, max: 2, step: .25 },
    { label: text('고정 온도 T (K)', 'Fixed temperature T (K)'), min: 250, max: 800, step: 10 },
    { label: text('고정 외부 압력 P (Pa)', 'Fixed external pressure P (Pa)'), min: 25000, max: 200000, step: 1000 },
    { label: text('현재 체적 V (m³)', 'Query volume V (m³)'), min: .001, max: 1, step: 'any' },
    { label: text('기준 체적 Vr (m³)', 'Reference volume Vr (m³)'), min: .001, max: 1, step: 'any' },
    { label: text('기준 Helmholtz Fr (J)', 'Reference Helmholtz Fr (J)'), min: -10000, max: 10000, step: 100 },
  ],
} satisfies Record<FreeEnergyKind, { label: FreeEnergyText; min: number; max: number; step: number | 'any' }[]>;
export const freeEnergyPresets = {
  'finite-spectrum-helmholtz-kl-variational': [
    { label: text('균등 시도 분포', 'Uniform trial'), values: [300, 330, 1, 1, 1, 0, 1] },
    { label: text('거의 정준 분포', 'Nearly canonical'), values: [300, 300 * Math.log(2), 2, 2, 1, 0, 1] },
    { label: text('숨은 내부 불균형', 'Hidden conditional imbalance'), values: [300, 300 * Math.log(2), 4, 0, 1, 0, 1] },
    { label: text('정확한 퇴화 평형', 'Exact degenerate equilibrium'), values: [300, 0, 1, 1, 1, 0, 1] },
    { label: text('극저온 로그 꼬리', 'Cold logarithmic tail'), values: [Number.MIN_VALUE, 600, 1, 1, 0, 0, Number.MAX_VALUE] },
    { label: text('총량에서 복원되는 차이', 'Gap recovered in the total'), values: [1, 1000, 1, 1, 0, 0, Number.MAX_VALUE] },
  ],
  'ideal-gas-helmholtz-gibbs-volume-availability': [
    { label: text('팽창 상태 비교', 'Expanded state'), values: [1, 300, 100000, .05, .025, 1000] },
    { label: text('압축 상태 비교', 'Compressed state'), values: [1, 300, 100000, .0125, .025, 1000] },
    { label: text('반올림한 평형 체적', 'Rounded equilibrium volume'), values: [1, 300, 100000, 8.31446261815324 * 300 / 100000, .025, 1000] },
    { label: text('에너지 기준 이동', 'Shifted energy reference'), values: [1, 300, 100000, .05, .025, 5000] },
  ],
} satisfies Record<FreeEnergyKind, { label: FreeEnergyText; values: number[] }[]>;
export function validFreeEnergyValues(kind: FreeEnergyKind, value: unknown): value is number[] {
  if (!Array.isArray(value) || value.length !== freeEnergyControls[kind].length) return false;
  if (!Array.from(value).every((v, i) => typeof v === 'number' && Number.isFinite(v) && v >= freeEnergyControls[kind][i].min && v <= freeEnergyControls[kind][i].max)) return false;
  return kind !== 'finite-spectrum-helmholtz-kl-variational' || Number.isInteger(value[6]) && value.slice(2, 5).some(w => w > 0);
}
export const validHelmholtzValues = (v: unknown): v is number[] => validFreeEnergyValues('finite-spectrum-helmholtz-kl-variational', v);
export const validGasFreeEnergyValues = (v: unknown): v is number[] => validFreeEnergyValues('ideal-gas-helmholtz-gibbs-volume-availability', v);
export function evaluateHelmholtzValues(v: readonly number[]) {
  if (!validHelmholtzValues(v)) throw new RangeError('Invalid Helmholtz reader parameters');
  const [T, gap, w0, w1, w2, offset, copies] = v;
  return finiteCanonicalHelmholtzGap([kB * offset, kB * offset, kB * (offset + gap)], T, [w0, w1, w2], copies);
}
export function evaluateGasFreeEnergyValues(v: readonly number[]) {
  if (!validGasFreeEnergyValues(v)) throw new RangeError('Invalid gas reader parameters');
  return gasVolumeFreeEnergy({ moles: v[0], temperature: v[1], externalPressure: v[2], volume: v[3], referenceVolume: v[4], referenceHelmholtz: v[5] });
}
/** Never plot an unresolved anchored cancellation as a physical zero. */
export function freeEnergyPlotValue(q: BinaryObservation | null): number | null {
  if (q === null || ('zeroStatus' in q && q.zeroStatus === 'unresolved-transcendental-cancellation')) return null;
  return q.value;
}
/** Sample an explicit comparison family; index is not time or a solver step.
 * Canonical trials use uniform ground conditionals. Gas samples stay on the
 * currently supplied isotherm and retain its reference anchor.
 */
export function freeEnergyComparisonValues(kind: FreeEnergyKind, values: readonly number[], index: number) {
  if (!validFreeEnergyValues(kind, values) || !Number.isInteger(index) || index < 0 || index > 32)
    throw new RangeError('Invalid free-energy comparison frame');
  const next = [...values], x = index / 32;
  if (kind === 'finite-spectrum-helmholtz-kl-variational') {
    next[2] = 1 - x; next[3] = 1 - x; next[4] = 2 * x;
  } else next[3] = 10 ** (-3 + 3 * x);
  return next;
}
