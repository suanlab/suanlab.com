import type { QuantumVariationalSuccess } from './learning-quantum-variational-protocol';
type QuantumRitzResult = Extract<QuantumVariationalSuccess, { kind: 'nonorthogonal-ritz-subspace-certificates' }>['result'];

export type RitzResidualComponent = 'full' | 'inside' | 'outside';
export type RitzVector = [number, number, number];
export interface RitzGeometry {
  state: RitzVector;
  residual: RitzVector;
  axes: RitzVector[];
  columns: number;
}

const norm = (v: number[]) => Math.hypot(...v);
const unit = (v: RitzVector): RitzVector => {
  const length = norm(v);
  return v.map(x => x / length) as RitzVector;
};
const cross = (a: RitzVector, b: RitzVector): RitzVector => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

/** Render only an actually real three-dimensional slice. Never discard an
 * imaginary coordinate or hide a fourth component to manufacture a picture.
 * The orthogonal projector gives stable directions even for poorly scaled B.
 */
export function quantumRitzGeometry(
  result: QuantumRitzResult,
  component: RitzResidualComponent,
  energyScale: number,
): RitzGeometry | null {
  if (result.dimension !== 3 || !Number.isFinite(energyScale) || energyScale <= 0) return null;
  const selected = component === 'full' ? result.residualNumerators
    : component === 'inside' ? result.projectedResidualNumerators : result.outsideResidualNumerators;
  const real = (v: QuantumRitzResult['state'][number]): number | null => {
    const q = v.real;
    return v.imaginary.sign === 0 && q.value !== null && Number.isFinite(q.value)
      && q.resolution !== 'unresolved-at-precision-ceiling'
      && (q.value !== 0 || q.sign === 0) ? q.value : null;
  };
  const all = [...result.normalizedState, ...selected, ...result.spaceProjector.flat(), ...result.hamiltonian.flat()];
  if (all.some(v => real(v) === null)) return null;
  const N = result.rawNormSquared.value;
  if (N === null || N <= 0 || !Number.isFinite(N)) return null;
  const state = result.normalizedState.map(v => real(v)!) as RitzVector;
  const residual = selected.map(v => real(v)! / Math.sqrt(N) / energyScale) as RitzVector;
  if (![...state, ...residual].every(Number.isFinite)) return null;
  const projector = result.spaceProjector.map(row => row.map(v => real(v)!));
  const columns = [0, 1, 2].map(j => projector.map((row, i) =>
    result.columns === 2 ? Number(i === j) - row[j] : row[j]) as RitzVector);
  const strongest = columns.reduce((a, b) => norm(a) >= norm(b) ? a : b);
  if (norm(strongest) === 0) return null;
  let axes: RitzVector[];
  if (result.columns === 1) axes = [unit(strongest)];
  else if (result.columns === 2) {
    const normal = unit(strongest);
    const index = normal.map(Math.abs).indexOf(Math.min(...normal.map(Math.abs)));
    const reference = [0, 0, 0] as RitzVector;
    reference[index] = 1;
    const u = unit(cross(normal, reference));
    axes = [u, unit(cross(normal, u))];
  } else axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  return { state, residual, axes, columns: result.columns };
}
