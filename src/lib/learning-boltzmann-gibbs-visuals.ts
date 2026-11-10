/** Bounded visual approximations, never normalization/definiteness decisions. */
import type { quadraticGibbsModel } from './learning-quadratic-gibbs';
export type GaussianVisual = { status: 'not-normalizable' | 'outside-plot-range' | 'too-thin' } | {
  status: 'ready'; scale: number; frame: 'x' | 'z'; paths: { radius: number; points: string }[];
};
export function gaussianRelativeContours(result: ReturnType<typeof quadraticGibbsModel>, frame: 'x' | 'z' = 'x'): GaussianVisual {
  if (!result.normalizable) return { status: 'not-normalizable' };
  const cov = frame === 'z' ? result.transformedCoordinates.covariance : result.covariance;
  const l0 = cov[0][0].logMagnitude, l1 = cov[1][1].logMagnitude, rho = result.correlation.value;
  if (l0 === null || l1 === null || rho === null || ![l0, l1, rho].every(Number.isFinite)) return { status: 'outside-plot-range' };
  const scales = result.parameters.coordinateScale, orientation = frame === 'z' ? Math.sign(scales[0]) * Math.sign(scales[1]) : 1;
  const logScale = Math.max(l0, l1) / 2, scale = Math.exp(logScale);
  if (!Number.isFinite(scale) || scale === 0) return { status: 'outside-plot-range' };
  const a = Math.exp(l0 / 2 - logScale), b = rho * orientation * Math.exp(l1 / 2 - logScale);
  // Conditional variance is tau/A22; do not subtract nearly equal rounded
  // covariance entries or compute sqrt(1-rho²) after rho has rounded to ±1.
  const logConditional = (Math.log(1.380649e-23) + Math.log(result.parameters.temperature) - Math.log(result.parameters.secondStiffness)) / 2
    - (frame === 'z' ? Math.log(Math.abs(scales[1])) : 0);
  const c = Math.exp(logConditional - logScale);
  if (![a, b, c].every(Number.isFinite) || a <= 0 || c <= 0) return { status: 'outside-plot-range' };
  if (Math.min(a, c) < .001) return { status: 'too-thin' };
  const paths = [1, 2, 3].map(radius => ({ radius, points: Array.from({ length: 97 }, (_, i) => {
    const angle = 2 * Math.PI * i / 96, u = radius * Math.cos(angle), v = radius * Math.sin(angle);
    return `${(175 + 32 * a * u).toFixed(4)},${(135 - 32 * (b * u + c * v)).toFixed(4)}`;
  }).join(' ') }));
  if (paths.some(p => /NaN|Infinity/.test(p.points))) return { status: 'outside-plot-range' };
  return { status: 'ready', scale, frame, paths };
}
