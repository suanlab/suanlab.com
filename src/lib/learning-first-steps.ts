/** Small transparent preparations for the first visual reading; independent of saved full labs. */
export const firstStepIds = ['inner-products', 'gradients', 'bayesian-inference', 'finite-qubit-circuit-order-interference'] as const;
export type FirstStepId = typeof firstStepIds[number];
export const isFirstStepId = (id: string): id is FirstStepId => (firstStepIds as readonly string[]).includes(id);
export function fractionTex(value: number) {
  if (!Number.isFinite(value)) throw new RangeError('Finite value required');
  for (let denominator = 1; denominator <= 2 ** 24; denominator *= 2) {
    const numerator = Math.round(value * denominator);
    if (Number.isSafeInteger(numerator) && value === numerator / denominator) return denominator === 1 ? String(numerator) : `${numerator < 0 ? '-' : ''}\\frac{${Math.abs(numerator)}}{${denominator}}`;
  }
  throw new RangeError('An exact dyadic value is required');
}
export function projectionPreparation(example: number) {
  const directions = [[1, 0], [1, 1], [0, 1]];
  const v = directions[example]; if (!v) throw new RangeError('Unknown direction');
  const u = [3, 2], coefficient = (u[0] * v[0] + u[1] * v[1]) / (v[0] ** 2 + v[1] ** 2);
  const projected = v.map(x => coefficient * x), residual = u.map((x, i) => x - projected[i]);
  return { u, v, coefficient, projected, residual, residualSquared: residual.reduce((n, x) => n + x * x, 0) };
}
export function descentPreparation(example: number, step: number) {
  const rate = [.25, .75, 1.25][example];
  if (rate === undefined || !Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Unknown preparation');
  const points = Array.from({ length: step + 1 }, (_, k) => { const x = 2 * (1 - 2 * rate) ** k, y = 2 * (1 - rate) ** k; return { x, y, loss: x * x + .5 * y * y }; });
  return { rate, points, current: points[step] };
}
export function bayesPreparation(example: number, step: number) {
  const sequences = [[1, 1, 0], [0, 0, 0]];
  if (!sequences[example] || !Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Unknown preparation');
  const seen = sequences[example].slice(0, step), heads = seen.reduce((n, x) => n + x, 0);
  const fairWeight = 2 ** step, biasedWeight = 3 ** heads, total = fairWeight + biasedWeight;
  return { seen, fairWeight, biasedWeight, total, fair: fairWeight / total, biased: biasedWeight / total };
}
export function interferencePreparation(example: number, step: number) {
  if (![0, 1].includes(example) || !Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Unknown preparation');
  const negative = example === 0 && step === 2;
  const p1 = step === 0 ? 0 : step < 3 ? .5 : example === 0 ? 1 : 0;
  const tex = step === 0 ? '|0\\rangle' : step === 1 ? '\\frac{|0\\rangle+|1\\rangle}{\\sqrt{2}}' : step === 2 ? `\\frac{|0\\rangle${negative ? '-' : '+'}|1\\rangle}{\\sqrt{2}}` : example === 0 ? '|1\\rangle' : '|0\\rangle';
  return { negative, p0: 1 - p1, p1, tex };
}

const pointTex = (v: number[]) => `\\left(${v.map(fractionTex).join(',')}\\right)`;
export function firstStepEquation(kind: FirstStepId, example: number, stage: number) {
  if (kind === 'inner-products') {
    const p = projectionPreparation(example);
    return [
      `\\mathbf u=${pointTex(p.u)}`,
      `\\mathbf v=${pointTex(p.v)},\\quad \\mathbf p=\\alpha\\mathbf v`,
      `\\alpha=\\frac{\\mathbf u\\cdot\\mathbf v}{\\mathbf v\\cdot\\mathbf v}=${fractionTex(p.coefficient)},\\quad \\mathbf p=${pointTex(p.projected)}`,
      `\\mathbf r=\\mathbf u-\\mathbf p=${pointTex(p.residual)},\\quad \\mathbf v\\cdot\\mathbf r=0`,
    ][stage];
  }
  if (kind === 'gradients') { const r = descentPreparation(example, stage); return `\\eta=${fractionTex(r.rate)},\\quad \\left(x_{${stage}},y_{${stage}}\\right)=${pointTex([r.current.x, r.current.y])},\\quad L(x_{${stage}},y_{${stage}})=${fractionTex(r.current.loss)}`; }
  if (kind === 'bayesian-inference') { const r = bayesPreparation(example, stage); return `D=${r.seen.length ? `\\mathrm{${r.seen.map(x => x ? 'H' : 'T').join('')}}` : '\\varnothing'},\\quad P(H_f\\mid D)=\\frac{${r.fairWeight}}{${r.total}},\\quad P(H_b\\mid D)=\\frac{${r.biasedWeight}}{${r.total}}`; }
  const r = interferencePreparation(example, stage); return `|\\psi_{${stage}}\\rangle=${r.tex},\\quad P(0)=${fractionTex(r.p0)},\\quad P(1)=${fractionTex(r.p1)}`;
}

/** Readable summaries never feed rounded values back into the mathematical model. */
export function firstStepDisplayEquation(kind: FirstStepId, example: number, stage: number) {
  const rows = (values: string[]) => `\\begin{gathered}${values.join('\\\\')}\\end{gathered}`;
  if (kind !== 'gradients') return rows(firstStepEquation(kind, example, stage).split(/,\s*\\quad\s*/));
  const r = descentPreparation(example, stage);
  const term = (name: string, value: number) => {
    const rounded = Number(value.toFixed(3));
    return `${name}${rounded === value ? '=' : '\\approx'}${rounded}`;
  };
  return rows([`\\eta=${r.rate}`, term(`x_{${stage}}`, r.current.x), term(`y_{${stage}}`, r.current.y), term(`L_{${stage}}`, r.current.loss)]);
}
