import type { Vec2 } from './learning-simulations';
export type Matrix2 = [Vec2, Vec2];
export type Activation = 'relu' | 'tanh';
export interface NeuronParameters { w1: number; w2: number; b: number; v: number; c: number; }
export const parameterNames = ['w1', 'w2', 'b', 'v', 'c'] as const;
function finite(values: number[]) { if (!values.every(Number.isFinite)) throw new Error('Inputs must be finite.'); }
export function chainState(x: number, a: number, b: number, delta: number, epsilon: number) {
  finite([x, a, b, delta, epsilon]); if (epsilon <= 0) throw new Error('Difference epsilon must be positive.');
  const evaluate = (point: number) => (a * point + b) ** 2;
  const u = a * x + b; const output = u * u; const localOuter = 2 * u; const derivative = localOuter * a;
  const actualChange = evaluate(x + delta) - output; const linearChange = derivative * delta;
  const numeric = (evaluate(x + epsilon) - evaluate(x - epsilon)) / (2 * epsilon);
  return { u, output, localInner: a, localOuter, derivative, actualChange, linearChange,
    remainder: actualChange - linearChange, numeric, checkError: Math.abs(numeric - derivative) };
}
export function vectorFunction(point: Vec2): Vec2 { finite(point); return [point[0] * point[1], point[0] ** 2 + point[1]]; }
export function multiply2(matrix: Matrix2, vector: Vec2): Vec2 { return [matrix[0][0] * vector[0] + matrix[0][1] * vector[1], matrix[1][0] * vector[0] + matrix[1][1] * vector[1]]; }
export function jacobianState(point: Vec2, direction: Vec2, seed: Vec2, epsilon: number) {
  finite([...point, ...direction, ...seed, epsilon]); if (epsilon <= 0) throw new Error('Difference epsilon must be positive.');
  const [x, y] = point; const output = vectorFunction(point); const matrix: Matrix2 = [[y, x], [2 * x, 1]];
  const jvp = multiply2(matrix, direction); const vjp: Vec2 = [matrix[0][0] * seed[0] + matrix[1][0] * seed[1], matrix[0][1] * seed[0] + matrix[1][1] * seed[1]];
  const moved = vectorFunction([x + direction[0], y + direction[1]]); const actualChange: Vec2 = [moved[0] - output[0], moved[1] - output[1]];
  const plus = vectorFunction([x + epsilon * direction[0], y + epsilon * direction[1]]); const minus = vectorFunction([x - epsilon * direction[0], y - epsilon * direction[1]]);
  const numeric: Vec2 = [(plus[0] - minus[0]) / (2 * epsilon), (plus[1] - minus[1]) / (2 * epsilon)];
  return { output, matrix, jvp, vjp, actualChange, remainder: [actualChange[0] - jvp[0], actualChange[1] - jvp[1]] as Vec2,
    numeric, checkError: Math.max(...numeric.map((v, i) => Math.abs(v - jvp[i]))), determinant: y - 2 * x * x,
    dualLeft: seed[0] * jvp[0] + seed[1] * jvp[1], dualRight: vjp[0] * direction[0] + vjp[1] * direction[1] };
}
export function neuronState(input: Vec2, parameters: NeuronParameters, target: number, activation: Activation) {
  finite([...input, ...parameterNames.map(name => parameters[name]), target]);
  if (!['relu', 'tanh'].includes(activation)) throw new Error('Unsupported activation.');
  const p = parameters; const z = p.w1 * input[0] + p.w2 * input[1] + p.b;
  const hidden = activation === 'relu' ? Math.max(0, z) : Math.tanh(z);
  const kink = activation === 'relu' && z === 0;
  // This is an explicit computational convention at the ReLU kink, not a classical derivative.
  const activationSlope = activation === 'relu' ? (z > 0 ? 1 : 0) : 1 - hidden * hidden;
  const prediction = p.v * hidden + p.c; const error = prediction - target; const loss = .5 * error * error;
  const adjointHidden = error * p.v; const adjointZ = adjointHidden * activationSlope;
  const gradients: NeuronParameters = { w1: adjointZ * input[0], w2: adjointZ * input[1], b: adjointZ, v: error * hidden, c: error };
  finite([z, hidden, prediction, loss, ...parameterNames.map(name => gradients[name])]);
  return { z, hidden, prediction, error, loss, kink, activationSlope, adjointHidden, adjointZ, gradients,
    inputGradient: [adjointZ * p.w1, adjointZ * p.w2] as Vec2 };
}
export function neuronStep(input: Vec2, parameters: NeuronParameters, target: number, activation: Activation, rate: number) {
  finite([rate]); if (rate <= 0) throw new Error('Learning rate must be positive.');
  const state = neuronState(input, parameters, target, activation);
  return Object.fromEntries(parameterNames.map(name => [name, parameters[name] - rate * state.gradients[name]])) as unknown as NeuronParameters;
}
export function neuronGradientCheck(input: Vec2, parameters: NeuronParameters, target: number, activation: Activation, epsilon: number) {
  finite([epsilon]); if (epsilon <= 0) throw new Error('Difference epsilon must be positive.');
  const base = neuronState(input, parameters, target, activation);
  const rows = parameterNames.map(name => {
    const plus = neuronState(input, { ...parameters, [name]: parameters[name] + epsilon }, target, activation);
    const minus = neuronState(input, { ...parameters, [name]: parameters[name] - epsilon }, target, activation);
    const numeric = (plus.loss - minus.loss) / (2 * epsilon); const analytic = base.gradients[name];
    const smooth = activation === 'tanh' || plus.z === minus.z || (!base.kink && plus.z * minus.z > 0);
    return { name, analytic, numeric, absoluteError: Math.abs(analytic - numeric), smooth };
  });
  return { rows, smooth: rows.every(row => row.smooth), maxError: Math.max(...rows.map(row => row.absoluteError)) };
}
export type { Vec2 };
