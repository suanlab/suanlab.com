import { chainState, jacobianState, neuronGradientCheck, neuronState, parameterNames, type Activation, type NeuronParameters } from '@/lib/learning-calculus';
import type { Vec2 } from '@/lib/learning-simulations';
export type CalculusKind = 'chain-rule' | 'jacobians' | 'backpropagation';
export function isCalculusKind(kind: string): kind is CalculusKind { return ['chain-rule', 'jacobians', 'backpropagation'].includes(kind); }
export interface CalculusSettings {
  x: number; a: number; b: number; delta: number; epsilon: number;
  point: Vec2; direction: Vec2; seed: Vec2;
  input: Vec2; parameters: NeuronParameters; target: number; activation: Activation; rate: number; stage: number; updates: number;
}
export const defaultCalculus: CalculusSettings = {
  x: 1, a: 2, b: 1, delta: .1, epsilon: .0001, point: [2, 3], direction: [.1, -.2], seed: [1, -1],
  input: [1, 2], parameters: { w1: .5, w2: -.25, b: .5, v: 2, c: 0 }, target: 0, activation: 'relu', rate: .01, stage: 3, updates: 0,
};
export function validCalculus(value: unknown): value is CalculusSettings {
  if (!value || typeof value !== 'object') return false; const s = value as CalculusSettings;
  const scalar = (x: number, min: number, max: number) => typeof x === 'number' && Number.isFinite(x) && x >= min && x <= max;
  const pair = (v: unknown, bound: number) => Array.isArray(v) && v.length === 2 && v.every(x => scalar(x, -bound, bound));
  return [s.x, s.a, s.b].every(x => scalar(x, -3, 3)) && scalar(s.delta, -.5, .5) && scalar(s.epsilon, .000001, .1)
    && pair(s.point, 3) && pair(s.direction, .5) && pair(s.seed, 3) && pair(s.input, 3)
    && !!s.parameters && parameterNames.every(name => scalar(s.parameters[name], -100, 100))
    && scalar(s.target, -3, 3) && ['relu', 'tanh'].includes(s.activation) && scalar(s.rate, .001, 2)
    && Number.isInteger(s.stage) && s.stage >= 0 && s.stage <= 3 && Number.isInteger(s.updates) && s.updates >= 0 && s.updates <= 50;
}
export const calculusPresets: Record<CalculusKind, Partial<CalculusSettings>[]> = {
  'chain-rule': [{}, { x: -.5 }, { x: -1 }, { a: 0 }],
  jacobians: [{}, { direction: [.01, -.02] }, { point: [.5, .5], direction: [.1, -.1] }, { direction: [.5, -.5] }],
  backpropagation: [{}, { parameters: { ...defaultCalculus.parameters, b: -1 }, target: 1 },
    { rate: 2 }, { parameters: { ...defaultCalculus.parameters, b: 0 }, target: 1 }, { activation: 'tanh' }],
};
export function calculusValues(s: CalculusSettings) {
  return { chain: chainState(s.x, s.a, s.b, s.delta, s.epsilon), jacobian: jacobianState(s.point, s.direction, s.seed, s.epsilon),
    neuron: neuronState(s.input, s.parameters, s.target, s.activation), check: neuronGradientCheck(s.input, s.parameters, s.target, s.activation, s.epsilon) };
}
