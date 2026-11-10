/** Bounded exact circuit kernel. Not imported by the reader until Worker integration.
 * Scalars belong to Q(sqrt(2), i); site zero is the most significant bit.
 */
import { f, zero, one, neg } from './learning-quantum-hilbert-core';
import { quadratic,quadraticNegate as qn,quadraticEqual as qe,circuitScalar,circuitZero,circuitOne,
  circuitAdd,circuitConjugate,circuitMultiply,circuitNormSquared,type Quadratic,type CircuitScalar } from './learning-qubit-scalar';
export { quadratic,circuitScalar,circuitZero,circuitOne,circuitAdd,circuitConjugate,circuitMultiply,circuitEqual,circuitNormSquared,circuitScalarWire } from './learning-qubit-scalar';
export type { Quadratic,CircuitScalar } from './learning-qubit-scalar';

export type SingleGate = 'I' | 'X' | 'Y' | 'Z' | 'H' | 'S' | 'Sdg' | 'T' | 'Tdg';
export type CircuitGate = { gate: SingleGate; target: number } | { gate: 'CX' | 'CZ'; control: number; target: number } | { gate: 'SWAP'; target: number; other: number };
const n1 = circuitScalar(quadratic(neg(one)));
const i = circuitScalar(quadratic(zero), quadratic(one));
const ni = circuitConjugate(i);
const h = circuitScalar(quadratic(zero, f(BigInt(1), BigInt(2))));
const nh = circuitScalar(qn(h.re));
const t = circuitScalar(h.re, h.re);
const singleMatrices: Record<SingleGate, readonly (readonly CircuitScalar[])[]> = {
  I: [[circuitOne, circuitZero], [circuitZero, circuitOne]],
  X: [[circuitZero, circuitOne], [circuitOne, circuitZero]],
  Y: [[circuitZero, ni], [i, circuitZero]],
  Z: [[circuitOne, circuitZero], [circuitZero, n1]],
  H: [[h, h], [h, nh]],
  S: [[circuitOne, circuitZero], [circuitZero, i]],
  Sdg: [[circuitOne, circuitZero], [circuitZero, ni]],
  T: [[circuitOne, circuitZero], [circuitZero, t]],
  Tdg: [[circuitOne, circuitZero], [circuitZero, circuitConjugate(t)]],
};

function validate(sites: number, gates: readonly CircuitGate[], state: readonly CircuitScalar[]) {
  if (!Number.isInteger(sites) || sites < 1 || sites > 3) throw new RangeError('Use one to three qubits');
  if (state.length !== 2 ** sites || gates.length > 16) throw new RangeError('State dimension or gate budget exceeded');
  const site = (v: number) => { if (!Number.isInteger(v) || v < 0 || v >= sites) throw new RangeError('Invalid qubit site'); };
  for (const op of gates) {
    site(op.target);
    if (op.gate === 'CX' || op.gate === 'CZ') { site(op.control); if (op.control === op.target) throw new RangeError('Distinct control and target required'); }
    else if (op.gate === 'SWAP') { site(op.other); if (op.other === op.target) throw new RangeError('Distinct swap sites required'); }
    else if (!Object.prototype.hasOwnProperty.call(singleMatrices, op.gate)) throw new RangeError('Unknown gate');
  }
  const validPair = (pair: Quadratic) => pair.length === 2 && pair.every(v => typeof v.n === 'bigint' && typeof v.d === 'bigint' && v.d > BigInt(0));
  if (state.some(z => !validPair(z.re) || !validPair(z.im))) throw new RangeError('Invalid exact scalar');
  if (qe(circuitNormSquared(state), quadratic(zero))) throw new RangeError('Zero vector is not a state');
}

/** Chronological gates: [A,B] returns B A |state>. Prefix zero is the input. */
export function runQubitCircuit(sites: number, gates: readonly CircuitGate[], input: readonly CircuitScalar[]) {
  validate(sites, gates, input);
  let state = input.slice();
  const prefixes = [state];
  for (const op of gates) {
    const targetMask = 2 ** (sites - 1 - op.target);
    const output = Array.from({ length: state.length }, () => circuitZero);
    for (let basis = 0; basis < state.length; basis++) {
      const bit = (basis & targetMask) ? 1 : 0;
      if (op.gate === 'CX' || op.gate === 'CZ') {
        const enabled = !!(basis & 2 ** (sites - 1 - op.control));
        const destination = enabled && op.gate === 'CX' ? basis ^ targetMask : basis;
        const amplitude = enabled && op.gate === 'CZ' && bit ? circuitMultiply(n1, state[basis]) : state[basis];
        output[destination] = circuitAdd(output[destination], amplitude);
      } else if (op.gate === 'SWAP') {
        const otherMask = 2 ** (sites - 1 - op.other);
        const different = bit !== ((basis & otherMask) ? 1 : 0);
        const destination = different ? basis ^ targetMask ^ otherMask : basis;
        output[destination] = circuitAdd(output[destination], state[basis]);
      } else {
        const matrix = singleMatrices[op.gate];
        for (let row = 0; row < 2; row++) {
          const destination = (basis & ~targetMask) | (row ? targetMask : 0);
          output[destination] = circuitAdd(output[destination], circuitMultiply(matrix[row][bit], state[basis]));
        }
      }
    }
    state = output;
    prefixes.push(state);
  }
  return { state, prefixes, normSquared: circuitNormSquared(state) };
}

export function inverseCircuit(gates: readonly CircuitGate[]): CircuitGate[] {
  const inverse: Partial<Record<SingleGate, SingleGate>> = { S: 'Sdg', Sdg: 'S', T: 'Tdg', Tdg: 'T' };
  return gates.slice().reverse().map(op => op.gate === 'CX' || op.gate === 'CZ' || op.gate === 'SWAP' ? { ...op } : { ...op, gate: inverse[op.gate] ?? op.gate });
}
