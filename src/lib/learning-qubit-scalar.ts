/** Shared exact scalar arithmetic, without circuit execution or channel models. */
import { f, zero, one, add, mul, neg, eq, type F } from './learning-quantum-hilbert-core';

export type Quadratic = readonly [F, F];
export type CircuitScalar = { re: Quadratic; im: Quadratic };
export const quadratic = (a: F, b: F = zero): Quadratic => [a, b];
const qa = (a: Quadratic, b: Quadratic): Quadratic => [add(a[0], b[0]), add(a[1], b[1])];
export const quadraticNegate = (a: Quadratic): Quadratic => [neg(a[0]), neg(a[1])];
const qm = (a: Quadratic, b: Quadratic): Quadratic => [add(mul(a[0], b[0]), mul(f(BigInt(2)), mul(a[1], b[1]))), add(mul(a[0], b[1]), mul(a[1], b[0]))];
export const quadraticEqual = (a: Quadratic, b: Quadratic) => eq(a[0], b[0]) && eq(a[1], b[1]);
export const circuitScalar = (re: Quadratic, im = quadratic(zero)): CircuitScalar => ({ re, im });
export const circuitZero = circuitScalar(quadratic(zero));
export const circuitOne = circuitScalar(quadratic(one));
export const circuitAdd = (a: CircuitScalar, b: CircuitScalar) => circuitScalar(qa(a.re, b.re), qa(a.im, b.im));
export const circuitConjugate = (a: CircuitScalar) => circuitScalar(a.re, quadraticNegate(a.im));
export const circuitMultiply = (a: CircuitScalar, b: CircuitScalar) => circuitScalar(qa(qm(a.re, b.re), quadraticNegate(qm(a.im, b.im))), qa(qm(a.re, b.im), qm(a.im, b.re)));
export const circuitEqual = (a: CircuitScalar, b: CircuitScalar) => quadraticEqual(a.re, b.re) && quadraticEqual(a.im, b.im);
export const circuitNormSquared = (state: readonly CircuitScalar[]) => state.reduce((s, z) => qa(s, qa(qm(z.re, z.re), qm(z.im, z.im))), quadratic(zero));
export const circuitScalarWire = (z: CircuitScalar) => {
  const pair = (x: Quadratic) => x.map(v => v.d === BigInt(1) ? String(v.n) : `${v.n}/${v.d}`);
  return { real: pair(z.re), imaginary: pair(z.im) };
};
