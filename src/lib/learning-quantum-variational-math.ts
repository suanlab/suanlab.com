/** Exact small matrix algebra shared by Rayleigh and forthcoming Ritz engines. */
import {c, ca, cm, scaleC, neg, one, zero, sum, type C, type F} from './learning-quantum-hilbert-core';
import type {Matrix} from './learning-observable-matrix';
export function variationalDeterminant(a: Matrix): C {
  if (a.length === 0) return c(one);
  return a[0].reduce((s, v, j) => ca(s, scaleC(cm(v, variationalDeterminant(a.slice(1).map(row => row.filter((_, k) => k !== j)))), j % 2 ? neg(one) : one)), c(zero));
}
export function variationalCharacteristic(a: Matrix): F[] {
  const minors = Array.from({length: 2 ** a.length - 1}, (_, k) => {
    const indices = Array.from({length: a.length}, (_, i) => i).filter(i => ((k + 1) & 2 ** i) !== 0);
    const value = variationalDeterminant(indices.map(i => indices.map(j => a[i][j])));
    if (value.im.n !== BigInt(0)) throw new RangeError('Real Hermitian characteristic required');
    return {order: indices.length, value: value.re};
  });
  return [one, ...Array.from({length: a.length}, (_, i) => {
    const value = sum(minors.filter(v => v.order === i + 1).map(v => v.value));
    return (i + 1) % 2 ? neg(value) : value;
  })];
}
