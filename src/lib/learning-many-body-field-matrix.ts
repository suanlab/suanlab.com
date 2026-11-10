/** Sparse complex biquadratic matrices used only for response attribution. */
import { zero, one, type F } from './learning-tensor-wire-math';
import { field, fieldZero, fieldOne, fieldAdd, fieldSub, fieldNeg, fieldMul, fieldScale, fieldSum, expectedField, expectedNormalizedField, type WireField } from './learning-many-body-field-wire';
import type { ManyBodyExpected } from './learning-many-body-wire-match';
export type FieldComplex = { re: WireField; im: WireField };
export type FieldMatrix = FieldComplex[][];
export const fc = (re: WireField, im = fieldZero): FieldComplex => ({ re, im });
export const fcZero = (a: FieldComplex) => [...a.re, ...a.im].every(v => v.n === BigInt(0));
export const fcAdd = (a: FieldComplex, b: FieldComplex) => fc(fieldAdd(a.re, b.re), fieldAdd(a.im, b.im));
export const fcSub = (a: FieldComplex, b: FieldComplex) => fc(fieldSub(a.re, b.re), fieldSub(a.im, b.im));
export const fcConjugate = (a: FieldComplex) => fc(a.re, fieldNeg(a.im));
export const fcMultiply = (a: FieldComplex, b: FieldComplex) => fc(fieldSub(fieldMul(a.re, b.re), fieldMul(a.im, b.im)), fieldAdd(fieldMul(a.re, b.im), fieldMul(a.im, b.re)));
export const fcScale = (a: FieldComplex, b: WireField) => fc(fieldMul(a.re, b), fieldMul(a.im, b));
export const fcAbsSquared = (a: FieldComplex) => fieldAdd(fieldMul(a.re, a.re), fieldMul(a.im, a.im));
export const fieldNormSquared = (a: FieldComplex[]) => fieldSum(a.map(fcAbsSquared));
export const fieldDot = (a: FieldComplex[], b: FieldComplex[]) => a.reduce((s, v, i) => fcAdd(s, fcMultiply(fcConjugate(v), b[i])), fc(fieldZero));
export const fmEmpty = (n: number, m = n): FieldMatrix => Array.from({ length: n }, () => Array.from({ length: m }, () => fc(fieldZero)));
export const fmIdentity = (n: number): FieldMatrix => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => fc(i === j ? fieldOne : fieldZero)));
export const fmAdjoint = (a: FieldMatrix): FieldMatrix => a[0].map((_, j) => a.map(row => fcConjugate(row[j])));
export const fmSub = (a: FieldMatrix, b: FieldMatrix) => a.map((row, i) => row.map((v, j) => fcSub(v, b[i][j])));
export const fmAdd = (a: FieldMatrix, b: FieldMatrix) => a.map((row, i) => row.map((v, j) => fcAdd(v, b[i][j])));
export const fmScale = (a: FieldMatrix, b: WireField) => a.map(row => row.map(v => fcScale(v, b)));
export function fmProduct(a: FieldMatrix, b: FieldMatrix): FieldMatrix {
  const result = fmEmpty(a.length, b[0].length);
  for (let i = 0; i < a.length; i++) for (let k = 0; k < b.length; k++) if (!fcZero(a[i][k])) {
    for (let j = 0; j < b[0].length; j++) if (!fcZero(b[k][j])) result[i][j] = fcAdd(result[i][j], fcMultiply(a[i][k], b[k][j]));
  }
  return result;
}
export const fmTensor = (a: FieldMatrix, b: FieldMatrix): FieldMatrix => a.flatMap(row => b.map(other => row.flatMap(v => other.map(w => fcMultiply(v, w)))));
export const fmApply = (a: FieldMatrix, z: FieldComplex[]) => a.map(row => row.reduce((s, v, i) => fcZero(v) || fcZero(z[i]) ? s : fcAdd(s, fcMultiply(v, z[i])), fc(fieldZero)));
export const fmNormSquared = (a: FieldMatrix) => fieldNormSquared(a.flat());
export const fmTrace = (a: FieldMatrix) => a.reduce((s, row, i) => fcAdd(s, row[i]), fc(fieldZero));
export const expectedFieldComplex = (a: FieldComplex): ManyBodyExpected => ({ real: expectedField(a.re), imaginary: expectedField(a.im) });
export const expectedFieldMatrix = (a: FieldMatrix): ManyBodyExpected => a.map(row => row.map(expectedFieldComplex));
export const expectedFieldNormalizedComplex = (a: FieldComplex, norm: F): ManyBodyExpected => ({ real: expectedNormalizedField(a.re, norm), imaginary: expectedNormalizedField(a.im, norm) });
// Rational constants avoid importing the production biquadratic calculator.
export const fieldHalf = fieldScale(field(one), { n: one.n, d: BigInt(2) });
export const fieldComplexZero = fc(field(zero));
