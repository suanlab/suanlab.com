/** Density-specific teaching observations, calculated off the main thread.
 * Nonphysical linear images get no purity/Bloch state visualization. */
import {
  requireMatrix, principalMinorCertificate, equal, trace, O, product,
  circuitScalarWire, plus, minus, times, complex, quadratic, zero, one,
  type M,
} from './learning-qubit-channel-algebra';
import { observeQubitComplex } from './learning-qubit-display';

export function observeQubitDensity(matrix: M) {
  if (matrix.length !== 2 && matrix.length !== 4)
    throw new RangeError('Display density dimension must be two or four');
  requireMatrix(matrix, matrix.length);
  const certificate = principalMinorCertificate(matrix);
  const unitTrace = equal(trace(matrix), O);
  const isDensity = certificate.positiveSemidefinite && unitTrace;
  if (!isDensity) return {
    dimension: matrix.length, isDensity: false as const,
    hermitian: certificate.hermitian, positiveSemidefinite: certificate.positiveSemidefinite,
    unitTrace, purity: null, bloch: null,
  };
  const purity = trace(product(matrix, matrix));
  const observable = (value: typeof purity) => ({ exact: circuitScalarWire(value), display: observeQubitComplex(value) });
  const i = complex(quadratic(zero), quadratic(one));
  return {
    dimension: matrix.length, isDensity: true as const,
    hermitian: true, positiveSemidefinite: true, unitTrace: true,
    purity: observable(purity),
    bloch: matrix.length === 2 ? [
      plus(matrix[0][1], matrix[1][0]),
      times(i, minus(matrix[0][1], matrix[1][0])),
      minus(matrix[0][0], matrix[1][1]),
    ].map(observable) : null,
  };
}
