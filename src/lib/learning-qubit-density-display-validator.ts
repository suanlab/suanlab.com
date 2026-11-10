/** Checks owned Worker observations using Schur PSD and indexed contractions.
 * This module never calls the display/density observation producers. */
import { channelValidatorPSD } from './learning-qubit-channel-validator';
import { validateQubitDisplayReal } from './learning-qubit-display-validator';
import { plus, times, conjugate, equal, Z, O, complex, quadratic, f,
  circuitScalarWire, type M, type C } from './learning-qubit-channel-algebra';
import type { observeQubitDensity } from './learning-qubit-density-display';

type Observation = ReturnType<typeof observeQubitDensity>;
export function validateQubitDensityObservation(matrix: M, observed: Observation): boolean {
  try {
    const n = matrix.length;
    if ((n !== 2 && n !== 4) || matrix.some(row => row.length !== n) || observed.dimension !== n) return false;
    const hermitian = matrix.every((row, i) => row.every((z, j) => equal(z, conjugate(matrix[j][i]))));
    const psd = channelValidatorPSD(matrix);
    const unitTrace = equal(matrix.reduce((total, row, i) => plus(total, row[i]), Z), O);
    const isDensity = psd && unitTrace;
    if (observed.hermitian !== hermitian || observed.positiveSemidefinite !== psd
      || observed.unitTrace !== unitTrace || observed.isDensity !== isDensity) return false;
    if (!isDensity) return observed.purity === null && observed.bloch === null;
    const check = (value: C, actual: NonNullable<Observation['purity']>) => actual
      && JSON.stringify(actual.exact) === JSON.stringify(circuitScalarWire(value))
      && validateQubitDisplayReal(value.re, actual.display.real)
      && validateQubitDisplayReal(value.im, actual.display.imaginary);
    let purity = Z;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
      purity = plus(purity, times(matrix[i][j], matrix[j][i]));
    if (!observed.purity || !check(purity, observed.purity)) return false;
    if (n === 4) return observed.bloch === null;
    const negative = complex(quadratic(f(BigInt(-1))));
    const imaginary = complex(quadratic(f(BigInt(0))), quadratic(f(BigInt(1))));
    const minusImaginary = conjugate(imaginary);
    const paulis: M[] = [[[Z, O], [O, Z]], [[Z, minusImaginary], [imaginary, Z]], [[O, Z], [Z, negative]]];
    if (!observed.bloch || observed.bloch.length !== 3) return false;
    return paulis.every((pauli, axis) => {
      let expected = Z;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++)
        expected = plus(expected, times(matrix[i][j], pauli[j][i]));
      return check(expected, observed.bloch![axis]);
    });
  } catch { return false; }
}
