/** Exact report decoding, Worker/CLI only. Called after report verification
 * on the validation side, not on arbitrary reader input. */
import { decodeQubitFraction } from './learning-qubit-decode';
import { complex, type Q, type C, type M } from './learning-qubit-channel-algebra';
import type { QubitReportScalar, QubitReportMatrix } from './learning-qubit-observations-schema';
export const decodeQubitReportScalar = (value: QubitReportScalar): C => complex(
  value.real.map(decodeQubitFraction) as unknown as Q,
  value.imaginary.map(decodeQubitFraction) as unknown as Q,
);
export const decodeQubitReportMatrix = (value: QubitReportMatrix): M => value.map(row => row.map(decodeQubitReportScalar));
