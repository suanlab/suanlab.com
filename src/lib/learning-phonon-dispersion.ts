/** Diatomic chain with unit spring stiffness and cell length. */
export function calculatePhononModes(massA: number, massB: number, q: number) {
  if (![massA, massB, q].every(Number.isFinite) || massA <= 0 || massB <= 0 || Math.abs(q) > Math.PI) {
    throw new RangeError('Expected positive masses and a wave number in the first Brillouin zone');
  }
  const a = 2 / massA, b = 2 / massB;
  const atEdge = Math.abs(q) === Math.PI;
  const coupling = atEdge ? 0 : 2 * Math.cos(q / 2) / Math.sqrt(massA * massB);
  const determinant = a * b * Math.sin(q / 2) ** 2;
  const upper = (a + b + Math.hypot(a - b, 2 * coupling)) / 2;
  const lower = atEdge ? Math.min(a, b) : determinant / upper;
  const degenerate = atEdge && a === b;

  let displacements: { acoustic: [number, number]; optical: [number, number] } | null = null;
  if (!degenerate) {
    const x = a >= b ? coupling : b - lower;
    const y = a >= b ? a - lower : coupling;
    const norm = Math.hypot(x, y);
    displacements = {
      acoustic: [x / (norm * Math.sqrt(massA)), y / (norm * Math.sqrt(massB))],
      optical: [-y / (norm * Math.sqrt(massA)), x / (norm * Math.sqrt(massB))],
    };
  }
  return {
    acoustic: Math.sqrt(lower), optical: Math.sqrt(upper), displacements,
    soundSpeed: Math.sqrt(1 / (2 * (massA + massB))),
  };
}
