/** Finite Gaussian calculation engine. Worker-side; reader registration is separate. */
import { f, zero, one, two, add, sub, mul, div, neg, sum, type F } from './learning-quantum-hilbert-core';
import { readThermalFieldInput, type ThermalJetInput } from './learning-thermal-field-input';
export type { ThermalFieldInput, ThermalJetInput } from './learning-thermal-field-input';
type Matrix = F[][];
type Jet = { value: F; first: F; second: F };
const integer = (n: number) => f(BigInt(n));
const text = (x: F) => x.d === BigInt(1) ? String(x.n) : `${x.n}/${x.d}`;
const zeros = (n: number, m = n): Matrix => Array.from({ length: n }, () => Array.from({ length: m }, () => zero));
const transpose = (a: Matrix): Matrix => a[0].map((_, j) => a.map(row => row[j]));
const product = (a: Matrix, b: Matrix): Matrix => a.map(row => b[0].map((_, j) => sum(row.map((x, k) => mul(x, b[k][j])))));
const matvec = (a: Matrix, v: F[]) => a.map(row => sum(row.map((x, j) => mul(x, v[j]))));
const dot = (a: F[], b: F[]) => sum(a.map((x, i) => mul(x, b[i])));
const trace = (a: Matrix) => sum(a.map((row, i) => row[i]));
const scale = (a: Matrix, x: F): Matrix => a.map(row => row.map(v => mul(v, x)));
const wire = (a: Matrix) => a.map(row => row.map(text));

function rational(value: string): F {
  const [n, d = '1'] = value.split('/');
  return f(BigInt(n), BigInt(d));
}
function jet(v: ThermalJetInput): Jet {
  return { value: rational(v.value), first: rational(v.first), second: rational(v.second) };
}
function readInput(value: unknown) {
  const v = readThermalFieldInput(value);
  if (!v) throw new RangeError('Invalid bounded finite thermal field input');
  const dimensions = v.shape, sites = dimensions.reduce((a, b) => a * b, 1);
  const spacing = rational(v.spacing), temperature = rational(v.temperature), referenceScale = rational(v.referenceScale);
  const mass = jet(v.mass), stiffness = jet(v.stiffness), background = jet(v.backgroundDensity);
  return { dimensions, sites, boundary: v.boundary, constrained: v.constraint === 'mean-zero',
    spacing, temperature, referenceScale, mass, stiffness, background, source: Array.from(v.sourceDensity, jet) };
}

/** A failed positive pivot proves failure of positive definiteness, not matrix singularity. */
function factorPositive(a: Matrix) {
  const n = a.length, lower = zeros(n), pivots: F[] = [];
  for (let i = 0; i < n; i++) {
    lower[i][i] = one;
    const pivot = sub(a[i][i], sum(Array.from({ length: i }, (_, k) => mul(mul(lower[i][k], lower[i][k]), pivots[k]))));
    if (pivot.n <= BigInt(0)) return { kind: 'not-positive' as const, index: i, pivot };
    pivots.push(pivot);
    for (let j = i + 1; j < n; j++) {
      lower[j][i] = div(sub(a[j][i], sum(Array.from({ length: i }, (_, k) => mul(mul(lower[j][k], lower[i][k]), pivots[k])))), pivot);
    }
  }
  const inverse = zeros(n);
  for (let col = 0; col < n; col++) {
    const y: F[] = [], z: F[] = [], x: F[] = [];
    for (let i = 0; i < n; i++) {
      y[i] = sub(i === col ? one : zero, sum(Array.from({ length: i }, (_, k) => mul(lower[i][k], y[k]))));
      z[i] = div(y[i], pivots[i]);
    }
    for (let i = n - 1; i >= 0; i--) {
      x[i] = sub(z[i], sum(Array.from({ length: n - i - 1 }, (_, k) => mul(lower[i + k + 1][i], x[i + k + 1]))));
    }
    for (let i = 0; i < n; i++) inverse[i][col] = x[i];
  }
  return { kind: 'positive' as const, inverse, determinant: pivots.reduce(mul, one), pivots };
}

export function calculateThermalField(value: unknown) {
  const input = readInput(value), { sites: n, dimensions, constrained, temperature: T } = input;
  const laplacian = zeros(n);
  const edge = (i: number, j: number) => {
    laplacian[i][i] = add(laplacian[i][i], one);
    laplacian[j][j] = add(laplacian[j][j], one);
    laplacian[i][j] = sub(laplacian[i][j], one);
    laplacian[j][i] = sub(laplacian[j][i], one);
  };
  let stride = 1;
  for (const length of dimensions) {
    for (let i = 0; i < n; i++) {
      const coordinate = Math.floor(i / stride) % length;
      if (coordinate + 1 < length) edge(i, i + stride);
      else if (input.boundary === 'periodic') edge(i, i - (length - 1) * stride);
      if (input.boundary === 'dirichlet' && coordinate === 0) laplacian[i][i] = add(laplacian[i][i], one);
      if (input.boundary === 'dirichlet' && coordinate === length - 1) laplacian[i][i] = add(laplacian[i][i], one);
    }
    stride *= length;
  }
  const cellVolume = dimensions.reduce(v => mul(v, input.spacing), one), volume = mul(integer(n), cellVolume);
  const spacingSquared = mul(input.spacing, input.spacing);
  const precision = (key: keyof Jet): Matrix => laplacian.map((row, i) => row.map((x, j) =>
    mul(cellVolume, add(i === j ? input.mass[key] : zero, div(mul(input.stiffness[key], x), spacingSquared)))));
  const K = precision('value'), K1 = precision('first'), K2 = precision('second');
  const J = input.source.map(x => mul(cellVolume, x.value)), J1 = input.source.map(x => mul(cellVolume, x.first)), J2 = input.source.map(x => mul(cellVolume, x.second));
  const E0 = mul(volume, input.background.value), E1 = mul(volume, input.background.first), E2 = mul(volume, input.background.second);
  const q = constrained ? n - 1 : n, basis = zeros(n, q);
  for (let i = 0; i < q; i++) { basis[i][i] = one; if (constrained) basis[n - 1][i] = neg(one); }
  const restricted = product(product(transpose(basis), K), basis), factor = factorPositive(restricted);
  if (factor.kind !== 'positive') return { kind: 'not-normalizable' as const, dimension: q, failedPivotIndex: factor.index,
    failedPivot: text(factor.pivot), precision: wire(K), laplacian: wire(laplacian),
    explanation: 'Not positive definite on the declared space; a zero pivot alone does not classify singularity.' };

  const G = product(product(basis, factor.inverse), transpose(basis)), covariance = scale(G, T), mean = matvec(G, J);
  const determinant = div(factor.determinant, constrained ? integer(n) : one);
  const minimumEnergy = sub(E0, div(dot(J, mean), two));
  const expectedEnergy = add(minimumEnergy, div(mul(integer(q), T), two));
  const Q = K.map((row, i) => row.map((x, j) => sub(x, mul(T, K1[i][j]))));
  const b = J.map((x, i) => sub(x, mul(T, J1[i]))), c = sub(E0, mul(T, E1));
  const QC = product(Q, covariance), Qmean = matvec(Q, mean), linear = Qmean.map((x, i) => sub(x, b[i]));
  const thermalEnergy = sub(add(c, div(add(trace(QC), dot(mean, Qmean)), two)), dot(b, mean));
  const variance = add(div(trace(product(QC, QC)), two), dot(linear, matvec(covariance, linear)));
  const expectedSecondDerivative = sub(add(E2, div(add(trace(product(K2, covariance)), dot(mean, matvec(K2, mean))), two)), dot(J2, mean));
  const heatCapacity = sub(div(variance, mul(T, T)), mul(T, expectedSecondDerivative));
  return {
    kind: 'normalizable' as const, dimension: q, sites: n, cellVolume: text(cellVolume), volume: text(volume),
    laplacian: wire(laplacian), precision: wire(K), precisionFirst: wire(K1), precisionSecond: wire(K2),
    source: J.map(text), mean: mean.map(text), covariance: wire(covariance), inverseOnSpace: wire(G),
    restrictedDeterminant: text(determinant), coordinateDeterminant: text(factor.determinant), gramDeterminant: constrained ? String(n) : '1',
    minimumEnergy: text(minimumEnergy), expectedEnergy: text(expectedEnergy), thermalEnergy: text(thermalEnergy),
    varianceAdjustedEnergy: text(variance), expectedSecondThermalDerivative: text(expectedSecondDerivative), heatCapacity: text(heatCapacity),
    /** log Z = q/2 log(2πT) - 1/2 log(det) - q log(scale) + constant. No decimal certification claimed. */
    logPartition: { dimension: q, temperature: text(T), restrictedDeterminant: text(determinant), referenceScale: text(input.referenceScale), constant: text(neg(div(minimumEnergy, T))) },
    entropyRationalAddend: text(div(thermalEnergy, T)), freeEnergyMultiplier: text(neg(T)),
  };
}
