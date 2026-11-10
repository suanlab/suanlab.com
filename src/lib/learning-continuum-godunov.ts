/** Actual scalar Burgers updates. All quantities are nondimensional. */
export interface BurgersObservation {
  value: number | null;
  sign: number;
  logMagnitude: number | null;
}
interface Q { sign: number; log: number }
interface State { initial: number; change: Q }
export type BurgersMethod = 'godunov' | 'roe';
const zero: Q = { sign: 0, log: 0 };
function product(...xs: number[]): Q {
  return xs.includes(0) ? zero : { sign: xs.reduce((s, x) => s * Math.sign(x), 1), log: xs.reduce((s, x) => s + Math.log(Math.abs(x)), 0) };
}
const neg = (q: Q): Q => q.sign ? { sign: -q.sign, log: q.log } : zero;
const mul = (a: Q, b: Q): Q => a.sign && b.sign ? { sign: a.sign * b.sign, log: a.log + b.log } : zero;
const scale = (a: Q, ...xs: number[]) => mul(a, product(...xs));
const square = (a: Q) => mul(a, a);
const abs = (a: Q): Q => a.sign ? { sign: 1, log: a.log } : zero;
function sum(...qs: Q[]): Q {
  let result = zero;
  for (const q of qs) {
    if (!q.sign) continue;
    if (!result.sign) { result = q; continue; }
    const hi = result.log >= q.log ? result : q, lo = result.log >= q.log ? q : result;
    result = hi.sign === lo.sign ? { sign: hi.sign, log: hi.log + Math.log1p(Math.exp(lo.log - hi.log)) }
      : hi.log === lo.log ? zero : { sign: hi.sign, log: hi.log + Math.log(-Math.expm1(lo.log - hi.log)) };
  }
  return result;
}
function divide(a: Q, b: Q): Q {
  if (!b.sign) throw new RangeError('Division by zero');
  return a.sign ? { sign: a.sign * b.sign, log: a.log - b.log } : zero;
}
function observe(a: Q): BurgersObservation {
  if (!a.sign) return { value: 0, sign: 0, logMagnitude: null };
  const value = a.sign * Math.exp(a.log);
  return { value: value === 0 || !Number.isFinite(value) ? null : value, sign: a.sign, logMagnitude: a.log };
}
function compare(a: Q, b: Q): number {
  if (a.sign !== b.sign) return Math.sign(a.sign - b.sign);
  return a.sign ? a.sign * Math.sign(a.log - b.log) : 0;
}
const min = (a: Q, b: Q) => compare(a, b) <= 0 ? a : b;
const max = (a: Q, b: Q) => compare(a, b) >= 0 ? a : b;
const state = (q: State) => sum(product(q.initial), q.change);
// Subtract the ordinary initial values before adding changes. This avoids
// losing their resolved small difference by subtracting rounded logarithms.
const difference = (a: State, b: State) => sum(product(a.initial - b.initial), a.change, neg(b.change));
const combination = (a: State, b: State, x: number, y: number) => sum(product(x * a.initial + y * b.initial), scale(a.change, x), scale(b.change, y));
const poly2 = (a: Q, b: Q) => scale(sum(square(a), square(b), square(sum(a, b))), .5);
function bound(x: number, lo: number, hi: number, name: string, integer = false) {
  if (!Number.isFinite(x) || x < lo || x > hi || integer && !Number.isInteger(x)) throw new RangeError(`Invalid ${name}`);
}
function methodCheck(method: BurgersMethod) {
  if (method !== 'godunov' && method !== 'roe') throw new RangeError('Invalid Burgers method');
}
function face(a: State, b: State, method: BurgersMethod) {
  const jump = difference(a, b), speed = scale(combination(a, b, 1, 1), .5);
  let branch: 'left' | 'right' | 'sonic';
  if (method === 'roe' || jump.sign > 0) branch = speed.sign >= 0 ? 'left' : 'right';
  else branch = state(a).sign >= 0 ? 'left' : state(b).sign <= 0 ? 'right' : 'sonic';
  const selected: State = branch === 'sonic' ? { initial: 0, change: zero } : branch === 'left' ? a : b;
  const base = product(.5, selected.initial, selected.initial);
  const increment = sum(scale(selected.change, selected.initial), scale(square(selected.change), .5));
  const flux = sum(base, increment), left = state(a), right = state(b);
  const defect = branch === 'left' ? scale(mul(square(jump), combination(a, b, 2, 1)), 1 / 6)
    : branch === 'right' ? scale(mul(square(jump), combination(a, b, 1, 2)), -1 / 6)
    : scale(mul(neg(jump), poly2(left, right)), 1 / 6);
  const entropyFlux = branch === 'sonic'
    ? scale(mul(combination(a, b, 1, 1), poly2(left, neg(right))), -1 / 12)
    : scale(mul(square(combination(a, b, 1, 1)), branch === 'left' ? combination(a, b, 2, -1) : combination(a, b, -1, 2)), 1 / 12);
  return { selected, branch, flux, defect, entropyFlux, speed };
}
/** Interface diagnostics, with a declared left trace for a zero-speed shock. */
export function burgersInterface(left: number, right: number, method: BurgersMethod = 'godunov') {
  bound(left, -8, 8, 'left state'); bound(right, -8, 8, 'right state'); methodCheck(method);
  const result = face({ initial: left, change: zero }, { initial: right, change: zero }, method);
  return { branch: result.branch, flux: observe(result.flux), entropyDefect: observe(result.defect), numericalEntropyFlux: observe(result.entropyFlux), roeSpeed: observe(result.speed), semidiscreteEntropyAdmissible: result.defect.sign >= 0 };
}
export interface BurgersParameters {
  cellCountExponent: number;
  length: number;
  leftState: number;
  rightState: number;
  interfaceFraction: number;
  timeStep: number;
  steps: number;
}
function validate(p: BurgersParameters) {
  bound(p.cellCountExponent, 3, 6, 'cell count exponent', true); bound(p.length, .5, 4, 'length');
  bound(p.leftState, -2, 2, 'left state'); bound(p.rightState, -2, 2, 'right state');
  if (![.25, .5, .75].includes(p.interfaceFraction)) throw new RangeError('Invalid interface fraction');
  bound(p.timeStep, 0, .25, 'time step'); bound(p.steps, 0, 64, 'step count', true);
}
/** Integrate the full-line Riemann solution over one unit-width cell in x/dx.
 * Returns changes from the aligned initial state separately, including tiny t.
 */
function exactCell(j: number, left: number, right: number, time: Q, dx: number) {
  const initial = j < 0 ? left : right;
  if (left === right || !time.sign) return { initial, change: zero, entropyChange: zero };
  const lo = product(j), hi = product(j + 1), ratio = divide(time, product(dx));
  if (left > right) {
    const front = scale(ratio, left + right, .5);
    let width: Q;
    if (j < 0) width = compare(front, lo) <= 0 ? product(1) : compare(front, hi) >= 0 ? zero : sum(hi, neg(front));
    else width = compare(front, hi) >= 0 ? product(1) : compare(front, lo) <= 0 ? zero : sum(front, neg(lo));
    const other = j < 0 ? right : left;
    return { initial, change: scale(width, other - initial), entropyChange: scale(width, other - initial, other + initial, .5) };
  }
  const waveLeft = scale(ratio, left), waveRight = scale(ratio, right);
  const widthLeft = max(zero, sum(min(hi, waveLeft), neg(lo)));
  const widthRight = max(zero, sum(hi, neg(max(lo, waveRight))));
  const fanLo = max(lo, waveLeft), fanHi = min(hi, waveRight), fanWidth = max(zero, sum(fanHi, neg(fanLo)));
  const baseWave = j < 0 ? waveLeft : waveRight;
  let fanChange = zero, fanEntropyChange = zero;
  if (fanWidth.sign) {
    const u = divide(sum(fanLo, neg(baseWave)), ratio), v = divide(sum(fanHi, neg(baseWave)), ratio);
    fanChange = scale(mul(fanWidth, sum(u, v)), .5);
    fanEntropyChange = mul(fanWidth, sum(scale(sum(u, v), initial, .5), scale(poly2(u, v), 1 / 6)));
  }
  return { initial,
    change: sum(scale(widthLeft, left - initial), scale(widthRight, right - initial), fanChange),
    entropyChange: sum(scale(widthLeft, left - initial, left + initial, .5), scale(widthRight, right - initial, right + initial, .5), fanEntropyChange) };
}
/** L1 error of the piecewise-constant reconstruction against the continuous
 * exact profile. This differs from the error of cell averages on coarse cells.
 */
function reconstructionError(q: State, j: number, left: number, right: number, time: Q, dx: number) {
  const toLeft = abs(difference(q, { initial: left, change: zero }));
  const toRight = abs(difference(q, { initial: right, change: zero }));
  if (!time.sign || left === right) return scale(j < 0 ? toLeft : toRight, dx);
  const lo = product(j), hi = product(j + 1), ratio = divide(time, product(dx));
  if (left > right) {
    const front = scale(ratio, left + right, .5);
    const widthLeft = max(zero, sum(min(hi, front), neg(lo)));
    const widthRight = max(zero, sum(hi, neg(max(lo, front))));
    return scale(sum(mul(widthLeft, toLeft), mul(widthRight, toRight)), dx);
  }
  const waveLeft = scale(ratio, left), waveRight = scale(ratio, right);
  const widthLeft = max(zero, sum(min(hi, waveLeft), neg(lo)));
  const widthRight = max(zero, sum(hi, neg(max(lo, waveRight))));
  const fanLo = max(lo, waveLeft), fanHi = min(hi, waveRight);
  let fan = zero;
  if (compare(fanHi, fanLo) > 0) {
    const root = mul(state(q), ratio), middle = max(fanLo, min(fanHi, root));
    const a = abs(sum(middle, neg(fanLo))), b = abs(sum(fanHi, neg(middle)));
    if (compare(root, fanLo) < 0 || compare(root, fanHi) > 0) {
      const width = sum(fanHi, neg(fanLo));
      const distance = abs(sum(scale(sum(fanHi, fanLo), .5), neg(root)));
      fan = divide(mul(width, distance), ratio);
    } else fan = scale(divide(sum(square(a), square(b)), ratio), .5);
  }
  return scale(sum(mul(widthLeft, toLeft), mul(widthRight, toRight), fan), dx);
}
function discreteStep(cells: State[], ghosts: [State, State] | null, dx: number, dt: number, method: BurgersMethod) {
  const n = cells.length;
  const pairs: [State, State][] = ghosts ? [[ghosts[0], cells[0]], ...cells.slice(0, -1).map((c, i): [State, State] => [c, cells[i + 1]]), [cells[n - 1], ghosts[1]]]
    : cells.map((c, i): [State, State] => [c, cells[(i + 1) % n]]);
  const faces = pairs.map(([a, b]) => face(a, b, method));
  // Separate the initial selected flux and its change. Evaluate differences of
  // the initial squares by factorization, rather than subtracting two logs.
  const fluxDifference = (l: State, r: State) => {
    const initialDifference = product(.5, l.initial - r.initial, l.initial + r.initial);
    const correction = (q: State) => sum(scale(q.change, q.initial), scale(square(q.change), .5));
    return sum(initialDifference, correction(l), neg(correction(r)));
  };
  const baseRates = cells.map((_, i) => {
    const a = faces[ghosts ? i : (i + n - 1) % n].selected.initial;
    const b = faces[ghosts ? i + 1 : i].selected.initial;
    return product(.5, a - b, a + b, 1 / dx);
  });
  const changedRates = cells.map((_, i) => {
    const a = faces[ghosts ? i : (i + n - 1) % n].selected;
    const b = faces[ghosts ? i + 1 : i].selected;
    const correction = (q: State) => sum(scale(q.change, q.initial), scale(square(q.change), .5));
    return scale(sum(correction(a), neg(correction(b))), 1 / dx);
  });
  const rates = baseRates.map((r, i) => sum(r, changedRates[i]));
  const increments = rates.map(r => scale(r, dt));
  const diagnostics = cells.map((q, i) => {
    const l = faces[ghosts ? i : (i + n - 1) % n], r = faces[ghosts ? i + 1 : i];
    const defectSum = sum(l.defect, r.defect), correction = scale(square(rates[i]), .5, dx, dt, dt);
    const entropyChange = scale(mul(increments[i], sum(product(q.initial), q.change, scale(increments[i], .5))), dx);
    const localFromFaces = sum(scale(defectSum, -.5, dt), correction);
    const localAssembled = sum(entropyChange, scale(sum(r.entropyFlux, neg(l.entropyFlux)), dt));
    const margin = sum(defectSum, neg(scale(square(rates[i]), dt, dx)));
    return { entropyChange, correction, localFromFaces, localAssembled, margin };
  });
  const maxState = [...cells, ...(ghosts || [])].map(q => abs(state(q))).reduce(max, zero);
  const courant = scale(maxState, dt, 1 / dx);
  const boundaryMass = ghosts ? scale(fluxDifference(faces[0].selected, faces[n].selected), dt) : zero;
  // At each external face both initial states equal the same far-field C.
  // Expand the numerical entropy flux around C before taking its boundary
  // difference. This retains tiny boundary changes against a large background.
  function boundaryEntropyChange(index: number) {
    const [a, b] = pairs[index], C = a.initial;
    const u = a.change, v = b.change;
    const w = difference(faces[index].selected, { initial: C, change: zero });
    return sum(scale(w, C, C), scale(square(w), .5, C),
      scale(mul(sum(u, v), w), .5, C), scale(mul(sum(u, v), square(w)), .25),
      scale(sum(square(u), square(v)), -.25, C),
      scale(sum(mul(square(u), u), mul(square(v), v)), -1 / 12));
  }
  const boundaryEntropy = ghosts ? scale(sum(
    scale(mul(product(pairs[0][0].initial - pairs[n][1].initial),
      poly2(product(pairs[0][0].initial), product(pairs[n][1].initial))), 1 / 3),
    boundaryEntropyChange(0), neg(boundaryEntropyChange(n))), dt) : zero;
  const spatial = sum(...diagnostics.map((_, i) => {
    const l = faces[ghosts ? i : (i + n - 1) % n], r = faces[ghosts ? i + 1 : i];
    return scale(sum(l.defect, r.defect), -.5, dt);
  }));
  // Algebraically regroup the shared-face work before summing its large
  // boundary and dissipation components. Their separate rounded sum can hide
  // an actual entropy increment hundreds of orders of magnitude smaller.
  const sharedWork = sum(
    sum(...cells.map((q, i) => scale(baseRates[i], q.initial, dx))),
    sum(...cells.map((q, i) => scale(changedRates[i], q.initial, dx))),
    sum(...cells.map((q, i) => scale(mul(q.change, baseRates[i]), dx))),
    sum(...cells.map((q, i) => scale(mul(q.change, changedRates[i]), dx))));
  const positiveCorrection = sum(...diagnostics.map(d => d.correction));
  const entropyIncrementCombined = sum(scale(sharedWork, dt), positiveCorrection);
  return { faces, rates, increments, diagnostics, courant, boundaryMass, boundaryEntropy, spatial, entropyIncrementCombined,
    correction: sum(...diagnostics.map(d => d.correction)), entropyChange: sum(...diagnostics.map(d => d.entropyChange)),
    next: cells.map((q, i) => ({ initial: q.initial, change: sum(q.change, increments[i]) })) };
}
function faceObservations(f: ReturnType<typeof face>) {
  return { branch: f.branch, flux: observe(f.flux), entropyDefect: observe(f.defect), numericalEntropyFlux: observe(f.entropyFlux), semidiscreteEntropyAdmissible: f.defect.sign >= 0 };
}
function stepObservations(s: ReturnType<typeof discreteStep>) {
  return { faces: s.faces.map(faceObservations), rates: s.rates.map(observe), increments: s.increments.map(observe),
    local: s.diagnostics.map(d => ({ entropyChange: observe(d.entropyChange), positiveEulerCorrection: observe(d.correction), entropyResidualFromFaces: observe(d.localFromFaces), assembledEntropyResidual: observe(d.localAssembled), sufficientMargin: observe(d.margin), sufficientEntropyCondition: d.margin.sign >= 0 })),
    cfl: observe(s.courant), cflAtMostOne: !s.courant.sign || s.courant.log <= 0,
    boundaryMassIncrement: observe(s.boundaryMass), boundaryEntropyIncrement: observe(s.boundaryEntropy), spatialEntropyIncrement: observe(s.spatial), positiveEulerCorrection: observe(s.correction), entropyIncrementFromCells: observe(s.entropyChange), entropyIncrementFromFaces: observe(s.entropyIncrementCombined), roundedEntropyComponentSum: observe(sum(s.boundaryEntropy, s.spatial, s.correction)) };
}
/** One periodic step supports the local Roe counterexample and conservation control. */
export function burgersPeriodicStep(values: number[], dx: number, dt: number, method: BurgersMethod = 'godunov') {
  bound(values.length, 2, 64, 'cell count', true); bound(dx, .001, 4, 'cell width'); bound(dt, 0, .25, 'time step'); methodCheck(method);
  for (let i = 0; i < values.length; i++) bound(values[i], -4, 4, 'cell state');
  const s = discreteStep(values.map(initial => ({ initial, change: zero })), null, dx, dt, method);
  return { ...stepObservations(s), values: s.next.map(q => observe(state(q))), massChangeFromFaces: observe(zero), assembledMassChange: observe(scale(sum(...s.increments), dx)) };
}
/** Nonconservative advective positive-state control. Its mass defect is measured. */
export function burgersAdvectivePositiveStep(values: number[], dx: number, dt: number) {
  bound(values.length, 2, 64, 'cell count', true); bound(dx, .001, 4, 'cell width'); bound(dt, 0, .25, 'time step');
  for (let i = 0; i < values.length; i++) bound(values[i], 0, 4, 'positive cell state');
  const changes = values.map((q, i) => product(-dt, 1 / dx, q, q - values[(i + values.length - 1) % values.length]));
  const defect = product(-.5, dt);
  const jumpSquares = values.map((q, i) => product(q - values[(i + values.length - 1) % values.length], q - values[(i + values.length - 1) % values.length]));
  return { values: values.map((q, i) => observe(sum(product(q), changes[i]))), changes: changes.map(observe), assembledMassChange: observe(scale(sum(...changes), dx)), massDefectIdentity: observe(mul(defect, sum(...jumpSquares))) };
}
/** Finite interval driven by full-line exact ghost cell averages at each old time.
 * The numerical boundary ledger uses actual numerical fluxes, not the continuum
 * boundary flux. The exact full-line reference remains valid after waves exit.
 */
export function burgersRiemannSimulation(p: BurgersParameters) {
  validate(p);
  const n = 2 ** p.cellCountExponent, cut = n * p.interfaceFraction, dx = p.length / n;
  const initial = Array.from({ length: n }, (_, i) => i < cut ? p.leftState : p.rightState);
  const initialMass = sum(product(p.leftState, dx, cut), product(p.rightState, dx, n - cut));
  const initialEntropy = sum(product(.5, p.leftState, p.leftState, dx, cut), product(.5, p.rightState, p.rightState, dx, n - cut));
  const initialCells = initial.map(value => ({ initial: value, change: zero }));
  const initialGhosts: [State, State] = [{ initial: p.leftState, change: zero }, { initial: p.rightState, change: zero }];
  const initialGodunov = discreteStep(initialCells, initialGhosts, dx, p.timeStep, 'godunov');
  function confined(time: Q) {
    const speed = p.leftState > p.rightState ? abs(product(p.leftState + p.rightState, .5)) : max(abs(product(p.leftState)), abs(product(p.rightState)));
    return compare(scale(mul(time, speed), 1 / dx), product(1)) <= 0;
  }
  function referenceCell(j: number, time: Q) {
    if (!confined(time)) return exactCell(j, p.leftState, p.rightState, time, dx);
    const value = j < 0 ? p.leftState : p.rightState;
    const i = j + cut;
    const rate = i >= 0 && i < n ? initialGodunov.rates[i] : zero;
    let entropyRate = zero;
    if (p.leftState > p.rightState) {
      const sign = Math.sign(p.leftState + p.rightState);
      if (sign > 0 && j === 0 || sign < 0 && j === -1)
        entropyRate = product(.25, p.leftState - p.rightState, p.leftState + p.rightState, p.leftState + p.rightState, 1 / dx);
    } else if (j === -1 || j === 0) {
      const selected = initialGodunov.faces[cut].selected.initial;
      const a = j < 0 ? p.leftState : selected, b = j < 0 ? selected : p.rightState;
      entropyRate = scale(mul(product(a - b), poly2(product(a), product(b))), 1 / 3, 1 / dx);
    }
    return { initial: value, change: mul(time, rate), entropyChange: mul(time, entropyRate) };
  }
  function run(method: BurgersMethod) {
    let cells: State[] = initial.map(value => ({ initial: value, change: zero }));
    const referenceStep = discreteStep(initialCells, initialGhosts, dx, p.timeStep, method);
    let remainders: Q[] = Array(n).fill(zero);
    const initialRateDifference = initial.map((_, i) => {
      const differenceAtFace = (j: number) => {
        const a = referenceStep.faces[j].selected.initial, b = initialGodunov.faces[j].selected.initial;
        return product(.5, a - b, a + b);
      };
      return scale(sum(differenceAtFace(i), neg(differenceAtFace(i + 1))), 1 / dx);
    });
    let massLedger = zero, boundaryEntropyLedger = zero, spatialLedger = zero, timeLedger = zero, entropyCellLedger = zero, entropyCombinedLedger = zero;
    const history = [];
    let allCflAtMostOne = true, allLocalEntropySufficient = true, allFacesSemidiscreteAdmissible = true;
    for (let step = 0; step <= p.steps; step++) {
      const time = product(p.timeStep, step), exact = initial.map((_, i) => referenceCell(i - cut, time));
      const ghosts: [State, State] = [referenceCell(-cut - 1, time), referenceCell(n - cut, time)];
      const current = discreteStep(cells, ghosts, dx, p.timeStep, method);
      const errors = cells.map((q, i) => confined(time)
        ? sum(remainders[i], mul(time, initialRateDifference[i])) : sum(q.change, neg(exact[i].change)));
      const assembledChange = scale(sum(...cells.map(q => q.change)), dx);
      const entropyStableChange = scale(sum(...cells.map(q => mul(q.change, sum(product(q.initial), scale(q.change, .5))))), dx);
      const summary = { step, time: observe(time), cfl: observe(current.courant), massFromBoundaryLedger: observe(sum(initialMass, massLedger)), massChangeFromBoundary: observe(massLedger),
        assembledMass: observe(sum(initialMass, assembledChange)), assembledMassChange: observe(assembledChange), absoluteCellMassSum: observe(scale(sum(...cells.map(q => abs(state(q)))), dx)),
        quadraticEntropy: observe(sum(initialEntropy, entropyStableChange)), stableEntropyChange: observe(entropyStableChange), accumulatedEntropyChangeFromCells: observe(entropyCellLedger),
        boundaryEntropyLedger: observe(boundaryEntropyLedger), spatialEntropyLedger: observe(spatialLedger), positiveEulerLedger: observe(timeLedger), entropyChangeFromFaceLedger: observe(entropyCombinedLedger), roundedEntropyComponentSum: observe(sum(boundaryEntropyLedger, spatialLedger, timeLedger)),
        exactMassChange: observe(scale(sum(...exact.map(q => q.change)), dx)), exactContinuumEntropyChange: observe(scale(sum(...exact.map(q => q.entropyChange)), dx)),
        l1CellAverageError: observe(scale(sum(...errors.map(abs)), dx)), l1ReconstructedProfileError: observe(sum(...cells.map((q, i) => reconstructionError(q, i - cut, p.leftState, p.rightState, time, dx)))), linfinityCellAverageError: observe(errors.map(abs).reduce(max, zero)) };
      history.push(summary);
      if (step === p.steps) {
        const midpoint: State = { initial: p.leftState, change: product(p.rightState - p.leftState, .5) }, crossings: number[] = [];
        if (p.leftState > p.rightState) for (let i = 0; i < n - 1; i++) {
          const a = difference(cells[i], midpoint), b = difference(cells[i + 1], midpoint), jump = difference(cells[i + 1], cells[i]);
          if (a.sign >= 0 && b.sign <= 0 && jump.sign) {
            const fraction = observe(divide(neg(a), jump)).value;
            if (fraction !== null) { const x = (i + .5 + fraction) * dx; if (!crossings.includes(x)) crossings.push(x); }
          }
        }
        return { method, cells: cells.map((q, i) => ({ index: i, left: i * dx, right: (i + 1) * dx, initial: q.initial, value: observe(state(q)), change: observe(q.change), nonlinearRemainder: observe(remainders[i]), exactValue: observe(state(exact[i])), exactChange: observe(exact[i].change), error: observe(errors[i]) })),
          history, final: summary, nextStep: stepObservations(current), allExecutedCflAtMostOne: allCflAtMostOne, allExecutedLocalEntropySufficient: allLocalEntropySufficient, allExecutedFacesSemidiscreteAdmissible: allFacesSemidiscreteAdmissible, thresholdCrossings: crossings };
      }
      allCflAtMostOne &&= !current.courant.sign || current.courant.log <= 0;
      allLocalEntropySufficient &&= current.diagnostics.every(d => d.margin.sign >= 0);
      allFacesSemidiscreteAdmissible &&= current.faces.every(f => f.defect.sign >= 0);
      massLedger = sum(massLedger, current.boundaryMass);
      boundaryEntropyLedger = sum(boundaryEntropyLedger, current.boundaryEntropy);
      spatialLedger = sum(spatialLedger, current.spatial);
      timeLedger = sum(timeLedger, current.correction);
      entropyCellLedger = sum(entropyCellLedger, current.entropyChange);
      entropyCombinedLedger = sum(entropyCombinedLedger, current.entropyIncrementCombined);
      const changedFlux = current.faces.map((f, i) => {
        const selected = f.selected, reference = referenceStep.faces[i].selected.initial;
        return sum(product(.5, selected.initial - reference, selected.initial + reference),
          scale(selected.change, selected.initial), scale(square(selected.change), .5));
      });
      remainders = remainders.map((r, i) => sum(r, scale(sum(changedFlux[i], neg(changedFlux[i + 1])), p.timeStep, 1 / dx)));
      const nextTime = product(p.timeStep, step + 1);
      cells = cells.map((q, i) => ({ initial: q.initial, change: sum(mul(nextTime, referenceStep.rates[i]), remainders[i]) }));
    }
    throw new Error('Unreachable Burgers step count');
  }
  const time = product(p.timeStep, p.steps), speed = product(p.leftState + p.rightState, .5);
  const shock = p.leftState > p.rightState, displacement = mul(speed, time);
  const physicalShockPosition = shock ? sum(product(p.length, p.interfaceFraction), displacement) : zero;
  const shockInsideDomain = shock && compare(physicalShockPosition, zero) > 0 && compare(physicalShockPosition, product(p.length)) < 0;
  return { parameters: p, dx, cellCount: n, initialMass: observe(initialMass), initialQuadraticEntropy: observe(initialEntropy), time: observe(time),
    waveType: p.leftState === p.rightState ? 'constant' : shock ? 'shock' : 'rarefaction', shockSpeed: shock ? observe(speed) : null,
    shockDisplacement: shock ? observe(displacement) : null, exactShockPosition: shock ? observe(physicalShockPosition) : null, shockInsideDomain,
    shockEntropyDefect: shock ? observe(product(-1 / 12, p.leftState - p.rightState, p.leftState - p.rightState, p.leftState - p.rightState)) : null,
    rarefactionLeftPosition: p.leftState < p.rightState ? observe(sum(product(p.length, p.interfaceFraction), scale(time, p.leftState))) : null,
    rarefactionRightPosition: p.leftState < p.rightState ? observe(sum(product(p.length, p.interfaceFraction), scale(time, p.rightState))) : null,
    godunov: run('godunov'), roe: run('roe') };
}
