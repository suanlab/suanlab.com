/** Shared-face advection/diffusion on a nonuniform periodic finite-volume ring. */
export interface VolumeObservation {
    value: number | null;
    sign: number;
    logMagnitude: number | null;
}
interface Q {
    sign: number;
    log: number;
}
const zero: Q = { sign: 0, log: 0 }, ln2 = Math.log(2);
function product(...xs: number[]): Q { return xs.includes(0) ? zero : { sign: xs.reduce((s, x) => s * Math.sign(x), 1), log: xs.reduce((s, x) => s + Math.log(Math.abs(x)), 0) }; }
const neg = (q: Q): Q => q.sign ? { sign: -q.sign, log: q.log } : zero;
const mul = (a: Q, b: Q): Q => a.sign && b.sign ? { sign: a.sign * b.sign, log: a.log + b.log } : zero;
const scale = (q: Q, ...x: number[]) => mul(q, product(...x));
const square = (q: Q) => mul(q, q);
const shift = (q: Q, l: number): Q => q.sign ? { sign: q.sign, log: q.log + l } : zero;
function sum(...qs: Q[]): Q { let r = zero; for (const q of qs) {
    if (!q.sign)
        continue;
    if (!r.sign) {
        r = q;
        continue;
    }
    const hi = r.log >= q.log ? r : q, lo = r.log >= q.log ? q : r;
    r = hi.sign === lo.sign ? { sign: hi.sign, log: hi.log + Math.log1p(Math.exp(lo.log - hi.log)) } : hi.log === lo.log ? zero : { sign: hi.sign, log: hi.log + Math.log(-Math.expm1(lo.log - hi.log)) };
} return r; }
function observe(q: Q): VolumeObservation { if (!q.sign)
    return { value: 0, sign: 0, logMagnitude: null }; const n = q.log <= Math.log(Number.MIN_VALUE) - ln2 ? 0 : q.sign * Math.exp(q.log); return { value: n === 0 || !Number.isFinite(n) ? null : n, sign: q.sign, logMagnitude: q.log }; }
function bound(x: number, lo: number, hi: number, name: string, integer = false) { if (!Number.isFinite(x) || x < lo || x > hi || integer && !Number.isInteger(x))
    throw new RangeError(`Invalid ${name}`); }
const divide = (q: Q, n: number) => shift(q, -Math.log(n));
const total = (qs: Q[], weights: number[]) => sum(...qs.map((q, i) => scale(q, weights[i])));
const abs = (q: Q): Q => q.sign ? { sign: 1, log: q.log } : zero;
const maxPositive = (qs: Q[]) => qs.reduce((a, b) => !a.sign || b.sign && b.log > a.log ? b : a, zero);
function maximum(qs: Q[]) { return qs.reduce((a, b) => a.sign !== b.sign ? a.sign > b.sign ? a : b : !a.sign ? a : a.sign > 0 ? a.log > b.log ? a : b : a.log < b.log ? a : b); }
function minimum(qs: Q[]) { return qs.reduce((a, b) => a.sign !== b.sign ? a.sign < b.sign ? a : b : !a.sign ? a : a.sign > 0 ? a.log < b.log ? a : b : a.log > b.log ? a : b); }
function angleTrig(angle: Q) { if (!angle.sign)
    return { sin: zero, cos: product(1) }; const x = angle.sign * Math.exp(angle.log); return angle.log < Math.log(1e-8) ? { sin: scale(angle, 1 - x * x / 6), cos: product(1) } : { sin: product(Math.sin(x)), cos: product(Math.cos(x)) }; }
function cyclesTrig(cycles: number) { const f = cycles % 1; if (f === .125 || f === .375 || f === .625 || f === .875) {
    const q = product(Math.SQRT1_2);
    return { sin: f < .5 ? q : neg(q), cos: f < .25 || f > .75 ? q : neg(q) };
} const n = Math.round(4 * f), d = f - n / 4, a = angleTrig(product(2 * Math.PI, d)); switch ((n + 4) % 4) {
    case 0: return a;
    case 1: return { sin: a.cos, cos: neg(a.sin) };
    case 2: return { sin: neg(a.sin), cos: neg(a.cos) };
    default: return { sin: neg(a.cos), cos: a.sin };
} }
const compose = (a: {
    sin: Q;
    cos: Q;
}, b: {
    sin: Q;
    cos: Q;
}) => ({ sin: sum(mul(a.sin, b.cos), mul(a.cos, b.sin)), cos: sum(mul(a.cos, b.cos), neg(mul(a.sin, b.sin))) });
function loss(z: Q): Q { return !z.sign ? zero : { sign: 1, log: z.log < -36 ? z.log : Math.log(-Math.expm1(-Math.exp(z.log))) }; }
function attenuation(z: Q) { return z.sign ? -Math.exp(z.log) : 0; }
interface RingGeometry {
    volumes: number[];
    conductances: Q[];
    velocityArea: Q;
}
interface RingInput {
    volumes: number[];
    conductances: number[];
    velocityArea: number;
}
function operator(g: RingGeometry, q: Q[], denominators = g.volumes) {
    const n = q.length, jumps = q.map((v, i) => sum(v, neg(q[(i + 1) % n])));
    const diffusion = jumps.map((v, i) => mul(v, g.conductances[i]));
    const advection = q.map((v, i) => mul(g.velocityArea.sign >= 0 ? v : q[(i + 1) % n], g.velocityArea));
    const flux = diffusion.map((v, i) => sum(v, advection[i]));
    const rates = flux.map((v, i) => divide(sum(flux[(i + n - 1) % n], neg(v)), denominators[i]));
    const diffusiveLoss = sum(...jumps.map((v, i) => mul(square(v), g.conductances[i])));
    const upwindLoss = sum(...jumps.map(v => scale(mul(square(v), abs(g.velocityArea)), .5)));
    return { jumps, diffusion, advection, flux, rates, diffusiveLoss, upwindLoss };
}
function cfl(g: RingGeometry, dt: number, denominators = g.volumes) { return denominators.map((v, i) => divide(scale(sum(abs(g.velocityArea), g.conductances[(i + denominators.length - 1) % denominators.length], g.conductances[i]), dt), v)); }
/** A shared-face ledger for physical mass under the wrong uniform storage denominator. */
function wrongMassRate(g: RingGeometry, flux: Q[], referenceVolume: number) { return sum(...flux.map((f, i) => scale(f, -(g.volumes[i] - g.volumes[(i + 1) % g.volumes.length]) / referenceVolume))); }
function validateRing(g: RingInput, concentration: number[], dt: number) {
    const n = g.volumes.length;
    bound(n, 3, 64, 'cell count', true);
    if (g.conductances.length !== n || concentration.length !== n)
        throw new RangeError('Wrong ring array length');
    for (let i = 0; i < n; i++) {
        bound(g.volumes[i], 1e-6, 16, 'cell volume');
        bound(g.conductances[i], 0, 1e6, 'face conductance');
        bound(concentration[i], -8, 8, 'initial cell concentration');
    }
    bound(g.velocityArea, -8, 8, 'velocity times area');
    bound(dt, 0, .25, 'time step');
}
/** Explicit one-step diagnostic for an arbitrary declared ring, including the rational teaching example. */
export function sharedVolumeRingStep(volumes: number[], conductances: number[], concentration: number[], velocityArea: number, timeStep: number) {
    const input = { volumes, conductances, velocityArea };
    validateRing(input, concentration, timeStep);
    const g: RingGeometry = { volumes, conductances: conductances.map(v => product(v)), velocityArea: product(velocityArea) };
    const n = volumes.length, referenceVolume = volumes.reduce((a, b) => a + b, 0) / n, q = concentration.map(v => product(v)), current = operator(g, q), wrong = operator(g, q, Array(n).fill(referenceVolume));
    const summarize = (r: typeof current, isWrong: boolean) => {
        const changes = r.rates.map(v => scale(v, timeStep)), next = q.map((v, i) => sum(v, changes[i]));
        const rate = total(r.rates, volumes), normRate = sum(...q.map((v, i) => scale(mul(v, r.rates[i]), volumes[i]))), positive = scale(total(r.rates.map(square), volumes), timeStep, timeStep, .5);
        return { values: next.map(observe), rates: r.rates.map(observe), changes: changes.map(observe), assembledPhysicalMass: observe(total(next, volumes)), assembledMassChange: observe(total(changes, volumes)), massRateFromFaces: observe(isWrong ? wrongMassRate(g, r.flux, referenceVolume) : zero), assembledMassRate: observe(rate), quadraticNorm: observe(scale(total(next.map(square), volumes), .5)), stableNormChange: observe(sum(...changes.map((v, i) => scale(mul(v, sum(q[i], scale(v, .5))), volumes[i])))), normRateFromCells: observe(normRate), positiveEulerCorrection: observe(positive), minimumConcentration: observe(minimum(next)) };
    };
    const courant = cfl(g, timeStep);
    return { referenceVolume, faceFluxes: current.flux.map(observe), diffusiveFluxes: current.diffusion.map(observe), advectiveFluxes: current.advection.map(observe), diffusiveNormLoss: observe(current.diffusiveLoss), upwindNormLoss: observe(current.upwindLoss), normRateFromFaces: observe(neg(sum(current.diffusiveLoss, current.upwindLoss))), localCfl: courant.map(observe), positivitySufficient: courant.every(v => !v.sign || v.log <= 0), correct: summarize(current, false), wrong: summarize(wrong, true) };
}
export interface NonuniformVolumeParameters {
    cellCountExponent: number;
    length: number;
    area: number;
    meshWarp: number;
    velocity: number;
    diffusivity: number;
    meanConcentration: number;
    relativeContrast: number;
    mode: number;
    phaseCycles: number;
    timeStep: number;
    steps: number;
}
function validate(p: NonuniformVolumeParameters) {
    bound(p.cellCountExponent, 3, 6, 'cell-count exponent', true);
    bound(p.length, .25, 4, 'period length');
    bound(p.area, .25, 4, 'cross section');
    bound(p.meshWarp, -.75, .75, 'mesh warp');
    bound(p.velocity, -2, 2, 'transport speed');
    bound(p.diffusivity, 0, 2, 'diffusivity');
    bound(p.meanConcentration, 0, 4, 'mean concentration');
    bound(p.relativeContrast, -1, 1, 'relative contrast');
    bound(p.mode, 1, 3, 'Fourier mode', true);
    bound(p.phaseCycles, 0, 1, 'initial phase cycles');
    bound(p.timeStep, 0, .25, 'time step');
    bound(p.steps, 0, 64, 'step count', true);
}
export function nonuniformVolumeMesh(p: NonuniformVolumeParameters) {
    validate(p);
    const n = 2 ** p.cellCountExponent, L = p.length;
    const edges = Array.from({ length: n + 1 }, (_, i) => i === 0 ? 0 : i === n ? L : L * (i / n + p.meshWarp * Math.sin(2 * Math.PI * i / n) / (2 * Math.PI)));
    const lengths = edges.slice(0, n).map((x, i) => edges[i + 1] - x), centers = lengths.map((h, i) => edges[i] + h / 2), volumes = lengths.map(h => p.area * h);
    const distances = centers.map((x, i) => i === n - 1 ? L - x + centers[0] : centers[i + 1] - x);
    return { cellCount: n, edges, lengths, centers, volumes, distances, conductances: distances.map(d => observe(divide(product(p.diffusivity, p.area), d))) };
}
/** Physical cell averages of the periodic continuum solution; initial state is an actual integral, not a point sample. */
export function nonuniformVolumeSimulation(p: NonuniformVolumeParameters) {
    validate(p);
    const mesh = nonuniformVolumeMesh(p), n = mesh.cellCount, ones = Array(n).fill(1), referenceVolume = mesh.volumes.reduce((a, b) => a + b, 0) / n;
    const g: RingGeometry = { volumes: mesh.volumes, conductances: mesh.conductances.map(c => c.sign ? { sign: 1, log: c.logMagnitude! } : zero), velocityArea: product(p.velocity, p.area) };
    const k = 2 * Math.PI * p.mode / p.length, phase = cyclesTrig(p.phaseCycles), amplitude = product(p.meanConcentration, p.relativeContrast);
    const basis = mesh.lengths.map((h, i) => { const z = k * h / 2, sinc = Math.sin(z) / z, trig = compose(cyclesTrig(p.mode * mesh.centers[i] / p.length), phase); return { sinc, trig }; });
    const initial = basis.map(b => scale(mul(amplitude, b.trig.cos), b.sinc)), mean = product(p.meanConcentration);
    const initialOp = operator(g, initial), initialWrong = operator(g, initial, Array(n).fill(referenceVolume));
    let change: Q[] = Array(n).fill(zero), wrongChange: Q[] = Array(n).fill(zero), wrongMassLedger = zero;
    const courant = cfl(g, p.timeStep), wrongCourant = cfl(g, p.timeStep, Array(n).fill(referenceVolume));
    function values(c: Q[]) { return initial.map((v, i) => sum(v, c[i])); }
    function summary(c: Q[], wrong: boolean) {
        const departure = values(c), massDelta = total(c, mesh.volumes), centeredDelta = sum(...c.map((v, i) => scale(mul(v, sum(initial[i], scale(v, .5))), mesh.volumes[i]))), mass = sum(scale(mean, mesh.volumes.reduce((a, b) => a + b, 0)), total(departure, mesh.volumes));
        const states = departure.map(v => sum(mean, v)), norm = sum(product(.5, p.meanConcentration, p.meanConcentration, mesh.volumes.reduce((a, b) => a + b, 0)), scale(total(departure, mesh.volumes), p.meanConcentration), scale(total(departure.map(square), mesh.volumes), .5));
        return { assembledMass: observe(mass), massFromConservationLedger: observe(sum(product(p.meanConcentration, mesh.volumes.reduce((a, b) => a + b, 0)), wrong ? wrongMassLedger : zero)), absoluteCellMassSum: observe(total(states.map(abs), mesh.volumes)), assembledMassChange: observe(massDelta), massChangeFromFaces: observe(wrong ? wrongMassLedger : zero), unweightedSumChange: observe(total(c, ones)), quadraticNorm: observe(norm), centeredNorm: observe(scale(total(departure.map(square), mesh.volumes), .5)), stableCenteredNormChange: observe(centeredDelta), normChangeFromCells: observe(sum(centeredDelta, scale(massDelta, p.meanConcentration))), normChangeUsingMassLedger: observe(sum(centeredDelta, scale(wrong ? wrongMassLedger : zero, p.meanConcentration))), minimumConcentration: observe(minimum(states)), maximumConcentration: observe(maximum(states)) };
    }
    const history = [];
    for (let step = 0; step <= p.steps; step++) {
        history.push({ step, time: observe(product(p.timeStep, step)), correct: summary(change, false), wrong: summary(wrongChange, true) });
        if (step === p.steps)
            break;
        const dc = operator(g, change), dw = operator(g, wrongChange, Array(n).fill(referenceVolume));
        change = change.map((v, i) => sum(v, scale(sum(initialOp.rates[i], dc.rates[i]), p.timeStep)));
        wrongChange = wrongChange.map((v, i) => sum(v, scale(sum(initialWrong.rates[i], dw.rates[i]), p.timeStep)));
        wrongMassLedger = sum(wrongMassLedger, scale(sum(wrongMassRate(g, initialWrong.flux, referenceVolume), wrongMassRate(g, dw.flux, referenceVolume)), p.timeStep));
    }
    const correctOp = operator(g, change), wrongOp = operator(g, wrongChange, Array(n).fill(referenceVolume)), dep = values(change), wrongDep = values(wrongChange);
    const rates = initialOp.rates.map((v, i) => sum(v, correctOp.rates[i])), wrongRates = initialWrong.rates.map((v, i) => sum(v, wrongOp.rates[i]));
    const time = product(p.timeStep, p.steps), damping = scale(time, p.diffusivity, k, k), drift = scale(time, p.velocity, k), tr = angleTrig(drift), half = angleTrig(scale(drift, .5)), cosChange = scale(square(half.sin), -2);
    const exact = basis.map(b => { const difference = sum(mul(b.trig.cos, cosChange), mul(b.trig.sin, tr.sin)); const delta = scale(sum(shift(difference, attenuation(damping)), neg(mul(b.trig.cos, loss(damping)))), b.sinc); const initialDelta = scale(mul(amplitude, b.trig.cos), b.sinc); return { departure: shift(scale(mul(amplitude, sum(mul(b.trig.cos, tr.cos), mul(b.trig.sin, tr.sin))), b.sinc), attenuation(damping)), change: mul(amplitude, delta), initial: initialDelta }; });
    const errors = change.map((v, i) => sum(v, neg(exact[i].change))), wrongErrors = wrongChange.map((v, i) => sum(v, neg(exact[i].change)));
    const cells = dep.map((v, i) => ({ index: i, left: mesh.edges[i], right: mesh.edges[i + 1], center: mesh.centers[i], volume: mesh.volumes[i], initialDeparture: observe(initial[i]), initialConcentration: observe(sum(mean, initial[i])), departure: observe(v), concentration: observe(sum(mean, v)), change: observe(change[i]), rate: observe(rates[i]), wrongConcentration: observe(sum(mean, wrongDep[i])), wrongChange: observe(wrongChange[i]), wrongRate: observe(wrongRates[i]), exactConcentration: observe(sum(mean, exact[i].departure)), exactDeparture: observe(exact[i].departure), exactChange: observe(exact[i].change), error: observe(errors[i]), wrongError: observe(wrongErrors[i]) }));
    const faceFluxes = initialOp.flux.map((v, i) => ({ index: i, x: mesh.edges[i + 1], distance: mesh.distances[i], conductance: mesh.conductances[i], full: observe(sum(product(p.velocity, p.area, p.meanConcentration), v, correctOp.flux[i])), departure: observe(sum(v, correctOp.flux[i])), change: observe(correctOp.flux[i]) }));
    const diagnosticOp = operator(g, dep), normRate = neg(sum(diagnosticOp.diffusiveLoss, diagnosticOp.upwindLoss)), positiveCorrection = scale(total(rates.map(square), mesh.volumes), .5, p.timeStep, p.timeStep);
    const roundedInitial = cells.every(c => c.initialConcentration.value !== null) ? cells.reduce((s, c) => s + c.volume * c.initialConcentration.value!, 0) : null, roundedFinal = cells.every(c => c.concentration.value !== null) ? cells.reduce((s, c) => s + c.volume * c.concentration.value!, 0) : null;
    return { parameters: p, mesh, referenceVolume, time: observe(time), cells, faces: faceFluxes, history, correct: history.at(-1)!.correct, wrong: history.at(-1)!.wrong,
        localCfl: courant.map(observe), wrongLocalCfl: wrongCourant.map(observe), positivitySufficient: courant.every(c => !c.sign || c.log <= 0), wrongPositivitySufficient: wrongCourant.every(c => !c.sign || c.log <= 0),
        diffusiveNormLoss: observe(diagnosticOp.diffusiveLoss), upwindNormLoss: observe(diagnosticOp.upwindLoss), normRateFromFaces: observe(normRate), normRateFromAssembledCells: observe(sum(...rates.map((v, i) => scale(mul(sum(mean, dep[i]), v), mesh.volumes[i])))), positiveEulerCorrection: observe(positiveCorrection), predictedNextNormChangeFromFaces: observe(sum(scale(normRate, p.timeStep), positiveCorrection)),
        assembledMassRate: observe(total(rates, mesh.volumes)), massRateFromFaces: observe(zero), wrongMassRateFromFaces: observe(sum(wrongMassRate(g, initialWrong.flux, referenceVolume), wrongMassRate(g, wrongOp.flux, referenceVolume))),
        l1CellAverageError: observe(total(errors.map(abs), mesh.volumes)), linfinityCellAverageError: observe(maxPositive(errors.map(abs))), wrongL1CellAverageError: observe(total(wrongErrors.map(abs), mesh.volumes)), roundedMassChange: roundedInitial === null || roundedFinal === null ? null : roundedFinal - roundedInitial };
}
