import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { burgersInterface, burgersRiemannSimulation, burgersPeriodicStep, burgersAdvectivePositiveStep, type BurgersParameters, type BurgersObservation } from '../src/lib/learning-continuum-godunov';
interface Ref { value: string; sign: number; logMagnitude: string | null }
const fixture = JSON.parse(readFileSync('scripts/fixtures/continuum-godunov-independent.json', 'utf8'));
const refLog = (v: Ref) => v.logMagnitude === null ? -Infinity : Number(v.logMagnitude);
function compare(a: BurgersObservation, e: Ref, label: string, scale = -Infinity, digits = 160) {
  assert.ok(!a.sign ? a.value === 0 && a.logMagnitude === null : a.logMagnitude !== null && Number.isFinite(a.logMagnitude), label + ' representation');
  const log = refLog(e);
  if (!e.sign || log < scale - (digits - 60) * Math.LN10) {
    assert.ok(!a.sign || a.logMagnitude! < scale - 24, label + ' zero/cancellation ' + JSON.stringify(a)); return;
  }
  assert.equal(a.sign, e.sign, label + ' sign');
  assert.ok(Math.abs(a.logMagnitude! - log) < 3e-8, label + ' log ' + a.logMagnitude + ' != ' + log);
}
for (const [index, c] of fixture.interfaces.entries()) test(`independent convex-flux interface ${index + 1}`, () => {
  const s = burgersInterface(c.left, c.right, c.method);
  for (const key of ['flux', 'entropyDefect', 'numericalEntropyFlux'] as const) compare(s[key], c[key], `interface ${index} ${key}`, 3 * Math.log(Math.max(Math.abs(c.left), Math.abs(c.right))), 1600);
  if (c.method === 'godunov') assert.equal(s.semidiscreteEntropyAdmissible, true);
});
for (const [index, c] of [...fixture.ordinaryCases, ...fixture.limitCases].entries()) test(`independent nonlinear trajectory and exact Riemann cell integrals ${index + 1}`, () => {
  const result = burgersRiemannSimulation(c.parameters), digits = index < 24 ? 160 : 1600;
  for (const method of ['godunov', 'roe'] as const) {
    const s = result[method], r = c[method];
    const valueScale = Math.max(...r.cells.map((q: Record<string, Ref>) => refLog(q.value)));
    const changeScale = Math.max(...r.cells.map((q: Record<string, Ref>) => refLog(q.change)), ...r.cells.map((q: Record<string, Ref>) => refLog(q.exactChange)));
    for (const key of ['value', 'change', 'exactValue', 'exactChange', 'error'] as const) {
      const scale = key === 'value' || key === 'exactValue' ? valueScale : changeScale;
      r.cells.forEach((v: Record<string, Ref>, i: number) => compare(s.cells[i][key], v[key], `case ${index} ${method} cell ${i} ${key}`, scale, digits));
    }
    const entropyScale = Math.max(...['boundaryEntropyLedger','spatialEntropyLedger','positiveEulerLedger','stableEntropyChange'].map(key => refLog(r.final[key])));
    for (const [key, value] of Object.entries(r.final) as [keyof typeof s.final, Ref][]) {
      const scale = /entropy|Euler/i.test(key) ? entropyScale : /Mass|mass/.test(key) ? Math.max(valueScale + Math.log(c.parameters.length), changeScale) : changeScale;
      compare(s.final[key] as BurgersObservation, value, `case ${index} ${method} summary ${key}`, scale, digits);
    }
    r.nextStep.faces.forEach((f: Record<string, Ref>, i: number) => {
      for (const key of ['flux','entropyDefect','numericalEntropyFlux'] as const) compare(s.nextStep.faces[i][key], f[key], `case ${index} ${method} face ${i} ${key}`, (key === 'flux' ? 2 : 3) * valueScale, digits);
    });
    r.nextStep.rates.forEach((v: Ref, i: number) => compare(s.nextStep.rates[i], v, `case ${index} ${method} rate ${i}`, 2 * valueScale - Math.log(result.dx), digits));
    r.nextStep.local.forEach((d: Record<string, Ref>, i: number) => {
      const scale = Math.max(...Object.values(d).map(refLog), 3 * valueScale + Math.log(c.parameters.timeStep || 1));
      for (const key of ['entropyChange','positiveEulerCorrection','entropyResidualFromFaces','sufficientMargin'] as const) compare(s.nextStep.local[i][key], d[key], `case ${index} ${method} local ${i} ${key}`, scale, digits);
    });
    compare(s.nextStep.cfl, r.nextStep.cfl, `case ${index} ${method} CFL`);
    if (s.allExecutedCflAtMostOne !== r.allExecutedCflAtMostOne) {
      assert.equal(r.allExecutedCflAtMostOne, true);
      assert.ok(s.history.every(h => h.cfl.value! <= 1 + 1e-13));
    }
    assert.equal(s.allExecutedFacesSemidiscreteAdmissible, r.allExecutedFacesSemidiscreteAdmissible);
    assert.equal(s.history.length,c.parameters.steps+1);
  }
});
const base: BurgersParameters = { cellCountExponent: 4, length: 2, leftState: -1, rightState: 1, interfaceFraction: .5, timeStep: .01, steps: 16 };
test('sonic Roe preserves a stationary expansion jump while Godunov creates the entropy rarefaction', () => {
  const s = burgersRiemannSimulation(base);
  assert.ok(s.roe.cells.every(q=>q.change.sign===0));
  assert.ok(s.godunov.cells.some(q=>q.change.sign!==0));
  assert.equal(s.roe.allExecutedFacesSemidiscreteAdmissible,false);
  assert.equal(s.godunov.allExecutedFacesSemidiscreteAdmissible,true);
  assert.equal(burgersInterface(-1,1).flux.sign,0);
  assert.ok(Math.abs(burgersInterface(-1,1,'roe').flux.value!-.5)<1e-15);
  assert.ok(s.godunov.final.l1CellAverageError.value!<s.roe.final.l1CellAverageError.value!);
});
test('a global periodic entropy sum can hide a local Roe defect', () => {
  const s=burgersPeriodicStep([-1,1],1,.1,'roe');
  assert.equal(s.massChangeFromFaces.sign,0);assert.equal(s.entropyIncrementFromCells.sign,0);
  assert.ok(s.faces.some(f=>f.entropyDefect.sign<0));assert.ok(s.faces.some(f=>f.entropyDefect.sign>0));
  assert.ok(s.local.every(c=>c.sufficientEntropyCondition)); // Adjacent opposite defects cancel even in each two-cell balance.
});
test('positive nonconservative advection loses exactly the predicted mass while shared Godunov fluxes cancel', () => {
  const values=[1,.5,0,.25],dx=.5,dt=.1;
  const wrong=burgersAdvectivePositiveStep(values,dx,dt),correct=burgersPeriodicStep(values,dx,dt);
  assert.ok(wrong.assembledMassChange.value!<0);
  assert.ok(Math.abs(wrong.assembledMassChange.value!-wrong.massDefectIdentity.value!)<1e-14);
  assert.equal(correct.massChangeFromFaces.sign,0);
  assert.ok(!correct.assembledMassChange.sign||Math.abs(correct.assembledMassChange.value!)<1e-14);
});
test('stationary compressive shocks retain their state and distinct physical and numerical entropy fluxes', () => {
  const s=burgersRiemannSimulation({...base,leftState:1,rightState:-1});
  assert.equal(s.waveType,'shock');assert.equal(s.shockSpeed!.sign,0);
  assert.ok(s.godunov.cells.every(q=>q.change.sign===0));
  assert.equal(burgersInterface(1,-1).numericalEntropyFlux.sign,0);
  assert.ok(Math.abs(s.shockEntropyDefect!.value!+2/3)<1e-14);
});
test('tiny state/time products survive below ordinary range without changing null into zero', () => {
  const s=burgersRiemannSimulation({...base,leftState:Number.MIN_VALUE,rightState:0,timeStep:Number.MIN_VALUE});
  assert.ok(s.godunov.cells.some(c=>c.change.sign!==0&&c.change.value===null&&c.change.logMagnitude! < -2000));
  assert.ok(s.godunov.nextStep.faces.some(f=>f.flux.sign!==0&&f.flux.value===null));
  const periodic=burgersPeriodicStep([Number.MIN_VALUE,0],4,Number.MIN_VALUE);
  assert.ok(periodic.cfl.sign>0&&periodic.cfl.value===null);
});
test('invalid methods, sparse arrays, nonfinite and missing controls are rejected', () => {
  for(const key of Object.keys(base))for(const invalid of [NaN,Infinity,-Infinity,undefined])assert.throws(()=>burgersRiemannSimulation({...base,[key]:invalid}as BurgersParameters),RangeError);
  for(const patch of [{steps:65},{timeStep:-1},{leftState:3},{interfaceFraction:.3},{cellCountExponent:3.5}])assert.throws(()=>burgersRiemannSimulation({...base,...patch}),RangeError);
  assert.throws(()=>burgersInterface(1,0,'bad' as 'roe'),RangeError);
  assert.throws(()=>burgersPeriodicStep([1,,0]as number[],1,.1),RangeError);
  assert.throws(()=>burgersAdvectivePositiveStep([1,-1],1,.1),RangeError);
});

test('shared-face Euler entropy identity closes locally and globally for actual periodic updates', () => {
  for (const method of ['godunov','roe'] as const) for(let seed=1;seed<=20;seed++) {
    const q=Array.from({length:8},(_,i)=>1.8*Math.sin(seed*1.3+i*.7));
    const s=burgersPeriodicStep(q,.2,.003,method);
    const entropyChange=s.values.reduce((acc,v,i)=>acc+.1*(v.value!**2-q[i]**2),0);
    assert.ok(Math.abs(entropyChange-s.entropyIncrementFromFaces.value!)<1e-13);
    for (const cell of s.local) assert.ok(Math.abs(cell.assembledEntropyResidual.value!-cell.entropyResidualFromFaces.value!)<1e-13);
    assert.ok(!s.assembledMassChange.sign||Math.abs(s.assembledMassChange.value!)<1e-13);
  }
});
test('the exact hand stencil has the stated mass and finite Euler entropy budgets',()=>{
  const s=burgersPeriodicStep([-1,-1,1,1],1,.1);
  const expected=[-1,-.95,.95,1];s.values.forEach((v,i)=>assert.ok(Math.abs(v.value!-expected[i])<1e-14));
  assert.ok(Math.abs(s.entropyIncrementFromCells.value!+.0975)<1e-14);
  assert.ok(Math.abs(s.positiveEulerCorrection.value!-.0025)<1e-14);
  const w=burgersAdvectivePositiveStep([1,.5,0,.25],.5,.1);assert.ok(Math.abs(w.massDefectIdentity.value!+.05625)<1e-14);
});
test('reconstructed shock error and measured threshold position converge at a fixed physical time',()=>{
  const runs=[3,4,5,6].map(cellCountExponent=>burgersRiemannSimulation({...base,cellCountExponent,leftState:1.5,rightState:.5,timeStep:.001,steps:64}));
  for(let i=1;i<runs.length;i++){
    const a=runs[i-1],b=runs[i];assert.equal(b.godunov.thresholdCrossings.length,1);
    assert.ok(b.godunov.final.l1ReconstructedProfileError.value!<a.godunov.final.l1ReconstructedProfileError.value!);
    assert.ok(Math.abs(b.godunov.thresholdCrossings[0]-b.exactShockPosition!.value!)<Math.abs(a.godunov.thresholdCrossings[0]-a.exactShockPosition!.value!));
    assert.equal(b.godunov.allExecutedCflAtMostOne,true);
  }
});
test('waves exiting both ends retain the full-line reference and actual numerical boundary ledger',()=>{
  for(const [leftState,rightState]of [[1,.5],[-.5,-1],[.5,2],[-2,-.5]]){
    const s=burgersRiemannSimulation({...base,cellCountExponent:3,leftState,rightState,timeStep:.1,steps:32});
    for(const method of ['godunov','roe']as const){
      const r=s[method];assert.ok(Math.abs(r.final.assembledMassChange.value!-r.final.massChangeFromBoundary.value!)<1e-12);
      assert.ok(r.final.l1ReconstructedProfileError.sign>=0);
    }
    if(leftState>rightState)assert.equal(s.shockInsideDomain,false);
  }
});
test('excessive prescribed steps run unchanged and extreme output trees contain no NaN or infinity',()=>{
  const s=burgersRiemannSimulation({...base,cellCountExponent:6,length:.5,leftState:2,rightState:0,timeStep:.25,steps:64});
  assert.equal(s.parameters.timeStep,.25);assert.equal(s.godunov.allExecutedCflAtMostOne,false);
  assert.ok(s.godunov.cells.some(c=>c.value.value===null&&c.value.sign!==0));
  const inspect=(x:unknown):void=>{if(typeof x==='number')assert.ok(Number.isFinite(x));else if(x&&typeof x==='object')Object.values(x).forEach(inspect);};inspect(s);
  assert.doesNotThrow(()=>JSON.stringify(s));
});

test('constant states have neither a shock nor a fictitious rarefaction boundary',()=>{
  const s=burgersRiemannSimulation({...base,leftState:1,rightState:1});
  assert.equal(s.waveType,'constant');assert.equal(s.exactShockPosition,null);
  assert.equal(s.rarefactionLeftPosition,null);assert.equal(s.rarefactionRightPosition,null);
  assert.ok(s.godunov.cells.every(c=>c.change.sign===0));
});
