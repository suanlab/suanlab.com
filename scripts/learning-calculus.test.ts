import assert from 'node:assert/strict';
import test from 'node:test';
import { chainState, jacobianState, neuronState, neuronStep, neuronGradientCheck, parameterNames } from '../src/lib/learning-calculus';
import { defaultCalculus as s, calculusPresets, calculusValues, validCalculus, type CalculusKind } from '../src/components/learning/calculus-model';
import { learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-8)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const vector=(a:number[],b:number[])=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i]));};
test('chain rule reference distinguishes values, gains and finite changes',()=>{
 const c=chainState(1,2,1,.1,.0001);vector([c.u,c.output,c.localInner,c.localOuter,c.derivative,c.actualChange,c.linearChange,c.remainder],[3,9,2,6,12,1.24,1.2,.04]);close(c.numeric,12);
});
for(const delta of [-.5,-.1,0,.01,.1,.5])test(`quadratic chain remainder and central difference: ${delta}`,()=>{
 const c=chainState(.3,-2,1,delta,.001);close(c.remainder,4*delta*delta);close(c.numeric,c.derivative);
});
test('stationary point, sign and constant function retain their separate meanings',()=>{
 const turn=chainState(-.5,2,1,.1,.001);close(turn.derivative,0);close(turn.actualChange,.04);
 const negative=chainState(-1,2,1,.1,.001);close(negative.derivative,-4);close(negative.actualChange,-.36);
 const constant=chainState(1,0,2,.5,.001);close(constant.derivative,0);close(constant.actualChange,0);close(constant.output,4);
});
test('Jacobian reference, exact remainder and Euclidean duality',()=>{
 const j=jacobianState([2,3],[.1,-.2],[1,-1],.0001);vector(j.matrix.flat(),[3,2,4,1]);vector(j.output,[6,7]);vector(j.jvp,[-.1,.2]);vector(j.vjp,[-1,1]);vector(j.actualChange,[-.12,.21]);vector(j.remainder,[-.02,.01]);vector(j.numeric,j.jvp);close(j.dualLeft,-.3);close(j.dualLeft,j.dualRight);
});
test('small displacement scales the first-order and second-order terms differently',()=>{
 const j=jacobianState([2,3],[.01,-.02],[1,-1],.001);vector(j.jvp,[-.01,.02]);vector(j.remainder,[-.0002,.0001]);
});
test('singular Jacobian has a null direction without finite constancy',()=>{
 const j=jacobianState([.5,.5],[.1,-.1],[1,-1],.001);close(j.determinant,0);vector(j.jvp,[0,0]);vector(j.actualChange,[-.01,.01]);
});
test('backpropagation reference agrees with independently derived gradients',()=>{
 const b=neuronState(s.input,s.parameters,s.target,'relu');vector([b.z,b.hidden,b.prediction,b.loss,b.adjointHidden,b.adjointZ],[.5,.5,1,.5,2,2]);vector(parameterNames.map(k=>b.gradients[k]),[2,4,2,.5,1]);vector(b.inputGradient,[1,-.5]);
});
test('simultaneous update uses the old gradient for all parameters without mutating it',()=>{
 const before=JSON.stringify(s);const p=neuronStep(s.input,s.parameters,0,'relu',.01);vector(parameterNames.map(k=>p[k]),[.48,-.29,.48,1.995,-.01]);const b=neuronState(s.input,p,0,'relu');close(b.z,.38);close(b.prediction,.7481);close(b.loss,.279826805);assert.equal(JSON.stringify(s),before);
});
test('large correct-gradient step increases the recomputed loss',()=>{
 const p=neuronStep(s.input,s.parameters,0,'relu',2);vector(parameterNames.map(k=>p[k]),[-3.5,-8.25,-3.5,1,-2]);const b=neuronState(s.input,p,0,'relu');close(b.z,-23.5);close(b.loss,2);
});
test('inactive ReLU leaves the output-bias gradient alive',()=>{
 const b=neuronState(s.input,{...s.parameters,b:-1},1,'relu');vector(parameterNames.map(k=>b.gradients[k]),[0,0,0,0,-1]);close(b.loss,.5);assert.equal(b.kink,false);
});
for(const activation of ['relu','tanh'] as const)test(`all parameter gradients pass central differences on a smooth ${activation} interval`,()=>{
 const check=neuronGradientCheck(s.input,s.parameters,0,activation,.00001);assert.equal(check.smooth,true);assert.ok(check.maxError<1e-8);
});
test('ReLU kink uses an explicit convention and does not fabricate numerical agreement',()=>{
 const p={...s.parameters,b:0};const b=neuronState(s.input,p,1,'relu');assert.equal(b.kink,true);close(b.activationSlope,0);close(b.gradients.w1,0);const check=neuronGradientCheck(s.input,p,1,'relu',.00001);assert.equal(check.smooth,false);close(check.rows[0].numeric,-.99999,1e-7);assert.equal(check.rows[3].smooth,true);assert.equal(check.rows[4].smooth,true);
});
test('a difference interval crossing the gate is marked even away from the kink',()=>{
 const p={...s.parameters,b:.0001};const check=neuronGradientCheck(s.input,p,1,'relu',.001);assert.equal(check.smooth,false);assert.equal(check.rows[0].smooth,false);
});
test('out-of-control-range updates are detectable without clipping the candidate',()=>{
 const p={w1:10,w2:10,b:10,v:10,c:10};const next=neuronStep([1,2],p,0,'relu',2);assert.ok(Math.abs(next.w1)>100);assert.ok(Number.isFinite(neuronState([1,2],next,0,'relu').loss));assert.equal(validCalculus({...s,parameters:next}),false);
});
test('all calculus presets and guided examples use finite valid persisted settings',()=>{
 for(const kind of Object.keys(calculusPresets) as CalculusKind[]){for(const patch of calculusPresets[kind]){const state={...s,...patch};assert.ok(validCalculus(state));const values=calculusValues(state);assert.ok(Number.isFinite(values.neuron.loss));}learningGuides[kind].experiments.forEach(g=>assert.ok(calculusPresets[kind][g.preset]));}
 assert.equal(validCalculus({...s,stage:4}),false);assert.equal(validCalculus({...s,updates:51}),false);assert.equal(validCalculus({...s,point:[1]}),false);assert.equal(validCalculus({...s,parameters:null}),false);assert.equal(validCalculus({...s,epsilon:0}),false);
});
test('nonfinite model inputs and invalid difference intervals are rejected',()=>{
 assert.throws(()=>chainState(1,2,1,.1,0));assert.throws(()=>jacobianState([NaN,0],[1,0],[1,0],.001));assert.throws(()=>neuronState([1,2],{...s.parameters,v:Infinity},0,'relu'));assert.throws(()=>neuronStep(s.input,s.parameters,0,'relu',-1));assert.throws(()=>neuronGradientCheck(s.input,s.parameters,0,'relu',0));
});
