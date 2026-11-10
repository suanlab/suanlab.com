import test from 'node:test';
import assert from 'node:assert/strict';
import{viscoelasticAt as at,viscoelasticWindow as window,viscoelasticCycle as cycle,neoHookeanAt as neo,neoHookeanWork as path,type ConstitutiveValue}from '../src/lib/learning-continuum-constitutive';
import refs from './fixtures/continuum-constitutive-independent.json';
const p={elasticModulus:2,viscosity:1,strainAmplitude:.002,angularFrequency:1,initialElasticStrain:0};
const n={shearModulus:2,logStretch:Math.log(2),rotation:.3,normalAngle:.7,referenceArea:2,referenceLength:3};
const near=(a:ConstitutiveValue,b:number,tol=3e-11)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.max(1,Math.abs(b)),`${a} versus ${b}`);};
function relative(a:ConstitutiveValue,b:number,tol=3e-11){assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.abs(b)+Number.MIN_VALUE,`${a} versus ${b}`);}
function expected(actual:ConstitutiveValue,e:{value:string;sign:number;logMagnitude:string|null}){const v=Number(e.value);if(e.sign===0)assert.equal(actual,0);else if(v===0)assert.equal(actual,null);else relative(actual,v);}
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
test('24 independent convolution fields verify actual stress derivatives, powers, storage and dissipation',()=>{
 assert.equal(refs.viscoCases.length,24);for(const r of refs.viscoCases){const q=at(r.parameters,r.time);near(q.strain,r.expected.strain);near(q.strainRate,r.expected.strainRate);for(const name of['kelvin','maxwell']as const){for(const key of['stress','stressRate','energy','energyRate','dissipation','power']as const)relative(q[name][key],r.expected[name][key]);near(q[name].balanceResidual,0);}}
});
test('24 independent physical-time integrals balance work, energy change and dissipated energy',()=>{
 for(const r of refs.viscoCases){const q=window(r.parameters,r.start,r.end);for(const name of['kelvin','maxwell']as const){for(const key of['work','energyChange','dissipation']as const)relative(q[name][key],r.window[name][key]);near(q[name].balanceResidual,0);assert.ok(q[name].dissipation!>=0);}}
});
test('actual finite differences verify Maxwell stress rate and both stored-energy rates',()=>{
 const t=.7,h=1e-5,q=at(p,t),lo=at(p,t-h),hi=at(p,t+h);near((hi.maxwell.stress!-lo.maxwell.stress!)/(2*h),q.maxwell.stressRate!,1e-10);for(const name of['kelvin','maxwell']as const)near((hi[name].energy!-lo[name].energy!)/(2*h),q[name].energyRate!,1e-12);
});
test('initial memory changes Maxwell history while Kelvin response remains determined by current strain and rate',()=>{
 const a=at(p,0),b=at({...p,initialElasticStrain:.002},0),later=at({...p,initialElasticStrain:.002},1);near(a.maxwell.stress,0);near(b.maxwell.stress,.004);near(a.kelvin.stress,b.kelvin.stress!);near(later.maxwell.stress!-at(p,1).maxwell.stress!,.004*Math.exp(-2));near(later.maxwellElasticStrain!+later.maxwellViscousStrain!,later.strain!);near(later.maxwellActualViscousRate,later.maxwellDashpotRate!);
});
test('steady storage and loss moduli match the worked example and positive cyclic work',()=>{
 const q=cycle(p);near(q.maxwellStorage,.4);near(q.maxwellLoss,.8);near(q.kelvinStorage,2);near(q.kelvinLoss,1);near(q.period,2*Math.PI);relative(q.maxwellSteadyCycleWork,Math.PI*.002**2*.8);relative(q.integratedMaxwellSteadyWork,q.maxwellSteadyCycleWork!);relative(q.integratedKelvinSteadyWork,q.kelvinSteadyCycleWork!);
});
test('actual first-cycle work is not the steady-cycle loss when startup memory remains',()=>{
 const q=cycle(p),first=q.actualFirstCycle!;assert.ok(first);assert.notEqual(first.maxwell.work,q.maxwellSteadyCycleWork);assert.ok(first.maxwell.energyChange!>0);near(first.maxwell.work!-first.maxwell.energyChange!,first.maxwell.dissipation!);assert.equal(q.firstCycleStatus,'available');
});
test('the stationary periodic initial condition removes the transient and restores cycle closure',()=>{
 const initialElasticStrain=.0008,q=cycle({...p,initialElasticStrain}),t=at({...p,initialElasticStrain},1);near(t.maxwellTransientStress,0);relative(q.actualFirstCycle!.maxwell.work,q.maxwellSteadyCycleWork!);near(q.actualFirstCycle!.maxwell.energyChange,0);
});
test('zero frequency has no cycle and releases only pre-existing Maxwell energy without external power',()=>{
 const s={...p,angularFrequency:0,initialElasticStrain:.002},q=at(s,1),w=window(s,0,1),c=cycle(s);assert.equal(q.strain,0);assert.equal(q.strainRate,0);assert.equal(q.maxwell.power,0);assert.ok(q.maxwell.energyRate!<0);near(q.maxwell.energyRate!+q.maxwell.dissipation!,0);near(w.maxwell.energyChange!+w.maxwell.dissipation!,0);assert.equal(c.periodStatus,'no-cycle');assert.equal(c.maxwellSteadyCycleWork,null);assert.equal(c.actualFirstCycle,null);
});
test('negative mechanical power can return stored energy while dissipation stays nonnegative',()=>{
 const q=at({...p,initialElasticStrain:.004},2);assert.ok(q.maxwell.power!<0);assert.ok(q.maxwell.energyRate!<0);assert.ok(q.maxwell.dissipation!>0);near(q.maxwell.power!-q.maxwell.energyRate!,q.maxwell.dissipation!);
});
test('omitting relaxation or dissipation exposes the same actual field terms',()=>{
 const q=at(p,.7),w=window(p,0,1);near(q.omittedRelaxationResidual,-q.maxwell.stress!*q.relaxationRate);near(q.maxwell.omittedDissipationResidual,q.maxwell.dissipation!);near(q.kelvin.omittedDissipationResidual,q.kelvin.dissipation!);near(w.maxwell.omittedDissipationResidual,w.maxwell.dissipation!);
});
test('eight1200-digit viscoelastic limits retain small changes, cubic startup and logarithmic observations',()=>{
 assert.equal(refs.viscoLimits.length,8);for(const r of refs.viscoLimits){const q=at(r.parameters,r.time),w=window(r.parameters,0,r.time);expected(q.maxwell.stress,r.expected.stress);expected(q.stableMaxwellStressChange,r.expected.change);assert.equal(q.stressChangeSign,r.expected.change.sign);expected(q.maxwell.stressRate,r.expected.stressRate);expected(w.maxwell.energyChange,r.expected.energyChange);assert.equal(w.maxwell.energyChangeSign,r.expected.energyChange.sign);if(r.expected.change.logMagnitude!==null)near(q.logStressChangeMagnitude,Number(r.expected.change.logMagnitude));if(r.expected.energyChange.logMagnitude!==null)near(w.maxwell.logEnergyChangeMagnitude,Number(r.expected.energyChange.logMagnitude));}
 const tuned=at({...p,initialElasticStrain:.001},1e-20);assert.equal(tuned.roundedMaxwellStressChange,0);relative(tuned.stableMaxwellStressChange,-2/3*1e-63);assert.ok(tuned.maxwell.stressRate!<0);
});
test('nonzero homogeneous memory can underflow while its log and sign remain physically distinct from zero memory',()=>{
 const q=at({...p,elasticModulus:4,viscosity:.0625,angularFrequency:0,initialElasticStrain:.001},16);assert.equal(q.homogeneousMemory,null);assert.ok(q.logMemoryMagnitude!<-1000);assert.equal(q.maxwell.stress,null);assert.equal(q.maxwell.stressSign,1);assert.equal(at({...p,angularFrequency:0},16).maxwell.stress,0);
});
test('cycle horizon and arithmetic range do not become invented finite periods or zero losses',()=>{
 assert.equal(cycle({...p,angularFrequency:.01}).firstCycleStatus,'outside-teaching-time-horizon');const q=cycle({...p,angularFrequency:Number.MIN_VALUE});assert.equal(q.period,null);assert.equal(q.periodStatus,'outside-arithmetic-range');assert.equal(q.maxwellSteadyCycleWork,null);assert.ok(Number.isFinite(q.logMaxwellCycleWork));
});
test('zero-duration time windows have exact zero integrals while reversed times are rejected',()=>{
 const q=window(p,.5,.5);for(const k of['kelvin','maxwell']as const){assert.equal(q[k].work,0);assert.equal(q[k].energyChange,0);assert.equal(q[k].dissipation,0);}assert.throws(()=>window(p,1,0),RangeError);
});
test('24 full-matrix references verify neo-Hookean energy derivatives, stresses, traction and area-converted force',()=>{
 assert.equal(refs.neoCases.length,24);for(const r of refs.neoCases){const q=neo(r.parameters);for(const k of['stretch','energyDensity','nominalStress','cauchyStress','nominalTangent','pressure','currentArea','force','storedEnergy','normalStress','signedShearStress']as const)near(q[k],r.expected[k]);for(const k of['F','stress','firstPiola']as const)for(let i=0;i<3;i++)for(let j=0;j<3;j++)near(q[k][i][j],r.expected[k][i][j]);for(let i=0;i<3;i++)near(q.traction[i],r.expected.traction[i]);}
});
test('24 independent stretch-coordinate integrals verify reference-volume mechanical work and reject wrong conjugacy',()=>{
 for(const r of refs.neoCases){const q=path(r.parameters,r.start,r.end);near(q.workDensity,r.path.workDensity);near(q.energyDensityChange,r.path.energyDensityChange);near(q.wrongCauchyWorkDensity,r.path.wrongCauchyWorkDensity);near(q.balanceResidual,0);}
});
test('worked stretch2 has energy2, nominal stress3.5, Cauchy stress7 and the same actual force',()=>{
 const q=neo(n);near(q.energyDensity,2);near(q.nominalStress,3.5);near(q.cauchyStress,7);near(q.currentArea,1);near(q.force,7);near(q.storedEnergy,12);near(q.nominalTangent,2.5);near(q.wrongSameAreaForce,14);near(path(n,0,Math.log(2)).work,12);
});
test('rigid rotation changes components covariantly while energy, principal stresses and axial force remain unchanged',()=>{
 const a=neo({...n,rotation:0,normalAngle:0}),b=neo({...n,rotation:.8,normalAngle:.8});near(a.energyDensity,b.energyDensity!);near(a.force,b.force!);near(a.normalStress,b.normalStress!);near(b.signedShearStress,0);assert.notEqual(a.stress[0][0],b.stress[0][0]);near(b.stress[0][1],b.cauchyStress!*Math.cos(.8)*Math.sin(.8));near(Math.hypot(...b.traction.map(v=>v!)),Math.abs(b.cauchyStress!));
});
test('actual finite energy differences verify stress and tangent with the correct reference/current areas',()=>{
 const lambda=1.7,h=1e-5,lo=neo({...n,logStretch:Math.log(lambda-h)}),hi=neo({...n,logStretch:Math.log(lambda+h)}),q=neo({...n,logStretch:Math.log(lambda)});near((hi.energyDensity!-lo.energyDensity!)/(2*h),q.nominalStress!,1e-9);near((hi.nominalStress!-lo.nominalStress!)/(2*h),q.nominalTangent,1e-9);near(q.cauchyStress!*q.currentArea,q.nominalStress!*q.referenceArea);
});
test('eight1200-digit hyperelastic limits preserve signed stress, positive energy and scale-rescued total energy',()=>{
 assert.equal(refs.neoLimits.length,8);for(const r of refs.neoLimits){const q=neo(r.parameters);for(const k of['energyDensity','nominalStress','cauchyStress','force','storedEnergy']as const)expected(q[k],r.expected[k]);if(r.expected.energyDensity.logMagnitude!==null)near(q.logEnergyDensity,Number(r.expected.energyDensity.logMagnitude));}
 const q=neo({...n,logStretch:1e-20});assert.equal(q.roundedStretchChange,0);assert.equal(q.roundedEnergyDensity,0);assert.ok(q.stableStretchChange!>0);assert.ok(q.energyDensity!>0);const rescue=neo(refs.neoLimits[4].parameters);assert.equal(rescue.energyDensity,null);assert.equal(rescue.storedEnergy,Number.MIN_VALUE);
});
test('stable energy differences and physical quadrature retain tiny and symmetric log-stretch paths',()=>{
 for(const t of[1e-20,1e-200,Number.MIN_VALUE]){const q=path(n,0,t),field=neo({...n,logStretch:t});near(q.logEnergyChangeMagnitude,field.logEnergyDensity!);near(q.logWorkDensityMagnitude,field.logEnergyDensity!);}
 const back=path(n,1e-200,0);assert.equal(back.workDensity,null);assert.equal(back.workSign,-1);assert.equal(back.energyChangeSign,-1);const s=path(n,-1e-20,1e-20);assert.notEqual(s.energyDensityChange,0);relative(s.energyDensityChange,2e-60,2e-10);near(s.logWorkDensityMagnitude,s.logEnergyChangeMagnitude!,1e-7);
});
test('forward and reverse elastic paths exchange energy with zero closed-path net work; tangent is only a path property',()=>{
 const a=path(n,-.7,.8),b=path(n,.8,-.7);near(a.workDensity!+b.workDensity!,0);assert.ok(neo({...n,logStretch:-.7}).nominalTangent>0);assert.ok(neo({...n,logStretch:.8}).nominalTangent>0);assert.ok(Math.abs(a.wrongCauchyWorkDensity!-a.workDensity!)>.1);
});
test('supported extremes remain finite or explicitly missing and invalid model domains are rejected',()=>{
 for(const bad of[NaN,Infinity,0,-1])assert.throws(()=>at({...p,elasticModulus:bad},0),RangeError);assert.throws(()=>at({...p,viscosity:0},0),RangeError);assert.throws(()=>at(p,17),RangeError);assert.throws(()=>neo({...n,logStretch:2}),RangeError);assert.throws(()=>neo({...n,shearModulus:0}),RangeError);
 for(const amp of[0,Number.MIN_VALUE,1e-200,.005])for(const omega of[0,Number.MIN_VALUE,4]){finiteTree(at({...p,strainAmplitude:amp,angularFrequency:omega},16));finiteTree(cycle({...p,strainAmplitude:amp,angularFrequency:omega}));}
 for(const ell of[-Math.log(4),-Number.MIN_VALUE,0,Number.MIN_VALUE,Math.log(4)])finiteTree(neo({...n,logStretch:ell}));
});
