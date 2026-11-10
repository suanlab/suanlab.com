import test from 'node:test';
import assert from 'node:assert/strict';
import{continuumMultiphysicsKinds,thermoelasticLab,thermoelectricLab,thermoelasticBounds,thermoelectricBounds}from '../src/lib/learning-continuum-multiphysics-lab';
import{thermoelasticAt,thermoelasticProfile,type MultiphysicsObservation,type ThermoelasticCoupling}from '../src/lib/learning-continuum-multiphysics';
import{evaluateContinuum,isContinuumKind}from '../src/lib/learning-continuum';
import{continuumLabs,learningUnits,learningGuides}from '../src/data/interactive-book';
import{curriculumDomains}from '../src/data/learning-curriculum';
import refs from './fixtures/continuum-multiphysics-independent.json';
const near=(a:number|null,b:number,tol=2e-8)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.max(Math.abs(b),1e-290),`${a} != ${b}`);};
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
function bilingual(v:unknown):void{if(!v||typeof v!=='object')return;const r=v as Record<string,unknown>;if('ko'in r&&'en'in r){assert.ok(typeof r.ko==='string'&&r.ko.length>0);assert.ok(typeof r.en==='string'&&r.en.length>0);assert.doesNotMatch(r.en as string,/[가-힣]/);}Object.values(r).forEach(bilingual);}
for(const[index,id]of continuumMultiphysicsKinds.entries()){
 const spec=continuumLabs[id];
 test(`${id}: registration connects bilingual concepts, experiments, quizzes and actual prerequisites`,()=>{
  const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit&&isContinuumKind(id));assert.ok(curriculumDomains.some(d=>d.unitIds.includes(id)));
  unit.prerequisiteIds.forEach(id=>assert.ok(learningUnits.some(u=>u.id===id)));assert.equal(unit.questions.length,2);
  assert.equal(learningGuides[id].steps.length,4);assert.equal(learningGuides[id].experiments.length,3);learningGuides[id].experiments.forEach(e=>assert.ok(spec.presets[e.preset]));
  assert.equal(spec.presets.length,20);assert.equal(spec.controls.length,index===0?12:10);assert.deepEqual(spec.controls.map(c=>[c.min,c.max]),index===0?thermoelasticBounds:thermoelectricBounds);assert.ok(spec.controls.every(c=>c.step==='any'));bilingual({unit,guide:learningGuides[id],spec});
 });
 test(`${id}: every preset preserves caller data and finite or explicitly missing observations`,()=>{
  for(const p of spec.presets){const v=[...p.values],before=[...v],r=evaluateContinuum(id,v);assert.deepEqual(v,before);finiteTree(r);assert.ok(r.thermoelastic||r.thermoelectric);}
 });
 test(`${id}: sparse, nonfinite, wrong-length and out-of-range controls are rejected`,()=>{
  const v=spec.controls.map(c=>c.initial);assert.throws(()=>evaluateContinuum(id,v.slice(1)),RangeError);
  for(let i=0;i<v.length;i++){for(const bad of[NaN,Infinity,spec.controls[i].min-1,spec.controls[i].max+1]){const a=[...v];a[i]=bad;assert.throws(()=>evaluateContinuum(id,a),RangeError);}const a=[...v];delete a[i];assert.throws(()=>evaluateContinuum(id,a),RangeError);}
 });
}
test('shared spatial evaluation retains independently differentiated physical-variable references and point temperature rates',()=>{
 for(const row of refs.thermalCases){const profile=thermoelasticProfile(row.parameters,row.time,[0,row.phase,2*Math.PI],row.mode as ThermoelasticCoupling),q=profile[1];
  for(const key of['displacement','velocity','strain','temperatureDeviation','stress','heatFlux']as const)near(q[key].value,Number(row.expected[key].value));
  near(q.temperatureRate.value,Number(row.expected.temperatureRateAmplitude.value)*Math.cos(row.phase));assert.deepEqual(q,thermoelasticAt(row.parameters,row.time,row.phase,row.mode as ThermoelasticCoupling));
 }
});
test('eight independent fixed-material references verify the actual teaching adapter, both models and time budgets',()=>{
 assert.equal(refs.teachingCases.length,8);
 for(const row of refs.teachingCases){const p=row.parameters,r=thermoelasticLab([p.elasticModulus,p.coupling,p.conductivity,p.wavenumber,p.area,p.initialDisplacement,p.initialVelocity,p.initialTemperatureDeviation,row.time,row.phase,0,row.time]),q=row.mode==='reciprocal'?r.query:r.oneWay,w=row.mode==='reciprocal'?r.window!:r.oneWayWindow!;
  for(const key of['displacementAmplitude','velocityAmplitude','temperatureAmplitude','totalAvailability','mechanicalExchangeRate','conductionAvailabilityLoss']as const)near(q[key].value,Number(row.expected[key].value));
  for(const key of['mechanicalExchange','conductionAvailabilityLoss','balanceDerivedAvailabilityChange']as const){if(row.mode==='reciprocal'&&p.conductivity===0&&key==='balanceDerivedAvailabilityChange')assert.equal(w[key].value,0);else near(w[key].value,Number(row.window![key].value));}
 }
});
test('24 independent electric references retain geometry, potential, local work and actual terminal ports',()=>{
 for(const row of refs.electricCases){const p=row.parameters,r=thermoelectricLab([p.length,p.area,p.resistivity,p.conductivity,p.seebeck,p.leftTemperature,p.rightTemperature,p.currentDensity,p.referencePotential,row.xi]);for(const key of['temperature','potential','electricField','totalHeatFlux','localElectricalWork','leftHeatInput','rightHeatOutput','electricalPowerInput','partialElectricalPower']as const)near(r.query[key].value,Number(row.expected[key].value));}
});
const tv=continuumLabs[continuumMultiphysicsKinds[0]].controls.map(c=>c.initial),ev=continuumLabs[continuumMultiphysicsKinds[1]].controls.map(c=>c.initial);
const change=(v:number[],i:number,value:number)=>{const a=[...v];a[i]=value;return a;};
test('all12 thermal controls affect declared material, geometry, initial data, observation or time integral',()=>{
 const b=thermoelasticLab(tv);
 near(thermoelasticLab(change(tv,0,4)).query.stress.value,4*.001*Math.cos(.7)-.02*.02*Math.cos(.7));
 assert.equal(thermoelasticLab(change(tv,1,0)).query.mechanicalExchangeRate.sign,0);
 assert.equal(thermoelasticLab(change(tv,2,0)).query.conductionAvailabilityLoss.sign,0);
 near(thermoelasticLab(change(tv,3,2)).query.cellLength,Math.PI);
 near(thermoelasticLab(change(tv,4,2)).query.totalAvailability.value,2*b.query.totalAvailability.value!);
 assert.equal(thermoelasticLab(change(tv,5,0)).query.displacementAmplitude.sign,0);
 assert.equal(thermoelasticLab(change(tv,6,0)).query.mechanicalExchangeRate.sign,0);
 assert.equal(thermoelasticLab(change(tv,7,0)).query.thermalAvailability.sign,0);
 assert.notEqual(thermoelasticLab(change(tv,8,1)).query.temperatureAmplitude.value,b.query.temperatureAmplitude.value);
 near(thermoelasticLab(change(tv,9,0)).query.temperatureDeviation.value,.02);
 assert.equal(thermoelasticLab(change(tv,10,1)).window!.conductionAvailabilityLoss.sign,0);
 assert.ok(thermoelasticLab(change(tv,11,2)).window!.conductionAvailabilityLoss.value!>b.window!.conductionAvailabilityLoss.value!);
});
test('all10 electrical controls change actual geometry, constitutive response, drive or gauge observation',()=>{
 const b=thermoelectricLab(ev);
 near(thermoelectricLab(change(ev,0,2)).query.joulePower.value,2*b.query.joulePower.value!);
 near(thermoelectricLab(change(ev,1,2)).query.current.value,2*b.query.current.value!);
 near(thermoelectricLab(change(ev,2,2)).query.joulePower.value,2*b.query.joulePower.value!);
 near(thermoelectricLab(change(ev,3,2)).query.stableJouleTemperatureRise.value,b.query.stableJouleTemperatureRise.value!/2);
 assert.equal(thermoelectricLab(change(ev,4,0)).query.reversibleElectricalPower.sign,0);
 assert.notEqual(thermoelectricLab(change(ev,5,305)).query.terminalVoltageDrop.value,b.query.terminalVoltageDrop.value);
 assert.equal(thermoelectricLab(change(ev,6,300)).query.reversibleElectricalPower.sign,0);
 near(thermoelectricLab(change(ev,7,1.8)).query.joulePower.value,4*b.query.joulePower.value!);
 const gauge=thermoelectricLab(change(ev,8,3));near(gauge.query.potential.value!-b.query.potential.value!,3);assert.deepEqual(gauge.query.electricalPowerInput,b.query.electricalPowerInput);
 near(thermoelectricLab(change(ev,9,.25)).query.position.value,.25);
});
test('tiny thermal times and invalid time windows keep their different physical meanings',()=>{
 const spec=continuumLabs[continuumMultiphysicsKinds[0]],tiny=thermoelasticLab(spec.presets[9].values),reversed=thermoelasticLab(spec.presets[14].values);
 assert.equal(tiny.query.temperatureAmplitude.sign,-1);assert.equal(tiny.window!.conductionAvailabilityLoss.sign,1);assert.equal(tiny.window!.conductionAvailabilityLoss.value,null);assert.ok(tiny.window!.conductionAvailabilityLoss.logMagnitude!<-2000);
 assert.equal(reversed.window,null);assert.equal(reversed.oneWayWindow,null);assert.equal(reversed.windowStatus,'reversed-time-order');assert.ok(reversed.query.totalAvailability.value!>0);
});
test('zero mechanisms and the pure thermal tail keep exact zeros and nonzero decay distinct',()=>{
 const a=thermoelasticLab(continuumLabs[continuumMultiphysicsKinds[0]].presets[4].values);assert.deepEqual(a.query.temperatureAmplitude,a.oneWay.temperatureAmplitude);assert.equal(a.query.mechanicalExchangeRate.sign,0);
 const tail=thermoelasticLab(continuumLabs[continuumMultiphysicsKinds[0]].presets[12].values);near(tail.query.temperatureAmplitude.value,.02*Math.exp(-16));
 const z=thermoelasticLab(continuumLabs[continuumMultiphysicsKinds[0]].presets[7].values);assert.equal(z.query.totalAvailability.sign,0);
});
test('both adapters retain exact spatial/time endpoints and the actual precision query',()=>{
 const p=thermoelasticLab(change(tv,9,Number.MIN_VALUE));assert.equal(p.spatial[0].phase,0);assert.equal(p.spatial.at(-1)!.phase,2*Math.PI);assert.ok(p.spatial.some(s=>s.phase===Number.MIN_VALUE));assert.equal(p.temporal[0].time,0);assert.equal(p.temporal.at(-1)!.time,4);
 const e=thermoelectricLab(change(ev,9,Number.MIN_VALUE));assert.equal(e.spatial[0].xi,0);assert.equal(e.spatial.at(-1)!.xi,1);assert.ok(e.spatial.some(s=>s.xi===Number.MIN_VALUE));
});
test('cooling, generation, open circuit and unavailable finite COP remain distinct in actual presets',()=>{
 const spec=continuumLabs[continuumMultiphysicsKinds[1]],cool=thermoelectricLab(spec.presets[1].values),generation=thermoelectricLab(spec.presets[3].values),open=thermoelectricLab(spec.presets[4].values),tiny=thermoelectricLab(spec.presets[10].values);
 near(cool.query.leftCoolingCOP!.value,24.75);assert.equal(generation.query.electricalRegime,'generation');assert.equal(generation.query.leftCoolingCOP,null);
 assert.equal(open.query.current.sign,0);assert.equal(open.query.electricalPowerInput.sign,0);assert.notEqual(open.query.totalHeatFlux.sign,0);near(open.query.terminalVoltageDrop.value,.1);
 assert.equal(tiny.query.leftCoolingStatus,'defined');assert.equal(tiny.query.leftCoolingCOP!.sign,1);assert.equal(tiny.query.leftCoolingCOP!.value,null);assert.ok(tiny.query.leftCoolingCOP!.logMagnitude!>740);
});
test('shared spatial evaluation preserves inputs and nonzero point observations without converting underflow to zero',()=>{
 const p={elasticModulus:2,density:1,heatCapacity:1,referenceTemperature:300,coupling:.02,conductivity:.1,wavenumber:1,area:1,initialDisplacement:0,initialVelocity:0,initialTemperatureDeviation:Number.MIN_VALUE};
 const phases=[0,.7,2*Math.PI],before=[...phases],r=thermoelasticProfile(p,1,phases);assert.deepEqual(phases,before);assert.equal(r[0].temperatureRate.sign,-1);assert.equal(r[0].temperatureRate.value,null);
 assert.throws(()=>thermoelasticProfile(p,1,[NaN]),RangeError);const sparse=[0,.7];delete sparse[0];assert.throws(()=>thermoelasticProfile(p,1,sparse),RangeError);
 const zero:MultiphysicsObservation={value:0,sign:0,logMagnitude:null};assert.notDeepEqual(r[0].temperatureRate,zero);
});
