import test from'node:test';import assert from'node:assert/strict';
import refs from'./fixtures/polarization-independent.json';
import{jonesPolarizationExperiment as jones,stokesPolarizationExperiment as stokes,polarizationCoherencyState as state,coherencyFromStokes,polarizationRetarder,polarizationMueller,type JonesPolarizationExperiment,type StokesPolarizationExperiment}from'../src/lib/learning-polarization';
import{jonesPolarizationObservation,stokesPolarizationObservation}from'../src/lib/learning-polarization';
const close=(a:number,b:number,tol=3e-11)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=3e-11)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
const s:JonesPolarizationExperiment={amplitudeX:1,amplitudeY:0,relativePhase:0,globalPhase:0,retarderAngle:0,retardance:0,analyzerAngle:0,order:'retarder-only',observationPhase:0};
const ensemble:StokesPolarizationExperiment={population:[{amplitudeX:1,amplitudeY:0,relativePhase:0,globalPhase:0,weight:1}],retainedPolarization:1,retarderAngle:0,retardance:0,analyzerAngle:0,order:'retarder-only'};
test('efficient observations share all validated physical states with the complete trajectory and scan models',()=>{
 for(const r of refs.jones.slice(0,12)){const full=jones(r.system as JonesPolarizationExperiment),fast=jonesPolarizationObservation(r.system as JonesPolarizationExperiment);for(const key of['input','intermediate','output','observed','transmittance','meanRealSquare']as const)assert.deepEqual(fast[key],full[key]);}
 for(const r of refs.ensembles.slice(0,12)){const full=stokes(r.system as StokesPolarizationExperiment),fast=stokesPolarizationObservation(r.system as StokesPolarizationExperiment);for(const key of['input','depolarized','output','measurements','mueller','outputPopulation']as const)assert.deepEqual(fast[key],full[key]);}
});
test('48 independent vector-projection systems,SVD ellipses and816real-field observations agree with the Jones model',()=>{
 assert.equal(refs.jones.length,48);assert.equal(refs.provenance.productImports,false);assert.equal(refs.provenance.symbolicMalus,'0');
 for(const r of refs.jones){const a=jones(r.system as JonesPolarizationExperiment);for(const key of['input','intermediate','output']as const)a[key].jones.forEach((v,i)=>vector(v,r[key][i]));vector(a.output.stokes,r.outputStokes);close(a.output.majorAxis,r.axes[0]);close(a.output.minorAxis,r.axes[1]);close(a.output.intensity,r.cycleIntensity);close(a.meanRealSquare,r.cycleIntensity/2);for(const p of r.trajectory){const i=Math.round(p.phase/(2*Math.PI)*128);for(const k of['input','intermediate','output']as const)vector(a.trajectory[i][k],p[k]);}}
});
test('48 independent ensembles,288real analyzer integrals,eigenvalue degrees and six-state Mueller reconstructions agree',()=>{
 assert.equal(refs.ensembles.length,48);
 for(const r of refs.ensembles){const a=stokes(r.system as StokesPolarizationExperiment);for(const key of['input','depolarized','output']as const){close(a[key].coherency.xx,r[key].xx);close(a[key].coherency.yy,r[key].yy);vector(a[key].coherency.xy,r[key].xy);}vector(a.output.stokes,r.stokes);if(r.degree===null)assert.equal(a.output.degree,null);else close(a.output.degree!,r.degree);vector(a.measurements.values,r.measurements);vector(a.measurements.reconstructed,r.stokes);a.mueller.forEach((row,i)=>vector(row,r.mueller[i]));vector(a.muellerStokes,r.stokes);}
});
test('quarter plate produces the declared circular handedness and real field rotation,not a second displacement phasor',()=>{
 const a=jones({...s,amplitudeX:Math.SQRT1_2,amplitudeY:Math.SQRT1_2,retardance:Math.PI/2,observationPhase:Math.PI/2});close(a.output.stokes[3],1);close(a.output.degree!,1);assert.equal(a.output.orientation,null);vector(a.observed.output,[0,Math.SQRT1_2]);close(a.output.majorAxis,Math.SQRT1_2);close(a.output.minorAxis,Math.SQRT1_2);close(a.output.ellipticityRatio!,1);
 const b=jones({...s,amplitudeX:Math.SQRT1_2,amplitudeY:Math.SQRT1_2,relativePhase:-Math.PI/2,observationPhase:Math.PI/2});close(b.output.stokes[3],-1);close(b.observed.output[1],-Math.SQRT1_2);
});
test('a half plate rotates linear input by twice the axis angle while preserving normalized intensity',()=>{
 const angle=.31,a=jones({...s,retarderAngle:angle,retardance:Math.PI});close(a.output.intensity,1);close(a.output.orientation!,2*angle);close(a.output.minorAxis,0);close(a.output.majorAxis,1);
});
test('the worked elliptical state retains axis geometry and its physical analyzer transmission',()=>{
 const a=jones({...s,amplitudeX:Math.sqrt(3)/2,amplitudeY:.5,relativePhase:Math.PI/3});vector(a.output.stokes,[1,.5,Math.sqrt(3)/4,.75]);close(a.output.majorAxis**2+a.output.minorAxis**2,1);close(2*a.output.majorAxis*a.output.minorAxis,.75);const b=jones({...s,amplitudeX:Math.sqrt(3)/2,amplitudeY:.5,relativePhase:Math.PI/3,order:'analyzer-only'});close(b.output.intensity,.75);close(b.meanRealSquare,.375);
});
test('retarder and analyzer order is physically noncommutative,including exact blocked and transmitted cases',()=>{
 const base={...s,retarderAngle:Math.PI/4,retardance:Math.PI,analyzerAngle:0},a=jones({...base,order:'retarder-then-analyzer'}),b=jones({...base,order:'analyzer-then-retarder'});assert.equal(a.output.intensity,0);assert.equal(a.output.degree,null);assert.equal(a.transmittance,0);close(b.output.intensity,1);close(b.output.orientation!,Math.PI/2);
});
test('Malus law applies to field projection and the independent actual real-square average carries a half factor',()=>{
 const alpha=.37,beta=.9,a=jones({...s,amplitudeX:Math.cos(alpha),amplitudeY:Math.sin(alpha),analyzerAngle:beta,order:'analyzer-only'});close(a.output.intensity,Math.cos(alpha-beta)**2);close(a.meanRealSquare,a.output.intensity/2);close(a.transmittance!,a.output.intensity);
});
test('global phase changes the observed real vector but preserves all polarimetric observations and ellipse axes',()=>{
 const base={...s,amplitudeX:.8,amplitudeY:.5,relativePhase:.7,retarderAngle:.2,retardance:.9},a=jones(base),b=jones({...base,globalPhase:1.2});vector(a.output.stokes,b.output.stokes);close(a.output.majorAxis,b.output.majorAxis);close(a.output.minorAxis,b.output.minorAxis);a.analyzerScan.forEach((p,i)=>close(p.intensity,b.analyzerScan[i].intensity));assert.ok(Math.hypot(a.observed.output[0]-b.observed.output[0],a.observed.output[1]-b.observed.output[1])>.1);
});
test('zero Jones fields and absent components preserve unavailable observables',()=>{
 const a=jones({...s,amplitudeX:0});assert.equal(a.output.intensity,0);assert.equal(a.output.degree,null);assert.equal(a.output.relativePhase,null);assert.equal(a.output.orientation,null);assert.equal(a.output.ellipticityRatio,null);assert.equal(a.transmittance,null);assert.equal(jones(s).output.relativePhase,null);
});
test('exact quadrantal analyzer blocking does not snap neighboring angles into darkness',()=>{
 const a=jones({...s,analyzerAngle:Math.PI/2,order:'analyzer-only'}),b=jones({...s,analyzerAngle:Math.PI/2+1e-9,order:'analyzer-only'});assert.equal(a.output.intensity,0);assert.ok(b.output.intensity>0);close(b.output.intensity,1e-18,1e-20);
});
test('valid raw Stokes cone data round-trip with the specified circular sign,while invalid data are rejected rather than projected',()=>{
 for(const v of[[1,1,0,0],[1,0,0,1],[1,0,0,-1],[2,.3,-.4,.7],[0,0,0,0]]as[number,number,number,number][])vector(state(coherencyFromStokes(v)).stokes,v);
 for(const v of[[-1,0,0,0],[1,1+Number.EPSILON*2,0,0],[0,0,0,1],[1,NaN,0,0]]as[number,number,number,number][])assert.throws(()=>coherencyFromStokes(v));
 assert.throws(()=>state({xx:1,yy:1,xy:[2,0]}));assert.throws(()=>state({xx:0,yy:1,xy:[1e-30,0]}));assert.throws(()=>state({xx:-1,yy:1,xy:[0,0]}));assert.throws(()=>state({xx:128,yy:128,xy:[0,0]}));
});
test('physical finite populations average observations,not a single averaged Jones vector',()=>{
 const a=stokes({...ensemble,population:[{...ensemble.population[0],globalPhase:0},{...ensemble.population[0],globalPhase:Math.PI}]});close(a.input.intensity,1);close(a.input.degree!,1);vector(a.input.stokes,[1,1,0,0]);const mean=a.population.reduce((v,r)=>v+r.probability*r.field[0][0],0);assert.equal(mean,0);assert.ok(a.output.intensity>0);
});
test('retarders preserve intensity and degree for partially polarized input',()=>{
 const p=[ensemble.population[0],{...ensemble.population[0],amplitudeX:0,amplitudeY:1,weight:.4}];const a=stokes({...ensemble,population:p,retarderAngle:.7,retardance:1.2});close(a.input.intensity,a.output.intensity);close(a.input.degree!,a.output.degree!);
});
test('an ideal analyzer can increase conditional polarization degree while reducing total transmitted intensity',()=>{
 const a=stokes({...ensemble,retainedPolarization:0,order:'analyzer-only'});close(a.input.intensity,1);close(a.depolarized.degree!,0);close(a.output.intensity,.5);close(a.output.degree!,1);close(a.transmittance!,.5);
});
test('the worked3:1mixture rotates its surviving linear preference into the circular measurement',()=>{
 const a=stokes({...ensemble,population:[{...ensemble.population[0],weight:3},{...ensemble.population[0],amplitudeX:0,amplitudeY:1,weight:1}],retarderAngle:Math.PI/4,retardance:Math.PI/2});vector(a.input.stokes,[1,.5,0,0]);vector(a.output.stokes,[1,0,0,-.5]);close(a.output.degree!,.5);vector(a.measurements.values,[.5,.5,.5,.5,.25,.75]);
});
test('isotropic depolarization is an explicit nonnegative Pauli mixture and scales the degree before deterministic optics',()=>{
 for(const d of[0,.2,.8,1]){const a=stokes({...ensemble,retainedPolarization:d});close(a.channelWeights.reduce((sum,v)=>sum+v,0),1);assert.ok(a.channelWeights.every(v=>v>=0));close(a.input.intensity,a.depolarized.intensity);close(a.depolarized.degree!,d*a.input.degree!);assert.equal(a.depolarizationPlacement,'before-the-deterministic-optical-elements');}
});
test('complete linear analyzer scans cannot distinguish unpolarized light from a pure circular state',()=>{
 const a=stokes({...ensemble,retainedPolarization:0}),b=stokes({...ensemble,population:[{amplitudeX:Math.SQRT1_2,amplitudeY:Math.SQRT1_2,relativePhase:Math.PI/2,globalPhase:0,weight:1}]});a.analyzerScan.forEach((p,i)=>{close(p.intensity,.5);close(p.intensity,b.analyzerScan[i].intensity);});close(a.output.degree!,0);close(b.output.degree!,1);close(a.output.stokes[3],0);close(b.output.stokes[3],1);assert.ok(Math.abs(a.measurements.values[4]-b.measurements.values[4])>.49);
});
test('six positive analyzer measurements reconstruct all Stokes components,including the circular component',()=>{
 const a=stokes({...ensemble,retainedPolarization:.8,retarderAngle:.3,retardance:1.1});assert.ok(a.measurements.values.every(v=>v>=0));vector(a.measurements.reconstructed,a.output.stokes);close(a.measurements.values[0]+a.measurements.values[1],a.output.intensity);close(a.measurements.values[2]+a.measurements.values[3],a.output.intensity);close(a.measurements.values[4]+a.measurements.values[5],a.output.intensity);
});
test('Jones to Mueller identity and a physical retarder transform the same Stokes data',()=>{
 const id=polarizationMueller(polarizationRetarder(.7,0));id.forEach((r,i)=>vector(r,[0,1,2,3].map(j=>i===j?1:0)));const a=stokes({...ensemble,retarderAngle:.42,retardance:1.3,retainedPolarization:.7});vector(a.muellerStokes,a.output.stokes);close(a.outputViaMatrix.xx,a.output.coherency.xx);close(a.outputViaMatrix.yy,a.output.coherency.yy);vector(a.outputViaMatrix.xy,a.output.coherency.xy);
});
test('a cone-preserving transpose map still fails the stronger random-Jones physicality criterion',()=>{
 vector(refs.transposeChoiEigenvalues,[-1,1,1,1]);assert.ok(refs.transposeChoiEigenvalues[0]<0);const a=coherencyFromStokes([1,.2,.3,.4]),b={...a,xy:[a.xy[0],-a.xy[1]]as[number,number]};vector(state(b).stokes,[1,.2,.3,-.4]);close(state(a).degree!,state(b).degree!);
});
test('population normalization and zero weights leave physical predictions unchanged',()=>{
 const p=[ensemble.population[0],{...ensemble.population[0],amplitudeX:.3,amplitudeY:.8,relativePhase:.4,weight:3}];const a=stokes({...ensemble,population:p}),b=stokes({...ensemble,population:p.map(r=>({...r,weight:4*r.weight})).concat({...p[0],weight:0})});vector(a.output.stokes,b.output.stokes);vector(a.measurements.values,b.measurements.values);
});
test('zero-intensity ensembles have no normalized Stokes state or transmitted fraction',()=>{
 const a=stokes({...ensemble,population:[{...ensemble.population[0],amplitudeX:0}]});assert.equal(a.output.intensity,0);assert.equal(a.output.degree,null);assert.equal(a.output.normalizedStokes,null);assert.equal(a.transmittance,null);assert.ok(a.measurements.values.every(v=>v===0));
});
test('malformed optics,populations and physical parameter domains are rejected without mutating inputs',()=>{
 const before=JSON.stringify(ensemble);stokes(ensemble);assert.equal(JSON.stringify(ensemble),before);
 for(const patch of[{amplitudeX:-1},{relativePhase:NaN},{retardance:7*Math.PI},{order:'wrong'},{observationPhase:-1}])assert.throws(()=>jones({...s,...patch}as JonesPolarizationExperiment));
 assert.throws(()=>stokes({...ensemble,population:[]}));assert.throws(()=>stokes({...ensemble,population:new Array(2)}));assert.throws(()=>stokes({...ensemble,population:[{...ensemble.population[0],weight:0}]}));assert.throws(()=>stokes({...ensemble,retainedPolarization:1.01}));assert.throws(()=>coherencyFromStokes(new Array(4)as[number,number,number,number]));
});
for(const id of['jones-polarization-waveplates-analyzers','stokes-coherency-polarization-tomography'])for(const lang of['ko','en']as const)test(`${id}.${lang}: detailed theory,six explained exercises and strict KaTeX render`,async()=>{
 const{readFile}=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');const raw=await readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8'),rendered=await getLearningContent(id);assert.ok(raw.length>6500);assert.equal(rendered.outline[lang].length,16);assert.equal(rendered.outline.ko.length,rendered.outline.en.length);assert.equal(new Set(rendered.outline[lang].map(h=>h.id)).size,16);assert.ok(rendered[lang].includes('katex-display'));assert.ok(!rendered[lang].includes('katex-error'));assert.equal((raw.match(/^### (?:Exercise|문제) [1-6]/gm)||[]).length,6);assert.equal((raw.match(lang==='en'?/\*\*Solution:\*\*/g:/\*\*해설:\*\*/g)||[]).length,6);assert.doesNotMatch(raw,/TODO|TBD|coming soon/i);if(lang==='en')assert.doesNotMatch(raw,/[가-힣]/);
});
