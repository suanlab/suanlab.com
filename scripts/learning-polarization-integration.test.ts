import test from'node:test';import assert from'node:assert/strict';import{createHash}from'node:crypto';
import*as data from'../src/data/interactive-book';import prior from'./fixtures/pre-polarization-registry-hashes.json';import{evaluateWaves,wavesKinds}from'../src/lib/learning-waves';
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex'),close=(a:number,b:number,tol=3e-10)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
const[j,s]=[wavesKinds[10],wavesKinds[11]],J=(i:number)=>evaluateWaves(j,data.wavesLabs[j].presets[i].values).jones!,S=(i:number)=>evaluateWaves(s,data.wavesLabs[s].presets[i].values).stokes!;
test('all452 prior units,guides,lab entries and relative order remain identical as canonical data',()=>{
 const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.equal(units.length,452);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(k=>[k,(data as Record<string,any>)[name][k]]))]));assert.equal(hash(labs),prior.labs);
});
test('the quarter-plate preset has a true circular real trajectory and no invented principal-axis orientation',()=>{
 const a=J(0),q=a.query;close(q.output.intensity,1);close(q.meanRealSquare,.5);close(q.output.majorAxis,Math.SQRT1_2);close(q.output.minorAxis,Math.SQRT1_2);assert.equal(q.output.orientation,null);close(q.output.stokes[3],1);close(q.observed.output[0],0);close(q.observed.output[1],Math.SQRT1_2);q.trajectory.forEach(p=>close(p.output[0]**2+p.output[1]**2,.5));close(a.commonRadius,1.15);
});
test('guide presets preserve noncommuting extinction,transmission and global-phase-only clock changes',()=>{
 const a=J(2).query,b=J(3).query;assert.equal(a.output.intensity,0);assert.equal(a.output.degree,null);close(b.output.intensity,1);const c=J(0).query,d=J(4).query;c.output.stokes.forEach((v,i)=>close(v,d.output.stokes[i]));c.analyzerScan.forEach((p,i)=>close(p.intensity,d.analyzerScan[i].intensity));assert.ok(Math.hypot(c.observed.output[0]-d.observed.output[0],c.observed.output[1]-d.observed.output[1])>.5);assert.equal(J(6).query.transmittance,null);assert.equal(J(7).query.output.intensity,0);
});
test('the worked ellipse preserves its exact π/3 phase and actual horizontal transmission',()=>{
 const a=J(8),b=J(9);assert.equal(a.system.relativePhase,Math.PI/3);close(a.query.output.intensity,1);close(a.query.output.stokes[1],.5);close(a.query.output.stokes[2],Math.sqrt(3)/4);close(a.query.output.stokes[3],.75);close(b.query.output.intensity,.75);close(b.query.meanRealSquare,.375);close(J(10).query.output.orientation!,Math.PI/4);
});
test('unpolarized and pure circular presets share complete linear scans but differ in actual circular-sensitive observations',()=>{
 const a=S(1).query,b=S(2).query;a.analyzerScan.forEach((p,i)=>{close(p.intensity,.5);close(p.intensity,b.analyzerScan[i].intensity);});close(a.output.degree!,0);close(b.output.degree!,1);close(a.measurements.values[4],.5);close(b.measurements.values[4],1);close(b.output.stokes[3],1);close(S(3).query.output.stokes[3],-1);close(a.output.normalizedStokes![0],b.output.normalizedStokes![0]);close(a.output.normalizedStokes![1],b.output.normalizedStokes![1]);
});
test('the3:1mixed-state experiment retains half degree while rotating preference into the negative circular channel',()=>{
 const a=S(0).query,b=S(9).query;close(a.input.stokes[1],.5);close(a.output.stokes[3],-.5);close(a.output.degree!,.5);close(b.output.degree!,.5);close(a.measurements.values[4],.25);close(a.measurements.values[5],.75);a.measurements.reconstructed.forEach((v,i)=>close(v,a.output.stokes[i]));const c=S(6).query;close(c.output.intensity,1);close(c.output.degree!,1);const mean=c.population.reduce((sum,r)=>sum+r.probability*r.field[0][0],0);assert.equal(mean,0);
});
test('depolarization before filtering separates lower transmitted power from higher conditional degree',()=>{
 const a=S(4).query,b=S(5).query;close(a.output.intensity,1);close(a.output.degree!,0);close(b.depolarized.degree!,0);close(b.output.intensity,.5);close(b.output.degree!,1);close(b.transmittance!,.5);for(const p of S(5).depolarizationScan)if(p.outputDegree!==null)close(p.outputDegree,1);
});
test('displayed representatives have positive probability and nonzero fields,while zero intensity invents no ellipse or normalized point',()=>{
 const a=S(3);assert.ok(a.representativeIndex>=4);assert.ok(a.representative!.probability>0);assert.equal(a.representativeTrajectory.length,129);const b=S(8);assert.equal(b.representative,null);assert.equal(b.representativePoint,null);assert.equal(b.representativeTrajectory.length,0);assert.equal(b.query.output.normalizedStokes,null);assert.equal(b.query.transmittance,null);const c=S(7).query,d=S(1).query;c.output.stokes.forEach((v,i)=>close(v,d.output.stokes[i]));
});
test('physical power changes retain their scale instead of independently normalizing plots or stages',()=>{
 const v=data.wavesLabs[s].presets[0].values,a=S(0),b=evaluateWaves(s,[4,...v.slice(1)]).stokes!;close(b.query.output.intensity,4*a.query.output.intensity);a.query.measurements.values.forEach((x,i)=>close(b.query.measurements.values[i],4*x));close(b.query.output.degree!,a.query.output.degree!);close(b.representativeRadius,2*a.representativeRadius);close(J(2).commonRadius,J(3).commonRadius);
});
test('continuous amplitudes and clock phases persist exactly without changing full trajectories or population measurements',()=>{
 const a=data.wavesLabs[j].presets[0].values,b=evaluateWaves(j,[.73123456789,...a.slice(1)]).jones!;assert.equal(b.system.amplitudeX,.73123456789);const c=data.wavesLabs[s].presets[0].values,d=evaluateWaves(s,[...c.slice(0,-1),.73123456789]).stokes!,e=S(0);assert.equal(d.observationPhase,.73123456789);assert.deepEqual(d.query,e.query);assert.notDeepEqual(d.representativePoint,e.representativePoint);for(const id of[j,s]){const v=data.wavesLabs[id].controls.map(c=>c.initial);assert.throws(()=>evaluateWaves(id,new Array(v.length)));assert.throws(()=>evaluateWaves(id,v.map((x,i)=>i===7?.5:x)));assert.throws(()=>evaluateWaves(id,v.map((x,i)=>i===0?NaN:x)));}
});
