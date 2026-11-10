import test from'node:test';import assert from'node:assert/strict';import{createHash}from'node:crypto';
import*as data from'../src/data/interactive-book';import prior from'./fixtures/pre-dispersion-registry-hashes.json';
import{evaluateWaves,wavesKinds}from'../src/lib/learning-waves';
import{gaussianDispersivePacket}from'../src/lib/learning-dispersion';
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
const close=(a:number,b:number,tol=3e-10)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
const[g,l]=[wavesKinds[8],wavesKinds[9]],G=(i:number)=>evaluateWaves(g,data.wavesLabs[g].presets[i].values).gaussian!,L=(i:number)=>evaluateWaves(l,data.wavesLabs[l].presets[i].values).lattice!;
test('all450 previous units,guides,lab entries and their relative order remain identical as canonical data',()=>{
 const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.equal(units.length,450);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(k=>[k,(data as Record<string,any>)[name][k]]))]));assert.equal(hash(labs),prior.labs);
});
test('the Gaussian worked example keeps amplitude width,intensity width,norm and three velocities distinct',()=>{
 const a=G(0).query;close(a.center,2);close(a.amplitudeWidth,Math.SQRT2);close(a.intensityStandardDeviation,1);close(a.intensity,Math.SQRT1_2);close(a.integratedNorm,Math.sqrt(Math.PI));close(a.carrierPhaseVelocity!,2);close(a.groupVelocity,1);close(a.localCrestVelocity!,2.03125);
});
test('opposite dispersion signs give identical complete intensities but distinct complex phases and opposite chirp',()=>{
 const a=G(0),b=G(2);a.field.forEach(p=>close(p.current.intensity,gaussianDispersivePacket(b.system,p.x,b.time).intensity));close(a.query.chirp,-b.query.chirp);assert.ok(Math.hypot(a.query.field[0]-b.query.field[0],a.query.field[1]-b.query.field[1])>.1);
});
test('zero dispersion preserves width while zero amplitude and zero carrier preserve unavailable observables',()=>{
 const a=G(1);for(const p of a.evolution)close(p.amplitudeWidth,1);close(a.query.carrierPhaseVelocity!,2);close(a.query.groupVelocity,1);const b=G(4);assert.equal(b.query.intensityCenter,null);assert.equal(b.query.observedPhase,null);assert.ok(b.field.every(p=>p.current.localWaveNumber===null));assert.equal(G(5).query.carrierPhaseVelocity,null);
});
test('adaptive spatial samples resolve both actual full phases across the widest supported fast-phase preset',()=>{
 for(const i of[0,3,8,9]){const a=G(i);assert.ok(a.display.maximumPhaseStep<=.150000000001);for(let j=1;j<a.field.length;j++){const p=a.field[j],prev=a.field[j-1];assert.ok(Math.abs(p.current.phase-prev.current.phase)<=.15000000001);assert.ok(Math.abs(p.initial.phase-prev.initial.phase)<=.15000000001);}assert.ok(a.display.lo>=-128&&a.display.hi<=128);assert.ok(a.field.length<17000);}
});
test('the lattice numerical example uses actual SI energy,power transport and a distinct displacement norm',()=>{
 const a=L(0).query;close(a.spacing,.25);close(a.mass,.2);close(a.stiffness,6.8);close(a.energy,.06790959833085997);close(a.energyTransportVelocity!,1.2863880942615573);close(a.analyticNorm,.007766439245517835);assert.ok(Math.abs(a.energyTransportVelocity!-a.carrierGroupVelocity)>.05);close(a.energyTransportVelocity!,a.energyWeightedGroupVelocity!);
});
test('narrow and broad presets change physical preparation energy and transport without changing the carrier',()=>{
 const a=L(1).query,b=L(2).query;close(a.carrierGroupVelocity,b.carrierGroupVelocity);close(a.energyTransportVelocity!,a.carrierGroupVelocity,1e-8);assert.ok(Math.abs(b.energyTransportVelocity!-b.carrierGroupVelocity)>.01);assert.ok(Math.abs(a.energy-b.energy)>.001);
});
test('periodic endpoint preparations match and a site translation rotates actual particle observables',()=>{
 const a=L(6).query,b=L(7).query;a.position.forEach((q,i)=>close(q,b.position[i]));const c=L(0).query,d=L(5).query;d.position.forEach((q,i)=>close(q,c.position[(i+29)%32]));close(c.energy,d.energy);const z=L(4).query;assert.equal(z.energyTransportVelocity,null);assert.equal(z.analyticCenter.position,null);assert.equal(z.energyCenter.position,null);
});
test('refining the same material updates mass and stiffness together while actual balance remains physical',()=>{
 const a=L(0).query,b=L(8).query;close(a.c,b.c);close(b.mass,a.mass/2);close(b.stiffness,2*a.stiffness);for(const r of[L(3),L(8),L(10)]){const q=r.query;for(let i=0;i<q.x.length;i++)close(q.assignedEnergyRate[i],q.linkFlux[(i+q.x.length-1)%q.x.length]-q.linkFlux[i]);for(const p of r.evolution){close(p.energy,q.energy);close(p.norm,q.analyticNorm);close(p.transport!,p.weighted!);}}
});
test('continuous physical controls remain exact while sparse arrays and fractional particle selectors are rejected',()=>{
 for(const id of[g,l]){const v=data.wavesLabs[id].controls.map(c=>c.initial);assert.throws(()=>evaluateWaves(id,new Array(v.length)));assert.throws(()=>evaluateWaves(id,v.map((x,i)=>i===1?NaN:x)));}
 const v=data.wavesLabs[g].presets[0].values,r=evaluateWaves(g,[.73123456789,...v.slice(1)]).gaussian!;assert.equal(r.system.amplitude,.73123456789);const w=data.wavesLabs[l].presets[0].values;assert.throws(()=>evaluateWaves(l,[.5,...w.slice(1)]));const a=evaluateWaves(l,[...w.slice(0,-1),.73123456789]).lattice!;assert.equal(a.system.phase,.73123456789);
});

test('continuous wave-number controls preserve finite small-gradient observations and explicit ratio range states',()=>{
 const v=[...data.wavesLabs[g].presets[0].values];v[2]=1e-13;v[3]=8;v[5]=0;v[6]=0;let a=evaluateWaves(g,v).gaussian!.query;close(a.localCrestVelocity!,8e13,2e-15);assert.equal(a.localCrestVelocityStatus,'finite');assert.equal(a.carrierPhaseVelocityStatus,'finite');
 v[2]=0;a=evaluateWaves(g,v).gaussian!.query;assert.equal(a.localCrestVelocity,null);assert.equal(a.localCrestVelocityStatus,'zero-phase-gradient');v[2]=Number.MIN_VALUE;a=evaluateWaves(g,v).gaussian!.query;assert.equal(a.localCrestVelocity,null);assert.equal(a.localCrestVelocityStatus,'unrepresentable-ratio');assert.equal(a.carrierPhaseVelocity,null);
});
