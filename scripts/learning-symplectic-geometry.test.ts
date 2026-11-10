import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import refs from './fixtures/symplectic-geometry-independent.json';
import {fourDimensionalSymplecticMaps as maps,lagrangianMomentumGraph as graphs} from '../src/lib/learning-symplectic-geometry';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} vs ${b}`);
const n=(s:string)=>{const[a,b='1']=s.split('/');return Number(a)/Number(b);};
const labels=(v:{label:string}[])=>v.map(x=>x.label),matrixLabels=(m:{label:string}[][])=>m.map(labels);
const vector=(a:number[],b:number[])=>{assert.equal(a.length,b.length);a.forEach((x,j)=>close(x,b[j]));};
const mp=[3,2,1,0,0,0,0,40],gp=[0,0,0,2,0,0,2,2,0,0];

test('300independent exact Hessian-generated4Dmaps verify full pullbacks,inverses,probes and symbolic edge integrals',()=>{
 assert.equal(refs.maps.length,300);
 for(const row of refs.maps){const r=maps(row.p);assert.deepEqual(matrixLabels(r.matrix),row.matrix);assert.deepEqual(matrixLabels(r.inverseMatrix),row.inverse);assert.deepEqual(matrixLabels(r.pullbackForm),row.pullback);assert.deepEqual(matrixLabels(r.pushforwardPoisson),row.pushforward);assert.equal(r.determinant.label,row.determinant);assert.equal(r.isSymplectic,row.isSymplectic);assert.equal(r.preservesCanonicalBrackets,row.isSymplectic);assert.equal(r.probeRank,row.probeRank);assert.deepEqual(labels(r.originalPairAreas),row.originalPairAreas);assert.deepEqual(labels(r.transformedPairAreas),row.transformedPairAreas);assert.equal(r.originalSymplecticArea.label,row.originalArea);assert.equal(r.transformedSymplecticArea.label,row.transformedArea);assert.deepEqual(labels(r.query.original),row.queryOriginal);assert.deepEqual(labels(r.query.transformed),row.queryTransformed);assert.equal(r.query.originalCumulativeAction.label,row.queryOriginalAction);assert.equal(r.query.transformedCumulativeAction.label,row.queryTransformedAction);assert.equal(r.originalClosedAction.label,row.closedOriginalAction);assert.equal(r.transformedClosedAction.label,row.closedTransformedAction);assert.equal(r.originalProbeIsLagrangian,row.probeRank===2&&row.originalArea==='0');assert.equal(r.transformedProbeIsLagrangian,row.probeRank===2&&row.transformedArea==='0');}
});
test('240independent exact graph derivatives,eigensigns and symbolic path integrals verify graph structure',()=>{
 assert.equal(refs.graphs.length,240);
 for(const row of refs.graphs){const r=graphs(row.p);assert.deepEqual(matrixLabels(r.matrix),row.matrix);assert.deepEqual(matrixLabels(r.pullbackForm),row.pullback);assert.equal(r.graphDimension,row.graphRank);assert.equal(r.momentumMapRank,row.momentumRank);assert.equal(r.momentumMapDeterminant.label,row.momentumDeterminant);assert.equal(r.isLagrangianGraph,row.isLagrangian);assert.equal(r.symmetricActionSignature,row.signature);vector(r.symmetricActionEigenvalues,row.eigenvalues.map(Number));assert.deepEqual(labels(r.query.position),row.queryPosition);assert.deepEqual(labels(r.query.momentum),row.queryMomentum);assert.deepEqual(labels(r.query.gradientSymmetricAction),row.queryGradient);assert.equal(r.query.symmetricAction.label,row.querySymmetricAction);assert.equal(r.query.cumulativeAction.label,row.queryCumulativeAction);assert.equal(r.query.potentialChange.label,row.queryPotentialChange);assert.equal(r.closedLoopAction.label,row.closedLoopAction);assert.equal(r.curlCoefficient.label,row.curl);assert.deepEqual(labels(r.lineTangent),row.lineTangent);}
});
test('all scale pairs separate symplectic from volume-preserving comparison families exactly',()=>{
 for(let kind=0;kind<4;kind++)for(let a=1;a<=4;a++)for(let b=1;b<=4;b++){const r=maps([kind,a,b,3,-2,3,1,17]);assert.equal(r.determinant.label,'1');assert.equal(r.mapRank,4);assert.equal(r.isVolumePreserving,true);assert.equal(r.symplecticVolumeScale.label,'1');assert.equal(r.isSymplectic,kind!==3||a===b);assert.equal(r.inverseIsSymplectic,r.isSymplectic);}
});
test('reciprocal scaling changes coordinates while retaining both canonical-pair action areas',()=>{
 const r=maps([0,2,1,0,0,0,0,40]);assert.deepEqual(labels(r.originalPairAreas),['1','0']);assert.deepEqual(labels(r.transformedPairAreas),['1','0']);assert.equal(r.isSymplectic,true);assert.deepEqual(labels(r.query.original),['1','0','1','0']);assert.deepEqual(labels(r.query.transformed),['2','0','1/2','0']);
});
test('coupled canonical shears exchange projected pair areas without changing the total form',()=>{
 const r=maps([2,1,1,0,2,0,0,40]);assert.deepEqual(labels(r.transformedProbeU),['2','0','0','1']);assert.deepEqual(labels(r.transformedProbeV),['0','1','1','0']);assert.deepEqual(labels(r.transformedPairAreas),['2','-1']);assert.equal(r.transformedSymplecticArea.label,'1');assert.equal(r.transformedClosedAction.label,'-1');assert.equal(r.isSymplectic,true);
});
test('determinant-one comparison scales two action pairings by2andonehalf',()=>{
 const first=maps(mp),p=[...mp];p[5]=1;const second=maps(p);assert.equal(first.determinant.label,'1');assert.equal(first.isSymplectic,false);assert.equal(first.originalSymplecticArea.label,'1');assert.equal(first.transformedSymplecticArea.label,'2');assert.equal(second.transformedSymplecticArea.label,'1/2');assert.equal(first.originalClosedAction.label,'-1');assert.equal(first.transformedClosedAction.label,'-2');
});
test('a passing zero-area configuration probe does not certify a failed full map',()=>{
 const r=maps([3,2,1,3,-2,2,0,0]);assert.equal(r.isSymplectic,false);assert.equal(r.selectedProbePreservesForm,true);assert.equal(r.probeRank,2);assert.equal(r.areaRatio,null);assert.equal(r.originalProbeIsLagrangian,true);assert.equal(r.transformedProbeIsLagrangian,true);assert.ok(r.formDefect.flat().some(v=>v.label!=='0'));
});
test('the volume-preserving map can destroy a rank-two Lagrangian plane',()=>{
 const r=maps([3,2,1,0,0,3,0,0]);assert.equal(r.originalSymplecticArea.label,'0');assert.equal(r.transformedSymplecticArea.label,'3/2');assert.equal(r.probeRank,2);assert.equal(r.originalProbeIsLagrangian,true);assert.equal(r.transformedProbeIsLagrangian,false);assert.equal(r.selectedProbePreservesForm,false);assert.equal(r.areaRatio,null);
});
test('dependent probes have rank one and undefined area ratios under every family',()=>{
 for(let kind=0;kind<4;kind++){const r=maps([kind,3,2,-3,4,5,3,51]);assert.equal(r.probeRank,1);assert.equal(r.originalSymplecticArea.label,'0');assert.equal(r.transformedSymplecticArea.label,'0');assert.equal(r.selectedProbePreservesForm,true);assert.equal(r.originalProbeIsLagrangian,false);assert.equal(r.transformedProbeIsLagrangian,false);assert.equal(r.areaRatio,null);}
});
test('inverse and pullback defect identities stay exact across rational scale thirds',()=>{
 for(const row of refs.maps.slice(0,50)){const r=maps(row.p);assert.ok(r.inverseIdentityResidual.flat().every(x=>x.label==='0'));assert.equal(r.probeDefectIdentityResidual.label,'0');assert.equal(r.pfaffianCanonical.label,'-1');assert.equal(r.pfaffianPullback.label,'-1');assert.equal(r.originalActionAreaResidual.label,'0');assert.equal(r.transformedActionAreaResidual.label,'0');}
});
test('inactive shear controls do not change a pure scaling;active shear ordering is substantive',()=>{
 const a=maps([0,2,3,0,0,0,0,20]),b=maps([0,2,3,4,-4,0,0,20]);assert.deepEqual(a.matrix,b.matrix);assert.deepEqual(a.cumulativeActionCurves,b.cumulativeActionCurves);assert.equal(a.activeKick,false);assert.equal(a.activeDrift,false);const k=maps([1,2,3,2,2,0,0,20]),tk=maps([2,2,3,2,2,0,0,20]);assert.equal(k.activeKick,true);assert.equal(k.activeDrift,false);assert.equal(tk.activeDrift,true);assert.notDeepEqual(k.matrix,tk.matrix);
});
test('translated loops preserve closed action but can change partial action and query state',()=>{
 const a=maps([1,1,1,2,0,0,0,10]),b=maps([1,1,1,2,0,0,1,10]);assert.equal(a.transformedClosedAction.label,b.transformedClosedAction.label);assert.equal(a.originalClosedAction.label,b.originalClosedAction.label);assert.notDeepEqual(a.query.original,b.query.original);assert.notEqual(a.query.transformedCumulativeAction.label,b.query.transformedCumulativeAction.label);
});
test('boundary selection changes a geometric cursor without altering maps or full loops',()=>{
 const a=maps(mp),p=[...mp];p[7]=80;const b=maps(p);assert.deepEqual(a.matrix,b.matrix);assert.deepEqual(a.originalLoopProjections,b.originalLoopProjections);assert.deepEqual(a.transformedLoopProjections,b.transformedLoopProjections);assert.deepEqual(a.cumulativeActionCurves,b.cumulativeActionCurves);assert.equal(b.query.boundaryParameter,4);vector(b.query.original.map(x=>x.value),[0,0,0,0]);assert.equal(b.query.originalCumulativeAction.label,b.originalClosedAction.label);
});
test('form and Poisson block coefficients retain different physical unit assignments',()=>{
 const r=maps([3,2,1,2,2,0,0,0]);assert.equal(r.mapCoefficientUnits[0][1],'1');assert.equal(r.mapCoefficientUnits[0][2],'s/kg');assert.equal(r.mapCoefficientUnits[2][0],'kg/s');assert.equal(r.mapCoefficientUnits[2][3],'1');assert.equal(r.formCoefficientUnits[0][1],'kg/s');assert.equal(r.formCoefficientUnits[0][2],'1');assert.equal(r.formCoefficientUnits[2][3],'s/kg');assert.equal(r.poissonCoefficientUnits[0][1],'s/kg');assert.equal(r.poissonCoefficientUnits[0][2],'1');assert.equal(r.poissonCoefficientUnits[2][3],'kg/s');assert.ok(Number.isFinite(r.transformedFloatingActionResidual));assert.ok(Math.abs(r.transformedFloatingActionResidual)<1e-9);
});
test('symmetric graphs remain Lagrangian for all six quadratic curvature signatures',()=>{
 for(const[coeffs,signature,convex]of [[[2,2,0],'positive-definite',true],[[-2,-2,0],'negative-definite',false],[[2,-2,0],'indefinite',false],[[2,0,0],'positive-semidefinite',true],[[-2,0,0],'negative-semidefinite',false],[[0,0,0],'zero',true]]as const){const r=graphs([...coeffs,0,0,0,2,2,0,40]);assert.equal(r.isLagrangianGraph,true);assert.equal(r.graphDimension,2);assert.equal(r.pullbackRank,0);assert.equal(r.symmetricActionSignature,signature);assert.equal(r.symmetricActionIsConvex,convex);assert.equal(r.hasGeneratingAction,true);assert.equal(r.closedLoopAction.label,'0');}
});
test('the zero section and singular momentum maps still have a full two-dimensional graph',()=>{
 const zero=graphs([0,0,0,0,0,0,2,2,0,0]);assert.equal(zero.momentumMapRank,0);assert.equal(zero.graphDimension,2);assert.equal(zero.isLagrangianGraph,true);const singular=graphs([0,0,2,2,0,0,2,2,0,0]);assert.equal(singular.momentumMapRank,1);assert.equal(singular.graphDimension,2);assert.equal(singular.isLagrangianGraph,false);
});
test('nonsymmetric graph passes the origin point fit while failing the form and loop tests',()=>{
 const r=graphs(gp);assert.deepEqual(labels(r.query.momentum),['0','0']);assert.deepEqual(labels(r.query.momentumMinusGradient),['0','0']);assert.equal(r.isLagrangianGraph,false);assert.equal(r.hasGeneratingAction,false);assert.deepEqual(matrixLabels(r.pullbackForm),[['0','2'],['-2','0']]);assert.equal(r.curlCoefficient.label,'-2');assert.equal(r.closedLoopAction.label,'-2');assert.equal(r.pullbackFlux.label,'2');assert.equal(r.stokesResidual.label,'0');assert.equal(r.symmetricActionIsConvex,true);
});
test('unit-square edge actions follow the stated counterclockwise orientation and omega=-dtheta sign',()=>{
 for(const[code,position,action]of [[0,['0','0'],'0'],[20,['1','0'],'0'],[40,['1','1'],'-1'],[60,['0','1'],'-2'],[80,['0','0'],'-2']]as const){const p=[...gp];p[9]=code;const r=graphs(p);assert.deepEqual(labels(r.query.position),position);assert.equal(r.query.cumulativeAction.label,action);}
});
test('symmetric action endpoint differences match every partial integral,including saddle graphs',()=>{
 const r=graphs([2,-2,2,0,1,-1,3,2,0,0]);assert.equal(r.symmetricActionSignature,'indefinite');assert.deepEqual(r.cumulativeActionCurves[0],r.cumulativeActionCurves[1]);assert.ok(r.cumulativeActionCurves[2].every(([,v])=>v===0));assert.equal(r.closedLoopAction.label,'0');
});
test('antisymmetric sign and rectangle dimensions control exact closed action independently of translation',()=>{
 const a=graphs([0,0,0,2,0,0,2,2,0,80]),b=graphs([0,0,0,-2,0,0,2,2,0,80]),wide=graphs([0,0,0,2,2,-2,4,3,0,80]);assert.equal(a.closedLoopAction.label,'-2');assert.equal(b.closedLoopAction.label,'2');assert.equal(wide.rectangleArea.label,'3');assert.equal(wide.closedLoopAction.label,'-6');assert.equal(wide.closedActionFormulaResidual.label,'0');
});
test('selected lines are isotropic but not half-dimensional even for failed full graphs',()=>{
 for(let code=0;code<4;code++){const p=[...gp];p[8]=code;const r=graphs(p);assert.equal(r.ambientDimension,4);assert.equal(r.ambientFormRank,4);assert.equal(r.ambientFormNondegenerate,true);assert.equal(r.graphDimension,2);assert.equal(r.pullbackRank,2);assert.equal(r.lineDimension,1);assert.equal(r.lineIsIsotropic,true);assert.equal(r.lineIsLagrangian,false);assert.equal(r.linePullback.label,'0');}
});
test('changing a displayed line does not change the full graph,rectangle or generating-action test',()=>{
 const a=graphs(gp),p=[...gp];p[8]=2;const b=graphs(p);assert.deepEqual(a.matrix,b.matrix);assert.deepEqual(a.configurationBoundary,b.configurationBoundary);assert.deepEqual(a.phaseProjections,b.phaseProjections);assert.deepEqual(a.cumulativeActionCurves,b.cumulativeActionCurves);assert.notDeepEqual(a.linePhaseProjections,b.linePhaseProjections);
});
test('the gradient mismatch is the antisymmetric field,not the symmetric action curvature',()=>{
 const r=graphs([2,-2,1,2,2,-2,2,2,0,0]);vector(r.query.momentumMinusGradient.map(x=>x.value),[-1,-1]);assert.equal(r.symmetricActionSignature,'indefinite');assert.equal(r.hasGeneratingAction,false);assert.equal(r.isIsotropicGraph,false);
});
test('graph cursor preserves complete loop data and returns to the same geometric base at closure',()=>{
 const a=graphs(gp),p=[...gp];p[9]=80;const b=graphs(p);assert.deepEqual(a.configurationBoundary,b.configurationBoundary);assert.deepEqual(a.cumulativeActionCurves,b.cumulativeActionCurves);assert.deepEqual(a.linePhaseProjections,b.linePhaseProjections);assert.deepEqual(b.query.position,b.base.position);assert.equal(b.query.potentialChange.label,'0');assert.equal(b.query.actionMinusPotential.label,'-2');
});
for(const[name,fn,p,ranges]of [['maps',maps,mp,[[0,3],[1,4],[1,4],[-4,4],[-4,4],[0,5],[0,3],[0,80]]],['graphs',graphs,gp,[[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[1,4],[1,4],[0,3],[0,80]]]]as const){
 test(`${name}: sparse,nonfinite,fractional,out-of-range and wrong-arity controls reject`,()=>{assert.throws(()=>fn([]));assert.throws(()=>fn([...p,0]));assert.throws(()=>fn(new Array(p.length)));for(let j=0;j<p.length;j++)for(const v of [NaN,Infinity,p[j]+.5,ranges[j][0]-1,ranges[j][1]+1]){const row:number[]=[...p];row[j]=v;assert.throws(()=>fn(row));}});
 test(`${name}: all valid boundaries stay finite,fresh and JSON-safe without exposing BigInts`,()=>{const input=[...p],before=[...p],a=fn(input),b=fn(input);assert.deepEqual(input,before);assert.deepEqual(a,b);const walk=(v:unknown):void=>{assert.notEqual(typeof v,'bigint');if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(walk);};walk(a);for(let j=0;j<p.length;j++)for(const bound of ranges[j]){const row:number[]=[...p];row[j]=bound;walk(fn(row));}a.query.boundaryParameter=999;assert.deepEqual(fn(input),b);assert.doesNotThrow(()=>JSON.stringify(b));});
}
for(const id of ['four-dimensional-symplectic-maps','lagrangian-momentum-graphs'])for(const locale of ['ko','en']as const)test(`${id}.${locale}: detailed theory,worked examples and six explained exercises render strict math`,async()=>{
 const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),c=await getLearningContent(id);assert.ok(c.outline[locale].length>=12);assert.equal(c.outline.ko.length,c.outline.en.length);assert.equal(new Set(c.outline[locale].map(h=>h.id)).size,c.outline[locale].length);assert.ok(c[locale].includes('katex-display'));assert.ok(!c[locale].includes('katex-error'));assert.equal((source.match(/^\d\. /gm)||[]).length,6);assert.ok(source.includes('https://arxiv.org/html/math/0505366v1'));if(locale==='en'){assert.ok(source.split(/\s+/).length>1400);assert.doesNotMatch(source,/[가-힣]/);assert.equal((source.match(/\*\*Solution:\*\*/g)||[]).length,6);}else{assert.ok((source.match(/[가-힣]/g)||[]).length>3000);assert.equal((source.match(/\*\*해설:\*\*/g)||[]).length,6);}
});
