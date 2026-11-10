import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {build} from 'esbuild';
import {calculateDuffingSecular} from '../src/lib/learning-duffing-secular';
import {duffingSecularDefault,duffingSecularExamples,readDuffingSecularInput,type DuffingSecularInput} from '../src/lib/learning-duffing-secular-input';
import {Rational as Q} from '../src/lib/learning-graph-gauge-validation-math';
import {snapshotQubitWire} from '../src/lib/learning-qubit-wire';
import {getLearningContent} from '../src/lib/learning-content';
const B=BigInt,S=B(1)<<B(384),Z=Q.read('0'),O=Q.read('1'),le=(a:Q,b:Q)=>a.n*b.d<=b.n*a.d;
type Range={lower:string;upper:string};
interface RefPoint {time:string;exactPosition:string;exactVelocity:string;naivePosition:string;firstPosition:string;secondPosition:string;}
interface RefCase {id:string;input:DuffingSecularInput;effectiveCoupling:string;initialEnergy:string;frequency:string|null;period:string|null;phaseDrift:string[]|null;probe:RefPoint&{methods:Record<string,string>[]};fullSamples:RefPoint[];localSamples:RefPoint[];localWindow:{start:string;width:string};residuals:{normalizedResidualPolynomialInCos:string[];normalizedResidualUniformBound:string;physicalResidualUniformBound:string}[];}
const fixture=JSON.parse(readFileSync('scripts/fixtures/duffing-secular-reference.json','utf8'))as{cases:RefCase[];symbolicResidualLowestPowers:number[]};
function agrees(range:Range,cell:string,label:string){const lo=Q.read(range.lower),hi=Q.read(range.upper),n=B(cell);assert(le(lo,hi),label+' ordered');assert(le(lo,new Q(n+B(2),S))&&le(new Q(n-B(1),S),hi),label+' independent numerical agreement');assert(le(hi.sub(lo),new Q(B(1),B(10)**B(30))),label+' useful enclosure width');}
for(const c of fixture.cases)test(`independent elliptic trajectory,frequency,residuals and long-time diagnostics: ${c.id}`,()=>{
 const r=calculateDuffingSecular(c.input);assert.equal(r.effectiveCoupling,c.effectiveCoupling);assert.equal(r.initialEnergy,c.initialEnergy);assert.deepEqual(r.localWindow,c.localWindow);
 if(c.frequency===null){assert.equal(r.mode,'equilibrium');assert.equal(r.frequency,null);assert.equal(r.period,null);assert.equal(r.horizonDiagnostics.phaseDrift,null);}else{assert(r.frequency&&r.period&&r.horizonDiagnostics.phaseDrift);agrees(r.frequency,c.frequency,'frequency');agrees(r.period,c.period!,'period');for(const[i,key]of ['naive','first','second'].entries())agrees(r.horizonDiagnostics.phaseDrift[key as 'naive'|'first'|'second'],c.phaseDrift![i],'unwrapped phase');}
 for(const key of ['fullSamples','localSamples']as const){assert.equal(r[key].length,65);for(const[i,p]of r[key].entries()){const ref=c[key][i];assert.equal(p.time,ref.time);for(const field of ['exactPosition','exactVelocity','naivePosition','firstPosition','secondPosition']as const)agrees(p[field],ref[field],`${key}.${i}.${field}`);}}
 agrees(r.probe.exactPosition,c.probe.exactPosition,'probe exact position');agrees(r.probe.exactVelocity,c.probe.exactVelocity,'probe exact velocity');for(const[i,method]of r.probe.methods.entries())for(const field of ['position','velocity','residual','energy','energyDefect','positionError','velocityError']as const)agrees(method[field],c.probe.methods[i][field],`probe.${i}.${field}`);
 for(const[i,method]of r.approximations.entries()){assert.deepEqual(method.normalizedResidualPolynomialInCos,c.residuals[i].normalizedResidualPolynomialInCos);assert.equal(method.normalizedResidualUniformBound,c.residuals[i].normalizedResidualUniformBound);assert.equal(method.physicalResidualUniformBound,c.residuals[i].physicalResidualUniformBound);}
});
test('seventeen immutable examples retain independent input provenance and symbolic cancellation orders',()=>{
 assert.equal(duffingSecularExamples.length,17);assert.deepEqual(fixture.cases.slice(0,17).map(({id,input})=>({id,input})),duffingSecularExamples);assert.deepEqual(fixture.symbolicResidualLowestPowers,[2,3]);assert.deepEqual(JSON.parse(readFileSync('scripts/fixtures/duffing-secular-examples.json','utf8')),duffingSecularExamples);
});
test('initial data and zero-amplitude equilibrium are handled as exact identities',()=>{
 const initial=calculateDuffingSecular({...duffingSecularDefault,amplitude:'1/10',horizon:'0'});assert.deepEqual(initial.probe.exactPosition,{lower:'1/10',upper:'1/10'});for(const m of initial.probe.methods){assert.deepEqual(m.position,initial.probe.exactPosition);assert.deepEqual(m.positionError,{lower:'0',upper:'0'});assert.deepEqual(m.energyDefect,{lower:'0',upper:'0'});}
 const equilibrium=calculateDuffingSecular({...duffingSecularDefault,amplitude:'0'});assert.equal(equilibrium.frequency,null);assert.equal(equilibrium.period,null);assert.equal(equilibrium.horizonDiagnostics.fullGridResolvesEightPointsPerCycle,null);for(const p of equilibrium.fullSamples){assert.deepEqual(p.exactPosition,{lower:'0',upper:'0'});assert.deepEqual(p.exactVelocity,{lower:'0',upper:'0'});}
});
test('effective coupling includes amplitude and strong-coupling truncations are not called physical frequencies',()=>{
 const r=calculateDuffingSecular({...duffingSecularDefault,lambda:'1',amplitude:'4'});assert.equal(r.effectiveCoupling,'16');assert.equal(r.approximations[1].positiveFrequency,false);assert(Q.read(r.approximations[1].frequency).n<B(0));assert(r.frequency&&Q.read(r.frequency.lower).n>B(0));
 const harmonic=calculateDuffingSecular({...duffingSecularDefault,lambda:'0'});assert.equal(harmonic.mode,'harmonic');assert.deepEqual(harmonic.frequency,{lower:'1',upper:'1'});for(const m of harmonic.probe.methods)assert.deepEqual(m.positionError,{lower:'0',upper:'0'});
});
test('conserved exact energy and bounded amplitude coexist with growing naive values and phase drift',()=>{
 const r=calculateDuffingSecular({...duffingSecularDefault,horizon:'1000000'});assert.equal(r.horizonDiagnostics.fullGridResolvesEightPointsPerCycle,false);assert(Q.read(r.horizonDiagnostics.physicalSecularEnvelope).n>B(0));assert(r.horizonDiagnostics.phaseDrift);assert(Q.read(r.horizonDiagnostics.phaseDrift.naive.upper).n<B(0));
 for(const point of r.fullSamples){assert(le(Q.read('-1'),Q.read(point.exactPosition.lower)));assert(le(Q.read(point.exactPosition.upper),O));}
 assert(r.frequency);const lower=Q.read(r.frequency.lower),upper=Q.read(r.frequency.upper);assert(le(Q.read('201/200'),lower.mul(lower)));assert(le(upper.mul(upper),Q.read('101/100')));
 const referenceEnergy=Q.read(r.initialEnergy);assert(le(Z,referenceEnergy));
 const sq=(range:Range)=>{const lo=Q.read(range.lower),hi=Q.read(range.upper),a=lo.mul(lo),b=hi.mul(hi);return{lower:le(lo,Z)&&le(Z,hi)?Z:le(a,b)?a:b,upper:le(a,b)?b:a};};
 for(const p of r.fullSamples){const x2=sq(p.exactPosition),v2=sq(p.exactVelocity),lambda=Q.read(r.input.lambda),half=Q.read('1/2'),quarter=Q.read('1/4');const lo=v2.lower.mul(half).add(x2.lower.mul(half)).add(x2.lower.mul(x2.lower).mul(lambda).mul(quarter)),hi=v2.upper.mul(half).add(x2.upper.mul(half)).add(x2.upper.mul(x2.upper).mul(lambda).mul(quarter));assert(le(lo,referenceEnergy)&&le(referenceEnergy,hi),'exact reference energy enclosure');}
assert(r.elliptic);assert(le(Q.read(r.elliptic.nome.upper),Q.read('1/16')));assert.equal(r.elliptic.fourierTerms,64);
});
test('bounded input snapshots preserve raw spellings and reject getters,symbols and coercions',()=>{
 const source={...duffingSecularDefault,lambda:'2/200'},owned=readDuffingSecularInput(source);assert(owned&&Object.isFrozen(owned));source.lambda='1';assert.equal(owned.lambda,'2/200');for(const key of ['lambda','amplitude','horizon','probeFraction']as const)for(const v of ['-1','1e-3','1/0',' 1','1234567890123'])assert.equal(readDuffingSecularInput({...duffingSecularDefault,[key]:v}),null);
 assert.equal(readDuffingSecularInput({...duffingSecularDefault,probeFraction:'2'}),null);assert.equal(readDuffingSecularInput({...duffingSecularDefault,lambda:'17'}),null);assert.equal(readDuffingSecularInput({...duffingSecularDefault,extra:true}),null);assert.equal(readDuffingSecularInput({...duffingSecularDefault,[Symbol('s')]:1}),null);let called=0;const accessor={...duffingSecularDefault};Object.defineProperty(accessor,'horizon',{get(){called++;throw Error('getter');}});assert.equal(readDuffingSecularInput(accessor),null);assert.equal(called,0);
});
test('maximum rational and tiny-amplitude outputs fit owned wire budgets without shared mutable values',()=>{
 const rows=[];for(const input of [duffingSecularExamples.find(c=>c.id==='maximum-coprime')!.input,{lambda:'1/999999999983',amplitude:'1/999999999989',horizon:'999999999979',probeFraction:'1'}]){const r=calculateDuffingSecular(input);assert.deepEqual(snapshotQubitWire(r),r);const bytes=JSON.stringify(r).length;assert(bytes<2000000);rows.push({input,bytes});const next=calculateDuffingSecular(input);r.fullSamples[1].exactPosition.lower='forged';assert.notEqual(next.fullSamples[1].exactPosition.lower,'forged');}mkdirSync('.runtime/learning/duffing-secular-foundation',{recursive:true});writeFileSync('.runtime/learning/duffing-secular-foundation/wire-budgets.json',JSON.stringify(rows,null,2)+'\n');
});
test('reader input bundle excludes the exact and transcendental engines',async()=>{
 const result=await build({entryPoints:['src/lib/learning-duffing-secular-input.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});assert.equal(Object.keys(result.metafile!.inputs).length,1);assert(!result.outputFiles[0].text.includes('BigInt('));
});

test('bilingual manuscripts render twenty substantive sections,fifteen display equations and eight explained exercises',async()=>{
 const id='duffing-secular-frequency-long-time',r=await getLearningContent(id),root='.runtime/learning/duffing-secular-foundation';const rows=[];
 for(const language of ['ko','en']as const){const source=readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8');assert.equal(r.outline[language].length,20);assert.equal((source.match(/^\d\. \*\*/gm)??[]).length,8);assert.equal((source.match(/^\$\$$/gm)??[]).length/2,15);assert(source.length>8000);assert(!r[language].includes('katex-error'));assert(!/<h1(?:\s|>)/.test(r[language]));assert(r[language].includes('<math'));if(language==='en')assert(!/[가-힣]/.test(r.en));writeFileSync(`${root}/manuscript-${language}.html`,r[language]);rows.push({language,sections:20,displayEquations:15,explainedExercises:8,sourceCharacters:source.length});}
 writeFileSync(root+'/manuscripts.json',JSON.stringify(rows,null,2)+'\n');
});

test('Fourier certificate fields bound tail formulas rather than claiming actual error lower bounds',()=>{
 const r=calculateDuffingSecular(duffingSecularDefault);assert(r.elliptic);assert(r.elliptic.tailScope.includes('not lower bounds on actual truncation errors'));for(const range of [r.elliptic.positionTailBound,r.elliptic.velocityTailBound]){assert(le(Z,Q.read(range.lower)));assert(Q.read(range.upper).n>B(0));assert(le(Q.read(range.upper),Q.read('1/1000000000000000000000000000000')));}
});
