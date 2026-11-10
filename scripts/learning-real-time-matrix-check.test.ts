import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import {verifyRealTimeMatrixObservations} from '../src/lib/learning-real-time-matrix-check';
import {buildRealTimeChainExact} from '../src/lib/learning-real-time-phase-exact';
import {realTimeChainCoreWire} from '../src/lib/learning-real-time-phase-core-wire';
import {packRealTimeSymmetricReal,packRealTimeSymmetricComplex} from '../src/lib/learning-real-time-matrix-observations';
import {realTimePhaseExamples,realTimeChainDefault,realTimePacketDefault,type RealTimeChainInput} from '../src/lib/learning-real-time-phase-input';

const root='.runtime/learning/real-time-matrix-check';mkdirSync(root,{recursive:true});
function candidate(input:RealTimeChainInput) {
  const core=buildRealTimeChainExact(input);
  return{exact:realTimeChainCoreWire(core),observations:{
    inverse:core.inertia.zero?null:packRealTimeSymmetricReal(core.d,core.inverseEntry),
    formalConnectedResponse:{interpretation:'complex-generating-response-not-probability-covariance',inverseMultiplier:input.hbar,imaginary:true},
    regulatedConnectedResponse:core.regulator?packRealTimeSymmetricComplex(core.d,core.regulator.covarianceEntry):null}};
}
const measurements:{id:string;realComponents:number;complexComponents:number;milliseconds:number}[]=[];
for(const item of realTimePhaseExamples)if(item.input.mode==='chain'){
  const input=item.input;
  test('all packed components: '+item.id,()=>{
  const {exact,observations}=candidate(input),start=performance.now();
  const result=verifyRealTimeMatrixObservations(input,exact,observations);
  assert(result);assert.equal(result.scope,'finite-matrix-observations-only');assert.equal(result.fullResponseValidated,false);
  const size=input.intervals*(input.intervals-1)/2;
  assert.equal(result.realComponents,observations.inverse?size:0);
  assert.equal(result.complexComponents,observations.regulatedConnectedResponse?size*2:0);
  assert(Object.isFrozen(result.observations));
  measurements.push({id:item.id,realComponents:result.realComponents,complexComponents:result.complexComponents,milliseconds:performance.now()-start});
  writeFileSync(root+`/components-node${process.versions.node.split('.')[0]}.json`,JSON.stringify(measurements,null,2)+'\n');
});}

test('every real and imaginary component rejects a false point interval',()=>{
  const input={...realTimeChainDefault,intervals:5,damping:'1/3'},data=candidate(input);let rejected=0;
  for(const name of ['inverse','regulatedConnectedResponse']as const) {
    const packed=data.observations[name]!;
    for(let i=0;i<packed.upperRows.length;i++)for(let j=0;j<packed.upperRows[i].length;j++) {
      const parts=name==='inverse'?[null]:['real','imaginary'];
      for(const part of parts) {
        const changed=structuredClone(data.observations),target=changed[name]!.upperRows[i][j];
        const range=(part?(target as unknown as Record<string,{lower:string;upper:string}>)[part]:target)as{lower:string;upper:string};
        const wrong=range.lower==='0'&&range.upper==='0'?'1':'0';range.lower=wrong;range.upper=wrong;
        assert.equal(verifyRealTimeMatrixObservations(input,data.exact,changed),null,`${name}/${i}/${j}/${part}`);rejected++;
      }
    }
  }
  assert.equal(rejected,30);
});

test('accept tighter semantic enclosures instead of matching producer bytes',()=>{
  const input={...realTimeChainDefault,intervals:2,time:'2',omegaSquared:'0'},data=candidate(input);
  const packed=data.observations.inverse!; // H=2, H^-1=1/2, exactly dyadic.
  const point=String(BigInt(1)<<BigInt(511));
  packed.upperRows[0][0]={lower:point,upper:point};
  assert(verifyRealTimeMatrixObservations(input,data.exact,data.observations));
  packed.upperRows[0][0]={lower:String(BigInt(point)-BigInt(4)),upper:point};
  assert.equal(verifyRealTimeMatrixObservations(input,data.exact,data.observations),null);
});

test('metadata, scale, shape, attribution and partial scope fail closed',()=>{
  const input={...realTimeChainDefault,damping:'1/3'},data=candidate(input);
  const invalid=[
    (x:typeof data.observations)=>{x.inverse!.scalePowerOfTwo+=100;},
    (x:typeof data.observations)=>{x.inverse!.dimension++;},
    (x:typeof data.observations)=>{x.inverse!.coefficientBits=511;},
    (x:typeof data.observations)=>{x.inverse!.upperRows[0].pop();},
    (x:typeof data.observations)=>{x.inverse!.upperRows[0][0].lower='-0';},
    (x:typeof data.observations)=>{x.inverse!.upperRows[0][0].upper='1'.repeat(161);},
    (x:typeof data.observations)=>{x.formalConnectedResponse.inverseMultiplier='1';},
    (x:typeof data.observations)=>{x.formalConnectedResponse.imaginary=false;},
    (x:typeof data.observations)=>{x.regulatedConnectedResponse=null;},
  ];
  for(const mutate of invalid){const x=structuredClone(data.observations);mutate(x);assert.equal(verifyRealTimeMatrixObservations(input,data.exact,x),null);}
  assert.equal(verifyRealTimeMatrixObservations({...input,mass:'2'},data.exact,data.observations),null);
  assert.equal(verifyRealTimeMatrixObservations(realTimePacketDefault,data.exact,data.observations),null);
  assert.equal(verifyRealTimeMatrixObservations(input,data.exact,{...data.observations,wave:{forged:true}}),null);
  let calls=0;const getter={...data.observations};Object.defineProperty(getter,'inverse',{get(){calls++;throw Error('getter');}});
  assert.equal(verifyRealTimeMatrixObservations(input,data.exact,getter),null);assert.equal(calls,0);
  const proxy=Proxy.revocable({},{});proxy.revoke();assert.equal(verifyRealTimeMatrixObservations(input,data.exact,proxy.proxy),null);
  assert.equal(verifyRealTimeMatrixObservations(input,data.exact,{...data.observations,[Symbol('hidden')]:true}),null);
});

test('singular inverse absence and zero imaginary cells have exact semantics',()=>{
  const input={...realTimeChainDefault,intervals:2,time:'2',omegaSquared:'2',start:'1',end:'-1',probe:1,damping:'1/3'},data=candidate(input);
  assert(verifyRealTimeMatrixObservations(input,data.exact,data.observations));
  const complex=data.observations.regulatedConnectedResponse!;
  assert.deepEqual(complex.upperRows[0][0].imaginary,{lower:'0',upper:'0'});
  complex.upperRows[0][0].imaginary={lower:'-1',upper:'1'};
  assert.equal(verifyRealTimeMatrixObservations(input,data.exact,data.observations),null);
  data.observations.inverse=packRealTimeSymmetricReal(1,()=>({n:BigInt(0),d:BigInt(1)}));
  assert.equal(verifyRealTimeMatrixObservations(input,data.exact,data.observations),null);
});

test('maximum model rejects last complex cell and missing hbar multiplier',()=>{
  const input=realTimePhaseExamples.find(x=>x.id==='maximum-mixed-coprime')!.input as RealTimeChainInput,data=candidate(input);
  const changed=structuredClone(data.observations);changed.regulatedConnectedResponse!.upperRows[30][0].imaginary={lower:'0',upper:'0'};
  assert.equal(verifyRealTimeMatrixObservations(input,data.exact,changed),null);
  const hbar={...realTimeChainDefault,damping:'1/3',hbar:'1/7'},wrong=candidate({...hbar,hbar:'1'});
  const correct=candidate(hbar);correct.observations.regulatedConnectedResponse=wrong.observations.regulatedConnectedResponse;
  assert.equal(verifyRealTimeMatrixObservations(hbar,correct.exact,correct.observations),null);
});

test('matrix checker runtime dependencies exclude every producer engine',()=>{
  const visited=new Set<string>(),queue=['src/lib/learning-real-time-matrix-check.ts'];
  const allowed=new Set(['learning-real-time-matrix-check','learning-real-time-exact-core-check','learning-real-time-validation-rational','learning-real-time-phase-input','learning-real-time-phase-protocol','learning-qubit-wire'].map(x=>'src/lib/'+x+'.ts'));
  while(queue.length){const file=queue.shift()!;if(visited.has(file))continue;visited.add(file);assert(allowed.has(file),file);
    const source=ts.createSourceFile(file,readFileSync(file,'utf8'),ts.ScriptTarget.Latest,true);
    for(const statement of source.statements)if(ts.isImportDeclaration(statement)&&!statement.importClause?.isTypeOnly&&ts.isStringLiteral(statement.moduleSpecifier)){
      const specifier=statement.moduleSpecifier.text;if(specifier.startsWith('.'))queue.push(path.normalize(path.join(path.dirname(file),specifier+'.ts')));
    }
  }
  assert.equal(visited.size,allowed.size);
});
