import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import ts from 'typescript';
import { resolve, dirname } from 'node:path';
import { reversalKinds, reversalPresets, reversalPresetInput, editReversalDraft, editReversalJson, reversalFields } from '../src/lib/learning-cycle-reversal';
import { isReversalInput, snapshotReversalInput, reversalInputKey, parseReversalStoredInput, isReversalSuccessForInput, type ReversalInput } from '../src/lib/learning-cycle-reversal-protocol';
import { evaluateReversalRequest as evaluate } from '../src/lib/learning-cycle-reversal-evaluator';
import { ReversalCalculationController, type ReversalWorkerPort } from '../src/lib/learning-cycle-reversal-controller';
const cases=JSON.parse(readFileSync('scripts/fixtures/cycle-reversal-prepared-cases.json','utf8')) as {label:string;input:ReversalInput;resultSha256:string}[];
const kind=reversalKinds[0];
function canonical(value:unknown):string {
  if(value===null||typeof value!=='object')return JSON.stringify(value);
  if(Array.isArray(value))return`[${value.map(canonical).join(',')}]`;
  return`{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${canonical((value as Record<string,unknown>)[k])}`).join(',')}}`;
}
for(const [i,c] of cases.entries())test(`${c.label}: all independently checked numerical fields preserved and transportable`,()=>{
  const input=reversalPresetInput(kind,i);assert.deepEqual(input,snapshotReversalInput(c.input));
  assert(Object.isFrozen(input)&&Object.isFrozen(input.parameters)&&Object.isFrozen(input.parameters.states)&&Object.isFrozen(input.parameters.dwellNumerators));
  assert.notEqual(input.parameters.states,reversalPresets[kind][i].input.parameters.states);
  const r=evaluate({version:1,requestId:7,input});assert(r.ok);assert(isReversalSuccessForInput(r,input,7));
  assert(isReversalSuccessForInput(structuredClone(r),input,7));assert(isReversalSuccessForInput(JSON.parse(JSON.stringify(r)),input,7));
  assert.equal(isReversalSuccessForInput(r,input,8),false);
  assert.equal(createHash('sha256').update(canonical(r.result)).digest('hex'),c.resultSha256);
  assert.deepEqual(parseReversalStoredInput(JSON.stringify({version:1,value:input}),kind),input);
});
test('scalar and full-path JSON editing preserves original integer lexemes',()=>{
  const input=reversalPresetInput(kind,2),before=reversalInputKey(input);
  for(const f of reversalFields(input)) {
    for(const raw of ['0','1.0','1e0','1e-999','1.0000000000000001','NaN','Infinity','','1000001'])assert.equal(editReversalDraft(input,f.id,raw),null);
    assert(editReversalDraft(input,f.id,String(f.value)));
  }
  const raw=JSON.stringify(input.parameters);
  assert.deepEqual(editReversalJson(input,raw),input);
  for(const bad of ['not-json',raw.replace('"clockwise01":2','"clockwise01":2.00000000000000001'),raw.replace('"states":[0,1,2,0]','"states":[0,1.00000000000000001,2,0]'),raw.replace('"dwellNumerators":[1,1,1,1]','"dwellNumerators":[1e-999,1,1,1]'),raw.replace('"rateDenominator":1','"rateDenominator":0')])assert.equal(editReversalJson(input,bad),null);
  const saved=JSON.stringify({version:1,value:input});
  for(const bad of [saved.replace('"version":1','"version":2'),saved.replace('"timeDenominator":4','"timeDenominator":4.00000000000000001'),saved.replace('"states":[0,1,2,0]','"states":[0,1e-999,2,0]'),JSON.stringify({version:1,value:input,extra:1})])assert.equal(parseReversalStoredInput(bad,kind),null);
  assert.equal(parseReversalStoredInput(saved,'other' as typeof kind),null);
  assert.equal(reversalInputKey(input),before);
});
test('strict payloads reject false availability, missing path rows, wrong prefixes and stale identities',()=>{
  const input=reversalPresetInput(kind,2),response=evaluate({version:1,requestId:1,input});assert(response.ok);
  const corrupt=(mutate:(v:any)=>void)=>{const r=structuredClone(response);mutate(r);assert.equal(isReversalSuccessForInput(r,input,1),false);};
  corrupt(r=>r.inputKey='old');corrupt(r=>r.result.path.pop());corrupt(r=>r.result.path[0].state=2);corrupt(r=>r.result.current.index=0);
  corrupt(r=>r.result.chosenPathOccupation=null);corrupt(r=>r.result.reversedGenerator[0].pop());corrupt(r=>r.result.orientedEdges[0].from=2);
  corrupt(r=>r.result.classification.detailedBalance=true);corrupt(r=>r.result.classification.thermalInterpretationSpecified=true);
  corrupt(r=>r.result.cycleRatio.bounds.lower.denominator.coefficient='0');corrupt(r=>r.result.path[0].totalLogRatio=null);corrupt(r=>r.result.extra=1);
  assert(!isReversalInput(Object.create(input)));assert(!isReversalInput({...input,parameters:Object.create(input.parameters)}));
  assert(!isReversalInput({...input,parameters:{...input.parameters,states:[0,,2],dwellNumerators:[1,1,1]}}));
  assert.equal(evaluate({version:2,requestId:1,input}).ok,false);
});
class Fake implements ReversalWorkerPort {
  onmessage:Worker['onmessage']=null;onerror:Worker['onerror']=null;onmessageerror:Worker['onmessageerror']=null;
  sent:unknown;terminated=0;postMessage(v:unknown){this.sent=v;}terminate(){this.terminated++;}
  send(v:unknown){this.onmessage?.call(this as unknown as Worker,{data:v} as MessageEvent);}
}
test('ACK, replacement, stale handlers, stop, invalid responses, retry and disposal own workers',()=>{
  const ports:Fake[]=[],controller=new ReversalCalculationController(()=>{const p=new Fake();ports.push(p);return p;},()=>{}),a=reversalPresetInput(kind,0),b=reversalPresetInput(kind,18);
  controller.run(a);const old=ports[0].onmessage;ports[0].send({version:1,requestId:1,phase:'started',kind});assert.equal(controller.getSnapshot().status,'pending');
  controller.run(b);assert.equal(ports[0].terminated,1);old?.call({} as Worker,{data:evaluate(ports[0].sent)} as MessageEvent);assert.equal(controller.getSnapshot().status,'pending');
  ports[1].send(evaluate(ports[1].sent));assert.equal(controller.getSnapshot().status,'ready');assert.equal(ports[1].terminated,1);
  controller.run(a);controller.cancel();assert.equal(ports[2].terminated,1);controller.run(a);ports[3].send({version:1,requestId:4,ok:true});assert.equal(controller.getSnapshot().status,'error');assert.equal(ports[3].terminated,1);
  controller.run(a);ports[4].send(evaluate(ports[4].sent));assert.equal(controller.getSnapshot().status,'ready');controller.run(a);controller.dispose();assert.equal(ports[5].terminated,1);assert.equal(controller.run(a),null);
});
test('UI import graph cannot reach this numerical evaluator or model except through the worker URL',()=>{
  const seen=new Set<string>(),visit=(path:string)=>{
    if(seen.has(path))return;seen.add(path);
    const ast=ts.createSourceFile(path,readFileSync(path,'utf8'),ts.ScriptTarget.Latest,true);
    for(const statement of ast.statements){
      if(!ts.isImportDeclaration(statement)||!ts.isStringLiteral(statement.moduleSpecifier)||statement.importClause?.isTypeOnly)continue;
      const clause=statement.importClause;
      if(clause&&!clause.name&&clause.namedBindings&&ts.isNamedImports(clause.namedBindings)&&clause.namedBindings.elements.every(e=>e.isTypeOnly))continue;
      const spec=statement.moduleSpecifier.text;
      if(!spec.startsWith('.')&&!spec.startsWith('@/'))continue;
      const base=spec.startsWith('@/')?resolve('src',spec.slice(2)):resolve(dirname(path),spec);
      if(base.endsWith('.css'))continue;
      let target='';for(const suffix of ['.ts','.tsx','/index.ts','/index.tsx']){try{readFileSync(base+suffix);target=base+suffix;break;}catch{}}
      if(target)visit(target);
    }
  };
  for(const path of ['src/components/learning/cycle-reversal-lab.tsx','src/components/learning/cycle-reversal-concept.tsx'])visit(resolve(path));
  assert(!seen.has(resolve('src/lib/learning-continuous-cycle-reversal.ts')));assert(!seen.has(resolve('src/lib/learning-cycle-reversal-evaluator.ts')));
  assert(seen.has(resolve('src/lib/learning-cycle-reversal-worker-factory.ts')));
});
