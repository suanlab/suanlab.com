import test from 'node:test';import assert from 'node:assert/strict';import{readFileSync}from'node:fs';import{createHash}from'node:crypto';import ts from'typescript';import{resolve,dirname}from'node:path';
import{fdtKinds,fdtPresets,fdtPresetInput,fdtFields,editFdtDraft}from'../src/lib/learning-fluctuation-dissipation';
import{isFdtInput,snapshotFdtInput,fdtInputKey,parseFdtStoredInput,isFdtSuccessForInput,type FdtInput}from'../src/lib/learning-fluctuation-dissipation-protocol';
import{evaluateFdtRequest as evaluate}from'../src/lib/learning-fluctuation-dissipation-evaluator';
import{FdtCalculationController,type FdtWorkerPort}from'../src/lib/learning-fluctuation-dissipation-controller';
const kind=fdtKinds[0],cases=JSON.parse(readFileSync('scripts/fixtures/fdt-prepared-cases.json','utf8'))as{label:string;input:FdtInput;resultSha256:string}[];
function canonical(v:unknown):string{if(v===null||typeof v!=='object')return JSON.stringify(v);if(Array.isArray(v))return`[${v.map(canonical).join(',')}]`;return`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${canonical((v as Record<string,unknown>)[k])}`).join(',')}}`;}
for(const[i,c]of cases.entries())test(c.label+': preserve every independently checked numerical field, identity and availability',()=>{
 const input=fdtPresetInput(kind,i);assert.deepEqual(input,snapshotFdtInput(c.input));assert(Object.isFrozen(input)&&Object.isFrozen(input.parameters));assert.notEqual(input.parameters,fdtPresets[kind][i].input.parameters);
 const r=evaluate({version:1,requestId:7,input});assert(r.ok);assert(isFdtSuccessForInput(r,input,7));assert(isFdtSuccessForInput(structuredClone(r),input,7));assert(isFdtSuccessForInput(JSON.parse(JSON.stringify(r)),input,7));assert.equal(isFdtSuccessForInput(r,input,8),false);
 assert.equal(createHash('sha256').update(canonical(r.result)).digest('hex'),c.resultSha256);
 assert.deepEqual(parseFdtStoredInput(JSON.stringify({version:1,value:input}),kind),input);
});
test('draft and saved integer lexemes reject silent rounding, underflow and cross-family values',()=>{
 const input=fdtPresetInput(kind,4),before=fdtInputKey(input);
 for(const f of fdtFields(input)){for(const raw of ['1.0','1e0','1e-999','1.00000000000000001','NaN','Infinity','','1000001'])assert.equal(editFdtDraft(input,f.id,raw),null);assert(editFdtDraft(input,f.id,String(f.value)));}
 assert.equal(editFdtDraft(input,'mobilityNumerator','0'),null);assert.equal(editFdtDraft(input,'angularNumerator','5'),null);assert.equal(editFdtDraft(input,'forceXNumerator','5'),null);assert.equal(editFdtDraft(input,'timeNumerator','5'),null);assert.equal(editFdtDraft(input,'sampleSteps','65'),null);
 assert(editFdtDraft(input,'forceXNumerator','-2'));assert(editFdtDraft(input,'timeNumerator','0'));assert.equal(fdtInputKey(input),before);
 const raw=JSON.stringify({version:1,value:input});for(const bad of ['not-json',raw.replace('"version":1','"version":2'),raw.replace('"angularNumerator":1','"angularNumerator":1.00000000000000001'),raw.replace('"forceYNumerator":0','"forceYNumerator":1e-999'),JSON.stringify({version:1,value:input,extra:1})])assert.equal(parseFdtStoredInput(bad,kind),null);
 assert.equal(parseFdtStoredInput(raw,'other' as typeof kind),null);assert(!isFdtInput(Object.create(input)));assert(!isFdtInput({...input,parameters:Object.create(input.parameters)}));
});
test('guard rejects false physical metadata, missing records and contradictory log/ratio availability',()=>{
 const input=fdtPresetInput(kind,4),response=evaluate({version:1,requestId:1,input});assert(response.ok);
 const corrupt=(change:(r:any)=>void)=>{const r=structuredClone(response);change(r);assert.equal(isFdtSuccessForInput(r,input,1),false);};
 corrupt(r=>r.inputKey='stale');corrupt(r=>r.result.history.pop());corrupt(r=>r.result.history[1].step=0);corrupt(r=>r.result.current.time=null);corrupt(r=>r.result.history[0].tZeroIsRightLimit=false);
 corrupt(r=>r.result.classification.thermalEinsteinRelationSpecified=false);corrupt(r=>r.result.classification.backgroundDetailedBalance=true);corrupt(r=>r.result.history[0].effectiveThermalEnergy[0][1]=r.result.thermalEnergy);corrupt(r=>r.result.history[0].effectiveThermalEnergyStatus[0][1]='available');
 corrupt(r=>r.result.history[0].absoluteResponseLogStatus[0][1]='available');corrupt(r=>r.result.history[1].logAbsoluteImpulseResponse[0][0]=null);corrupt(r=>r.result.history[1].impulseResponse[0].pop());corrupt(r=>r.result.diffusion.bounds.lower.denominator.coefficient='0');corrupt(r=>r.result.extra=1);
 assert.equal(evaluate({version:2,requestId:1,input}).ok,false);
});
class Fake implements FdtWorkerPort{onmessage:Worker['onmessage']=null;onerror:Worker['onerror']=null;onmessageerror:Worker['onmessageerror']=null;sent:unknown;terminated=0;postMessage(v:unknown){this.sent=v;}terminate(){this.terminated++;}send(v:unknown){this.onmessage?.call(this as unknown as Worker,{data:v}as MessageEvent);}}
test('ACK, stop, replacement, obsolete replies, malformed output, retry and unmount own worker lifetimes',()=>{
 const ports:Fake[]=[],c=new FdtCalculationController(()=>{const p=new Fake;ports.push(p);return p;},()=>{}),a=fdtPresetInput(kind,0),b=fdtPresetInput(kind,11);
 c.run(a);const old=ports[0].onmessage;ports[0].send({version:1,requestId:1,phase:'started',kind});c.run(b);assert.equal(ports[0].terminated,1);old?.call({}as Worker,{data:evaluate(ports[0].sent)}as MessageEvent);assert.equal(c.getSnapshot().status,'pending');ports[1].send(evaluate(ports[1].sent));assert.equal(c.getSnapshot().status,'ready');assert.equal(ports[1].terminated,1);
 c.run(a);c.cancel();assert.equal(ports[2].terminated,1);c.run(a);ports[3].send({version:1,requestId:4,ok:true});assert.equal(c.getSnapshot().status,'error');assert.equal(ports[3].terminated,1);c.run(a);ports[4].send(evaluate(ports[4].sent));assert.equal(c.getSnapshot().status,'ready');c.run(a);c.dispose();assert.equal(ports[5].terminated,1);assert.equal(c.run(a),null);
});
test('main UI dependency graph excludes the thermal numerical model and evaluator',()=>{
 const seen=new Set<string>(),visit=(path:string)=>{if(seen.has(path))return;seen.add(path);const ast=ts.createSourceFile(path,readFileSync(path,'utf8'),ts.ScriptTarget.Latest,true);for(const stmt of ast.statements){if(!ts.isImportDeclaration(stmt)||!ts.isStringLiteral(stmt.moduleSpecifier)||stmt.importClause?.isTypeOnly)continue;const c=stmt.importClause;if(c&&!c.name&&c.namedBindings&&ts.isNamedImports(c.namedBindings)&&c.namedBindings.elements.every(e=>e.isTypeOnly))continue;const spec=stmt.moduleSpecifier.text;if(!spec.startsWith('.')&&!spec.startsWith('@/'))continue;const base=spec.startsWith('@/')?resolve('src',spec.slice(2)):resolve(dirname(path),spec);for(const suffix of ['.ts','.tsx','/index.ts','/index.tsx']){try{readFileSync(base+suffix);visit(base+suffix);break;}catch{}}}};
 for(const p of ['src/components/learning/fluctuation-dissipation-lab.tsx','src/components/learning/fluctuation-dissipation-concept.tsx'])visit(resolve(p));assert(!seen.has(resolve('src/lib/learning-thermal-ou-response.ts')));assert(!seen.has(resolve('src/lib/learning-fluctuation-dissipation-evaluator.ts')));assert(seen.has(resolve('src/lib/learning-fluctuation-dissipation-worker-factory.ts')));
});
