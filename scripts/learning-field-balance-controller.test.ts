import { test,after } from 'node:test';
import assert from 'node:assert/strict';
import { FieldBalanceCalculationController as Controller,type FieldBalanceCalculationState as State,type FieldBalanceWorkerPort as PortType } from '../src/lib/learning-field-balance-controller';
import { evaluateFieldBalanceRequest as calculate } from '../src/lib/learning-field-balance-evaluator';
import { evaluateFieldBalanceValidationRequest as validate } from '../src/lib/learning-field-balance-validation-evaluator';
import {readFileSync} from 'node:fs';
import type {FieldBalanceInput} from '../src/lib/learning-field-balance-input';
const reference=JSON.parse(readFileSync('scripts/fixtures/field-balance-independent.json','utf8'))as{cases:{label:string;input:FieldBalanceInput;expected:Record<string,unknown>}[]};
const reps=reference.cases.filter(c=>['hand-share-half','fixed-instant-zero','quartic-cyclic-counterexample'].includes(c.label));
class Port implements PortType {
  onmessage:PortType['onmessage']=null;onerror:PortType['onerror']=null;onmessageerror:PortType['onmessageerror']=null;
  sent:unknown[]=[];terminated=0;throwPost=false;
  postMessage(value:unknown){if(this.throwPost)throw Error('clone');this.sent.push(structuredClone(value));}
  terminate(){this.terminated++;}
  emit(data:unknown){this.onmessage?.call(this as unknown as Worker,{data} as MessageEvent);}
}
const controllers:Controller[]=[];after(()=>controllers.forEach(c=>c.dispose()));
function harness(options:{pause?:()=>Promise<void>;nodesPerTask?:number}={},timeout=30000){
  const calculations:Port[]=[],validations:Port[]=[],states:State[]=[];
  const factories={calculate:()=>{const p=new Port();calculations.push(p);return p;},validate:()=>{const p=new Port();validations.push(p);return p;}};
  const controller=new Controller(factories,s=>states.push(s),{pause:async()=>{},...options},timeout);controllers.push(controller);
  return {controller,calculations,validations,states,factories};
}
async function until(check:()=>boolean){const start=performance.now();while(!check()){assert(performance.now()-start<3000,'controller deadline');await new Promise(r=>setTimeout(r,0));}}
function deliver(p:Port){const result=calculate(p.sent[0],m=>p.emit(m));p.emit(result);return result;}
async function finish(h:ReturnType<typeof harness>){await until(()=>h.validations.length>0);const p=h.validations.at(-1)!;p.emit(validate(p.sent[0]));await until(()=>h.controller.getSnapshot().status!=='pending');}
for(const c of reps){
  test(`controller owns two trusted ports and admits verified ${c.input.field.boundary}`,async()=>{
    const h=harness();h.controller.run(c.input);deliver(h.calculations[0]);assert.equal(h.calculations[0].terminated,1);await finish(h);
    const state=h.controller.getSnapshot();assert(state.status==='ready');assert.deepEqual(state.response.result,c.expected);
    assert(Object.isFrozen(state.response.result)&&Object.isFrozen(state.input));assert.equal(h.validations[0].terminated,1);
    assert.deepEqual(h.states.map(s=>s.status==='pending'?s.phase:s.status),['starting','calculating','validating','ready']);h.controller.dispose();
  });
  test(`controller rejects well-shaped wrong ${c.input.field.boundary} mathematics`,async()=>{
    const h=harness();h.controller.run(c.input);const response=structuredClone(calculate(h.calculations[0].sent[0]));assert(response.ok);
    (response.result as unknown as Record<string,unknown>).extra=true;h.calculations[0].emit(response);await finish(h);
    const state=h.controller.getSnapshot();assert(state.status==='error'&&state.error==='invalid-response');assert(!h.states.some(s=>s.status==='ready'));
  });
}
test('replacement terminates calculation and captured callbacks cannot publish stale state',()=>{
  const h=harness();h.controller.run(reps[0].input);const old=h.calculations[0],late=old.onmessage!;
  h.controller.run(reps[1].input);const before=h.controller.getSnapshot();late.call(old as unknown as Worker,{data:{bad:true}} as MessageEvent);
  assert.equal(h.controller.getSnapshot(),before);assert.equal(old.terminated,1);h.controller.dispose();assert.equal(h.calculations[1].terminated,1);
});
test('cancel during response copying prevents validation-worker creation',async()=>{
  let resume=()=>{};const h=harness({nodesPerTask:1,pause:()=>new Promise<void>(r=>{resume=r;})});h.controller.run(reps[0].input);deliver(h.calculations[0]);h.controller.cancel();resume();
  await new Promise(r=>setTimeout(r,0));assert.equal(h.controller.getSnapshot().status,'idle');assert.equal(h.validations.length,0);
});
test('cancel during validation rejects a late valid receipt and permits retry',async()=>{
  const h=harness();h.controller.run(reps[0].input);deliver(h.calculations[0]);await until(()=>h.validations.length>0);
  const p=h.validations[0],late=p.onmessage!,receipt=validate(p.sent[0]);h.controller.cancel();late.call(p as unknown as Worker,{data:receipt} as MessageEvent);
  assert.equal(h.controller.getSnapshot().status,'idle');assert.equal(p.terminated,1);
  h.controller.run(reps[1].input);deliver(h.calculations[1]);await until(()=>h.validations.length===2);h.validations[1].emit(validate(h.validations[1].sent[0]));await until(()=>h.controller.getSnapshot().status==='ready');
});
test('cancel during validated-result ownership prevents ready publication',async()=>{
  let admission=false,resume=()=>{};const h=harness({nodesPerTask:1,pause:()=>admission?new Promise<void>(r=>{resume=r;}):Promise.resolve()});
  h.controller.run(reps[0].input);deliver(h.calculations[0]);await until(()=>h.validations.length>0);const p=h.validations[0];admission=true;p.emit(validate(p.sent[0]));h.controller.cancel();resume();
  await new Promise(r=>setTimeout(r,0));assert.equal(h.controller.getSnapshot().status,'idle');assert(!h.states.some(s=>s.status==='ready'));
});
test('new input during verification cannot be overwritten by its predecessor',async()=>{
  const h=harness();h.controller.run(reps[0].input);deliver(h.calculations[0]);await until(()=>h.validations.length>0);
  const p=h.validations[0],late=p.onmessage!,receipt=validate(p.sent[0]);h.controller.run(reps[1].input);late.call(p as unknown as Worker,{data:receipt} as MessageEvent);
  assert.equal(h.controller.getSnapshot().requestId,2);assert.equal(h.controller.getSnapshot().status,'pending');assert.equal(p.terminated,1);
});
test('malformed current frames fail while stale positive IDs are ignored',()=>{
  const h=harness();h.controller.run(reps[0].input);const p=h.calculations[0];p.emit({version:1,requestId:500,ok:true});assert.equal(h.controller.getSnapshot().status,'pending');
  p.emit({version:1,requestId:1,ok:true});assert.equal(h.controller.getSnapshot().status,'error');assert.equal(p.terminated,1);
});
test('input failure creates no worker; factories, message errors and post failures release ports',()=>{
  const h=harness();h.controller.run({kind:'unknown'});assert.equal(h.calculations.length,0);
  h.factories.calculate=()=>{throw Error('unavailable');};h.controller.run(reps[0].input);const state=h.controller.getSnapshot();assert(state.status==='error'&&state.error==='unavailable');
  const p=new Port();p.throwPost=true;h.factories.calculate=()=>p;h.controller.run(reps[0].input);assert.equal(p.terminated,1);
  const g=harness();g.controller.run(reps[0].input);g.calculations[0].onmessageerror?.call(g.calculations[0] as unknown as Worker,{} as MessageEvent);assert.equal(g.calculations[0].terminated,1);
});
test('worker errors and a failed validation factory cannot leave a pending job',async()=>{
  const h=harness();h.controller.run(reps[0].input);let prevented=false;
  h.calculations[0].onerror?.call(h.calculations[0] as unknown as Worker,{preventDefault(){prevented=true;}} as ErrorEvent);assert(prevented);assert.equal(h.controller.getSnapshot().status,'error');
  const g=harness();g.factories.validate=()=>{throw Error('validator unavailable');};g.controller.run(reps[0].input);deliver(g.calculations[0]);await until(()=>g.controller.getSnapshot().status==='error');
});
test('a deadline releases an unresponsive calculation Worker and a retry can succeed',async t=>{
  // Advance the configured deadline explicitly. Inline fake Worker mathematics
  // must not race a 25ms wall clock when other numerical suites are running.
  t.mock.timers.enable({apis:['setTimeout']});
  const h=harness({},25);h.controller.run(reps[0].input);
  t.mock.timers.tick(24);assert.equal(h.controller.getSnapshot().status,'pending');
  t.mock.timers.tick(1);
  const state=h.controller.getSnapshot();assert(state.status==='error'&&state.error==='timeout');assert.equal(h.calculations[0].terminated,1);
  h.controller.run(reps[1].input);deliver(h.calculations[1]);
  for(let i=0;!h.validations.length&&i<10000;i++)await Promise.resolve();
  assert.equal(h.validations.length,1);const port=h.validations[0];port.emit(validate(port.sent[0]));
  for(let i=0;h.controller.getSnapshot().status==='pending'&&i<10000;i++)await Promise.resolve();
  assert.equal(h.controller.getSnapshot().status,'ready');
  t.mock.timers.tick(1000);assert.equal(h.controller.getSnapshot().status,'ready');
});
test('a deadline also cancels an unresponsive validator',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const h=harness({},50);h.controller.run(reps[0].input);deliver(h.calculations[0]);
  for(let i=0;!h.validations.length&&i<10000;i++)await Promise.resolve();
  assert.equal(h.validations.length,1);
  t.mock.timers.tick(49);assert.equal(h.controller.getSnapshot().status,'pending');
  t.mock.timers.tick(1);
  const state=h.controller.getSnapshot();assert(state.status==='error'&&state.error==='timeout');assert.equal(h.validations[0].terminated,1);
});
test('resource limits and hostile response getters fail without running caller code',()=>{
  const h=harness();h.controller.run(reps[0].input);h.calculations[0].emit({version:1,requestId:1,ok:false,error:'resource-limit'});
  const state=h.controller.getSnapshot();assert(state.status==='error'&&state.error==='resource-limit');
  const g=harness();g.controller.run(reps[0].input);let called=0;g.calculations[0].emit({get version(){called++;return 1;}});assert.equal(called,0);assert.equal(g.controller.getSnapshot().status,'error');
});
test('disposal is terminal and no factory is invoked afterwards',()=>{
  const h=harness();h.controller.run(reps[0].input);h.controller.dispose();assert.equal(h.calculations[0].terminated,1);assert.equal(h.controller.run(reps[1].input),null);assert.equal(h.calculations.length,1);
});
