import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { build } from 'esbuild';
import { Worker } from 'node:worker_threads';
import { evaluateThermalRequest } from '../src/lib/learning-thermal-evaluator';
import { ThermalCalculationController, type ThermalWorkerPort } from '../src/lib/learning-thermal-controller';
import { thermalKinds, snapshotThermalInput, thermalInputKey, isThermalInput, isThermalSuccessForInput, parseThermalInputDraft, parseThermalStoredInput, type ThermalInput } from '../src/lib/learning-thermal-protocol';
const families = ['driven', 'reservoir'].map((name, i) => ({ kind: thermalKinds[i], cases: JSON.parse(readFileSync(`scripts/fixtures/${name === 'driven' ? 'driven-two-state' : 'reservoir-two-state'}-thermal-reference.json`, 'utf8')).cases as { label: string; input: ThermalInput['parameters'] }[] }));
const inputs = families.map(f => snapshotThermalInput({ kind: f.kind, parameters: f.cases[0].input } as ThermalInput));
for (const family of families) for (const c of family.cases) test(`transport accepts independently checked ${family.kind}: ${c.label}`, () => {
  const input = snapshotThermalInput({ kind: family.kind, parameters: c.input } as ThermalInput);
  const response = evaluateThermalRequest({ version: 1, requestId: 7, input });
  assert.equal(response.ok, true); assert(isThermalSuccessForInput(structuredClone(response), input, 7));
  assert(!isThermalSuccessForInput(response, input, 8));
});
test('owned immutable snapshots, canonical key order and lexical integer storage', () => {
  for (const input of inputs) {
    const reversed = { kind: input.kind, parameters: Object.fromEntries(Object.entries(input.parameters).reverse()) } as ThermalInput;
    assert.equal(thermalInputKey(reversed), thermalInputKey(input));
    assert(Object.isFrozen(snapshotThermalInput(input).parameters));
    assert.deepEqual(parseThermalInputDraft(JSON.stringify(input)), input);
    assert.deepEqual(parseThermalStoredInput(JSON.stringify({ version: 1, value: input })), input);
    for (const token of ['1e-9999', '1.00000000000000001', '9007199254740993', '1e9999']) {
      const raw = JSON.stringify(input).replace(/"temperatureNumerator":\d+/, `"temperatureNumerator":${token}`);
      assert.equal(parseThermalInputDraft(raw), null);assert.equal(parseThermalStoredInput(`{"version":1,"value":${raw}}`), null);
    }
    const inherited = Object.assign(Object.create({ extra: 1 }), input.parameters);
    assert(!isThermalInput({ kind: input.kind, parameters: inherited }));
    assert(!isThermalInput({ ...input, extra: 1 }));
    assert(!isThermalInput({ ...input, parameters: { ...input.parameters, [Symbol('extra')]: 1 } }));
    assert.equal(parseThermalStoredInput(JSON.stringify({ version: 2, value: input })), null);
  }
});
test('rejects missing, nonfinite, extra, reordered and misattributed result fields', () => {
 for (const input of inputs) {
  const base = evaluateThermalRequest({ version: 1, requestId: 1, input }); assert(base.ok);
  const mutate = (fn: (x: any) => void) => { const x = structuredClone(base);fn(x);assert(!isThermalSuccessForInput(x, input, 1)); };
  mutate(x => { delete x.result.precisionBits; });mutate(x => { x.result.extra = 0; });
  mutate(x => { x.inputKey += ' '; });mutate(x => { x.kind = 'other'; });mutate(x => { x.result.parameters.temperatureNumerator++; });mutate(x => { x.result.parameters[Symbol('extra')] = 1; });
  mutate(x => { x.result.entropyIFTIdentity.value = NaN; });
  mutate(x => { x.result.entropyIFTIdentity.bounds.upper.denominator.coefficient = '0'; });
  mutate(x => { x.result.entropyIFTIdentity.value = 0; });
  if (input.kind === thermalKinds[0]) {
   mutate(x => { x.result.stages.reverse(); });mutate(x => { x.result.jarzynskiApplicable = false; x.result.initialDistributionIsCanonical = false; x.result.jarzynskiIdentity = null; });mutate(x => { x.result.endpointAtoms.pop(); });mutate(x => { x.result.workAtoms[0].reachable = false; });
  } else {
   mutate(x => { x.result.history.reverse(); });mutate(x => { x.result.path.baths[0] = 'C'; });mutate(x => { x.result.path.jumps[0].from = 1; });mutate(x => { x.result.classification.hotterChannel = 'B'; });
  }
 }
});
class Port implements ThermalWorkerPort {
 onmessage: ((event: MessageEvent) => void) | null = null; onerror: ((event: ErrorEvent) => void) | null = null; onmessageerror: ((event: MessageEvent) => void) | null = null;
 sent: unknown[] = []; terminated = 0;
 postMessage(value: unknown) { this.sent.push(value); }
 terminate() { this.terminated++; }
 emit(data: unknown) { this.onmessage?.({ data } as MessageEvent); }
}
test('replacement terminates old computation; stale callbacks cannot publish; final worker is released', () => {
 const ports: Port[] = [], states: string[] = [];const c = new ThermalCalculationController(() => {const p=new Port();ports.push(p);return p;}, s=>states.push(s.status));
 const id=c.run(inputs[0])!; const stale=ports[0].onmessage!;
 c.run(inputs[1]);assert.equal(ports[0].terminated,1);assert.equal(ports[0].onmessage,null);
 stale({data:evaluateThermalRequest({version:1,requestId:id,input:inputs[0]})} as MessageEvent);assert.equal(c.getSnapshot().status,'pending');
 ports[1].emit({version:1,requestId:id,phase:'started',kind:inputs[0].kind});assert.equal(c.getSnapshot().status,'pending');
 ports[1].emit({version:1,requestId:id+1,phase:'started',kind:inputs[1].kind});assert.equal(c.getSnapshot().status,'pending');
 ports[1].emit(evaluateThermalRequest(ports[1].sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(ports[1].terminated,1);
 c.cancel();assert.equal(c.getSnapshot().status,'idle');c.dispose();assert.equal(c.run(inputs[0]),null);assert.equal(ports.length,2);assert.deepEqual(states,['pending','pending','pending','ready','idle']);
});
test('unavailability, clone failure, malformed response, error and cancellation have no synchronous fallback', () => {
 const unavailable=new ThermalCalculationController(()=>{throw Error('unavailable');},()=>{});unavailable.run(inputs[0]);assert.equal(unavailable.getSnapshot().status,'error');unavailable.dispose();
 for(const action of ['clone','malformed','error','messageerror','cancel','dispose','invalid']){
 const port=new Port();if(action==='clone')port.postMessage=()=>{throw Error('clone');};
 const c=new ThermalCalculationController(()=>port,()=>{});c.run(inputs[0]);
 if(action==='malformed')port.emit({version:1,requestId:1,ok:true});
 if(action==='error')port.onerror?.({preventDefault(){}} as ErrorEvent);
 if(action==='messageerror')port.onmessageerror?.({} as MessageEvent);
 if(action==='cancel')c.cancel();if(action==='dispose')c.dispose();if(action==='invalid')c.run({kind:'bad'});
 assert.equal(port.terminated,1);assert.equal(port.onmessage,null);
 if(!['cancel','dispose'].includes(action))assert.equal(c.getSnapshot().status,'error');c.dispose();assert.equal(port.terminated,1);
 }
});
test('reentrant subscribers cannot strand a worker', () => {
 const port=new Port();let creations=0;let c: ThermalCalculationController;
 c=new ThermalCalculationController(()=>{creations++;return port;},s=>{if(s.status==='pending')c.cancel();});c.run(inputs[0]);assert.equal(creations,0);assert.equal(c.getSnapshot().status,'idle');c.dispose();
});
test('browser-facing imports bundle without numerical engines; actual worker entry sends started then matching result', async () => {
 const reader = await build({entryPoints:['src/lib/learning-thermal-controller.ts'],bundle:true,write:false,metafile:true,platform:'browser'});
 const bundled = Object.keys(reader.metafile!.inputs);assert(!bundled.some(p=>/thermodynamics|entropy-production-math|interval-transcendentals/.test(p)));
 const bundledWorker = await build({entryPoints:['src/lib/learning-thermal.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});
 const worker = new Worker(`const {parentPort}=require('node:worker_threads');globalThis.self={postMessage:x=>parentPort.postMessage(x)};${bundledWorker.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`,{eval:true});
 try {
  for(let i=0;i<inputs.length;i++){
   const messages: unknown[]=[];const done=new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('worker timeout')),15000);const error=(e:Error)=>{clearTimeout(timer);reject(e);};const receive=(message:unknown)=>{messages.push(message);if(messages.length===2){clearTimeout(timer);worker.off('message',receive);worker.off('error',error);resolve();}};worker.on('message',receive);worker.once('error',error);});
   worker.postMessage({version:1,requestId:i+1,input:inputs[i]});await done;
   assert.deepEqual(messages[0],{version:1,requestId:i+1,phase:'started',kind:inputs[i].kind});assert(isThermalSuccessForInput(messages[1],inputs[i],i+1));
  }
 }finally{await worker.terminate();}
});
test('request envelopes and parameter families reject malformed or mixed requests', () => {
 for(const value of [null,{}, {version:2,requestId:1,input:inputs[0]}, {version:1,requestId:0,input:inputs[0]}, {version:1,requestId:1,input:inputs[0],extra:1}])assert.equal(evaluateThermalRequest(value).ok,false);
 for(const input of [{kind:inputs[0].kind,parameters:inputs[1].parameters},{kind:'unknown',parameters:inputs[0].parameters}])assert.deepEqual(evaluateThermalRequest({version:1,requestId:1,input}),{version:1,requestId:1,ok:false,error:'invalid-input'});
});
test('pending input is an owned snapshot; malformed progress and errors release the worker', () => {
 for(const message of [{version:1,requestId:1,phase:'started',kind:thermalKinds[0],[Symbol('extra')]:1},{version:1,requestId:1,ok:false,error:'unknown'},{version:1,requestId:1,phase:'started',kind:thermalKinds[1]}]){
 const port=new Port(), c=new ThermalCalculationController(()=>port,()=>{}), input=structuredClone(inputs[0]);c.run(input);
 input.parameters.temperatureNumerator=400;
 const state=c.getSnapshot();assert(state.status==='pending');assert.equal(state.input.parameters.temperatureNumerator,inputs[0].parameters.temperatureNumerator);
 port.emit(message);assert.equal(c.getSnapshot().status,'error');assert.equal(port.terminated,1);c.dispose();
 }
});
test('a worker created during reentrant disposal is terminated before posting', () => {
 const port=new Port();let c:ThermalCalculationController;c=new ThermalCalculationController(()=>{c.dispose();return port;},()=>{});c.run(inputs[0]);assert.equal(port.terminated,1);assert.equal(port.sent.length,0);assert.equal(c.run(inputs[1]),null);
});
