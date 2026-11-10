import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { bindWorkerMessageError, clearWorkerMessageError } from '../src/lib/learning-worker-message-error';
import inputs from './fixtures/worker-messageerror-inputs.json';
const requireModule = createRequire(resolve('package.json'));
type State = {status:string; inputKey?:string|null; input?:Record<string,unknown>|null; error?:string};
type Controller = {run(value:unknown):number|null; cancel():void; dispose():void; getSnapshot():State};
class Port extends EventTarget {
  onmessage:Worker['onmessage']=null; onerror:Worker['onerror']=null; onmessageerror:Worker['onmessageerror']=null;
  terminated=0; listeners=new Set<EventListenerOrEventListenerObject>();
  postMessage(value:unknown) {void value;}
  terminate(){this.terminated++;}
  override addEventListener(type:string, listener:EventListenerOrEventListenerObject|null, options?:AddEventListenerOptions|boolean){
    if(type==='messageerror'&&listener)this.listeners.add(listener); super.addEventListener(type,listener,options);
  }
  override removeEventListener(type:string, listener:EventListenerOrEventListenerObject|null, options?:EventListenerOptions|boolean){
    if(type==='messageerror'&&listener)this.listeners.delete(listener); super.removeEventListener(type,listener,options);
  }
  emit(data:unknown){this.onmessage?.call(this as unknown as Worker,new MessageEvent('message',{data}));}
  decodeError(){this.dispatchEvent(new MessageEvent('messageerror'));}
}
const files=readdirSync('src/lib').filter(x=>/^learning-.*-controller\.ts$/.test(x));
test('every controller has a representative input',()=>{
  assert.deepEqual(Object.keys(inputs).sort(),files.map(x=>x.replace('-controller.ts','')).sort());
});
for(const file of files){
  const base=file.replace('-controller.ts',''), source=readFileSync('src/lib/'+file,'utf8');
  const dual=source.includes('this.factories.calculate()');
  const controllerExports=requireModule('./src/lib/'+file) as Record<string,unknown>;
  const Constructor=Object.entries(controllerExports).find(([name])=>name.endsWith('CalculationController'))![1] as new(f:unknown,notify:(s:State)=>void,options?:unknown,timeout?:number)=>Controller;
  const input=(inputs as Record<string,unknown>)[base];
  function harness(timeout=30000){
    const calculation:Port[]=[],validation:Port[]=[];
    const create=(ports:Port[])=>()=>{const p=new Port();ports.push(p);return p;};
    const factory=dual?{calculate:create(calculation),validate:create(validation)}:create(calculation);
    return {controller:new Constructor(factory,()=>{},{pause:async()=>{}},timeout),calculation,validation};
  }
  function released(p:Port){assert.equal(p.terminated,1);assert.equal(p.listeners.size,0);assert.equal(p.onmessageerror,null);}
  test(`${base}: EventTarget error, retry, replacement, cancel and disposal`,()=>{
    const {controller:c,calculation:ports}=harness();
    try{
      c.run(input);assert.equal(c.getSnapshot().status,'pending');assert.equal(ports.length,1);assert.equal(ports[0].listeners.size,1);assert.equal(ports[0].onmessageerror,null);
      const stale=[...ports[0].listeners][0] as EventListener;
      ports[0].decodeError();assert.equal(c.getSnapshot().status,'error');assert.equal(c.getSnapshot().error,'invalid-response');released(ports[0]);
      c.run(input);assert.equal(c.getSnapshot().status,'pending');stale(new MessageEvent('messageerror'));assert.equal(c.getSnapshot().status,'pending');
      c.run(input);released(ports[1]);ports[1].decodeError();assert.equal(c.getSnapshot().status,'pending');
      c.cancel();released(ports[2]);ports[2].decodeError();assert.equal(c.getSnapshot().status,'idle');
      c.run(input);c.dispose();released(ports[3]);assert.equal(c.run(input),null);assert.equal(ports.length,4);
    }finally{c.dispose();}
  });
  if(dual)test(`${base}: validation-port decode error and supported deadlines detach listeners`,async()=>{
    const h=harness(1000),c=h.controller;
    try{
      const id=c.run(input),state=c.getSnapshot();
      const protocol=readFileSync('src/lib/'+base+'-protocol.ts','utf8');
      const tag=(state.input?.kind as string)??protocol.match(/\bkind\s*:\s*'([^']+)'/)?.[1];
      const response:Record<string,unknown>={version:1,requestId:id,ok:true,kind:tag,input:state.input,inputKey:state.inputKey,result:{input:state.input}};
      if(base==='learning-qubit')response.observations={schemaVersion:1,coordinates:[],states:[]};
      h.calculation[0].emit(response);
      for(let i=0;i<100&&!h.validation.length&&c.getSnapshot().status==='pending';i++)await new Promise(r=>setTimeout(r,1));
      assert.equal(h.validation.length,1,JSON.stringify(c.getSnapshot()));released(h.calculation[0]);
      h.validation[0].decodeError();assert.equal(c.getSnapshot().error,'invalid-response');released(h.validation[0]);
    }finally{c.dispose();}
    if(!source.includes('timeoutMilliseconds'))return;
    const deadline=harness(10);
    try{deadline.controller.run(input);await new Promise(r=>setTimeout(r,30));assert.equal(deadline.controller.getSnapshot().error,'timeout');released(deadline.calculation[0]);}finally{deadline.controller.dispose();}
  });
}
test('minimal ports retain property fallback and rebinding replaces the prior handler',()=>{
  const p:Pick<Worker,'onmessageerror'>={onmessageerror:null};let first=0,second=0;
  bindWorkerMessageError(p,()=>first++);bindWorkerMessageError(p,()=>second++);
  p.onmessageerror!.call(p as Worker,new MessageEvent('messageerror'));assert.equal(first,0);assert.equal(second,1);
  clearWorkerMessageError(p);clearWorkerMessageError(p);assert.equal(p.onmessageerror,null);
  const native=new Port();bindWorkerMessageError(native,()=>first++);bindWorkerMessageError(native,()=>second++);
  assert.equal(native.listeners.size,1);native.decodeError();assert.equal(first,0);assert.equal(second,2);clearWorkerMessageError(native);native.decodeError();assert.equal(second,2);
});
