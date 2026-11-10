/** Bounded descriptor-only snapshots. No mathematics, JSON stringify or getters. */
export class QubitWireLimit extends RangeError {}
export class QubitCopyCancelled extends Error {constructor(){super('Qubit copy cancelled');this.name='AbortError';}}
export type QubitCopyOptions={active?:()=>boolean;pause?:()=>Promise<void>;nodesPerTask?:number};
function iterator(value:unknown){
  const completed=new WeakMap<object,unknown>(),ancestors=new WeakSet<object>();let nodes=0,characters=0;
  function* copy(v:unknown,depth:number):Generator<void,unknown>{
    if(++nodes>100000||depth>32)throw new QubitWireLimit('Wire node/depth budget');yield;
    if(v===null||typeof v==='boolean')return v;
    if(typeof v==='number'&&Number.isFinite(v))return v;
    if(typeof v==='string'){characters+=v.length;if(v.length>65536||characters>2000000)throw new QubitWireLimit('Wire string budget');return v;}
    if(!v||typeof v!=='object')throw new RangeError('Plain wire data required');
    if(ancestors.has(v))throw new RangeError('Wire cycle');if(completed.has(v))return completed.get(v);
    const isArray=Array.isArray(v);if(Object.getPrototypeOf(v)!==(isArray?Array.prototype:Object.prototype))throw new RangeError('Wire prototype');
    const keys=Reflect.ownKeys(v);if(keys.some(k=>typeof k!=='string'))throw new RangeError('Wire symbol');
    ancestors.add(v);let out:unknown;
    if(isArray){
      const d=Object.getOwnPropertyDescriptor(v,'length');if(!d||!Object.hasOwn(d,'value')||!Number.isInteger(d.value)||d.value<0||d.value>256||keys.length!==d.value+1)throw new QubitWireLimit('Dense bounded wire array required');
      const entries:unknown[]=[];for(let i=0;i<d.value;i++){const e=Object.getOwnPropertyDescriptor(v,String(i));if(!e||!Object.hasOwn(e,'value'))throw new RangeError('Wire accessor');entries.push(yield* copy(e.value,depth+1));}out=Object.freeze(entries);
    }else{
      if(keys.length>48)throw new QubitWireLimit('Wire record budget');const record:Record<string,unknown>={};
      for(const k of keys as string[]){const d=Object.getOwnPropertyDescriptor(v,k);if(!d||!Object.hasOwn(d,'value'))throw new RangeError('Wire accessor');Object.defineProperty(record,k,{value:yield* copy(d.value,depth+1),enumerable:true});}out=Object.freeze(record);
    }ancestors.delete(v);completed.set(v,out);return out;
  }return copy(value,0);
}
export function snapshotQubitWire(value:unknown):unknown {const work=iterator(value);for(;;){const step=work.next();if(step.done)return step.value;}}
export async function snapshotQubitWireInTasks(value:unknown,options:QubitCopyOptions={}):Promise<unknown>{
  const {active=()=>true,pause=()=>new Promise<void>(r=>setTimeout(r,0)),nodesPerTask=1024}=options;
  if(!Number.isInteger(nodesPerTask)||nodesPerTask<1||nodesPerTask>2048)throw new RangeError('Copy task budget');
  const work=iterator(value);let count=0,start=performance.now();
  for(;;){if(!active())throw new QubitCopyCancelled();const step=work.next();if(step.done)return step.value;
    if(++count>=nodesPerTask||performance.now()-start>=4){await pause();if(!active())throw new QubitCopyCancelled();count=0;start=performance.now();}}
}
