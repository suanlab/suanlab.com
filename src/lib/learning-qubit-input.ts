/** Reader-safe structural input ownership. No BigInt or mathematical engines. */
import type { CircuitGate } from './learning-qubit-circuit-exact';
export type QubitWireScalar=Readonly<{real:readonly[string,string];imaginary:readonly[string,string]}>;
export type QubitWireMatrix=readonly(readonly QubitWireScalar[])[];
export type QubitWireTerm=Readonly<{weight:string;operator:QubitWireMatrix}>;
export type QubitWireMap=Readonly<{kind:'transpose'}>|Readonly<{kind:'operator-sum';terms:readonly QubitWireTerm[]}>|Readonly<{kind:'matrix-units';images:readonly QubitWireMatrix[]}>;
export type QubitInput=
  |Readonly<{kind:'circuit';sites:number;gates:readonly CircuitGate[];input:readonly QubitWireScalar[]}>
  |Readonly<{kind:'channel';map:QubitWireMap;state:QubitWireMatrix}>
  |Readonly<{kind:'environment';parameter:string;state:QubitWireMatrix}>
  |Readonly<{kind:'instrument';outcomes:readonly[readonly QubitWireTerm[],readonly QubitWireTerm[]];state:QubitWireMatrix;correctZ:boolean}>
  |Readonly<{kind:'deferred';state:QubitWireMatrix}>;
export const qubitKinds=Object.freeze(['circuit','channel','environment','instrument','deferred'] as const);
export type QubitKind=QubitInput['kind'];
export function qubitData(value:unknown,keys:readonly string[]):Record<string,unknown>|null {
  try{
    if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return null;
    const out:Record<string,unknown>={};
    for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))return null;Object.defineProperty(out,key,{value:d.value,enumerable:true});}
    return out;
  }catch{return null;}
}
function ownField(value:unknown,key:string):unknown {
  if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype)return undefined;
  const d=Object.getOwnPropertyDescriptor(value,key);return d&&Object.hasOwn(d,'value')?d.value:undefined;
}
function array(value:unknown,max:number,exact?:number):unknown[]{
  if(!Array.isArray(value)||Object.getPrototypeOf(value)!==Array.prototype)throw Error('array');
  const d=Object.getOwnPropertyDescriptor(value,'length');
  if(!d||!Object.hasOwn(d,'value')||!Number.isSafeInteger(d.value)||d.value<0||d.value>max||(exact!==undefined&&d.value!==exact)||Reflect.ownKeys(value).length!==d.value+1)throw Error('length');
  return Array.from({length:d.value},(_,i)=>{const v=Object.getOwnPropertyDescriptor(value,String(i));if(!v||!Object.hasOwn(v,'value'))throw Error('entry');return v.value;});
}
function rational(value:unknown):string {
  if(typeof value!=='string'||value.length>66||!/^(?:0|-?[1-9]\d{0,31})(?:\/[1-9]\d{0,31})?$/.test(value))throw Error('rational');return value;
}
function scalar(value:unknown):QubitWireScalar {
  const d=qubitData(value,['real','imaginary']);if(!d)throw Error('scalar');
  const pair=(p:unknown)=>Object.freeze(array(p,2,2).map(rational)) as readonly[string,string];
  return Object.freeze({real:pair(d.real),imaginary:pair(d.imaginary)});
}
function matrix(value:unknown):QubitWireMatrix{return Object.freeze(array(value,2,2).map(row=>Object.freeze(array(row,2,2).map(scalar))));}
function terms(value:unknown):readonly QubitWireTerm[]{return Object.freeze(array(value,8).map(v=>{const d=qubitData(v,['weight','operator']);if(!d)throw Error('term');return Object.freeze({weight:rational(d.weight),operator:matrix(d.operator)});}));}
function map(value:unknown):QubitWireMap {
  const kind=ownField(value,'kind');
  if(kind==='transpose'&&qubitData(value,['kind']))return Object.freeze({kind});
  if(kind==='operator-sum'){const d=qubitData(value,['kind','terms']);if(d)return Object.freeze({kind,terms:terms(d.terms)});}
  if(kind==='matrix-units'){const d=qubitData(value,['kind','images']);if(d)return Object.freeze({kind,images:Object.freeze(array(d.images,4,4).map(matrix))});}
  throw Error('map');
}
function gate(value:unknown,sites:number):CircuitGate {
  const kind=ownField(value,'gate'),keys=kind==='CX'||kind==='CZ'?['gate','control','target']:kind==='SWAP'?['gate','target','other']:['gate','target'];
  const d=qubitData(value,keys);if(!d||typeof kind!=='string'||!['I','X','Y','Z','H','S','Sdg','T','Tdg','CX','CZ','SWAP'].includes(kind))throw Error('gate');
  const site=(v:unknown):number=>{if(typeof v!=='number'||!Number.isInteger(v)||v<0||v>=sites)throw Error('site');return v;};
  const target=site(d.target);
  if(kind==='CX'||kind==='CZ'){const control=site(d.control);if(control===target)throw Error('control');return Object.freeze({gate:kind,control,target});}
  if(kind==='SWAP'){const other=site(d.other);if(other===target)throw Error('swap');return Object.freeze({gate:kind,target,other});}
  return Object.freeze({gate:kind,target}) as CircuitGate;
}
export function readQubitInput(value:unknown):QubitInput|null {
  try{
    const kind=ownField(value,'kind');
    if(kind==='circuit'){
      const d=qubitData(value,['kind','sites','gates','input']);if(!d||typeof d.sites!=='number'||!Number.isInteger(d.sites)||d.sites<1||d.sites>3)return null;
      const sites=d.sites;return Object.freeze({kind,sites,gates:Object.freeze(array(d.gates,16).map(g=>gate(g,sites))),input:Object.freeze(array(d.input,8,2**sites).map(scalar))});
    }
    if(kind==='channel'){const d=qubitData(value,['kind','map','state']);return d?Object.freeze({kind,map:map(d.map),state:matrix(d.state)}):null;}
    if(kind==='environment'){
      const d=qubitData(value,['kind','parameter','state']);if(!d)return null;
      const parameter=rational(d.parameter),[n,den='1']=parameter.split('/');
      if(n.startsWith('-')||n.length>den.length||(n.length===den.length&&n>den))return null;
      return Object.freeze({kind,parameter,state:matrix(d.state)});
    }
    if(kind==='instrument'){
      const d=qubitData(value,['kind','outcomes','state','correctZ']);if(!d||typeof d.correctZ!=='boolean')return null;
      const outcomes=array(d.outcomes,2,2).map(terms);if(outcomes[0].length+outcomes[1].length>8)return null;
      return Object.freeze({kind,outcomes:Object.freeze(outcomes) as readonly[readonly QubitWireTerm[],readonly QubitWireTerm[]],state:matrix(d.state),correctZ:d.correctZ});
    }
    if(kind==='deferred'){const d=qubitData(value,['kind','state']);return d?Object.freeze({kind,state:matrix(d.state)}):null;}
    return null;
  }catch{return null;}
}
export function snapshotQubitInput(value:unknown):QubitInput {const input=readQubitInput(value);if(!input)throw new RangeError('Invalid qubit input');return input;}
export const qubitInputKey=(input:QubitInput)=>JSON.stringify(snapshotQubitInput(input));
/** JSON syntax is parsed first; then reject duplicate decoded keys and number
 * tokens other than small plain integers (only sites/gate positions use numbers).
 */
function strictJSON(source:string):unknown {
  const value:unknown=JSON.parse(source),stack:{object:boolean;key:boolean;seen:Set<string>}[]=[];
  const tokens=source.match(/"(?:\\.|[^"\\])*"|[{}\[\],:]|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/g)??[];
  for(const token of tokens){
    if(token==='{'||token==='['){stack.push({object:token==='{',key:token==='{',seen:new Set()});continue;}
    if(token==='}'||token===']'){stack.pop();continue;}
    const current=stack.at(-1);
    if(token===','){if(current?.object)current.key=true;continue;}
    if(token===':'){if(current)current.key=false;continue;}
    if(token.startsWith('"')){if(current?.object&&current.key){const key=JSON.parse(token) as string;if(current.seen.has(key))throw Error('Duplicate key');current.seen.add(key);}continue;}
    if(!/^(?:0|[1-9]\d{0,2})$/.test(token))throw Error('Plain small integer required');
  }return value;
}
export function parseQubitInputDraft(source:string):QubitInput|null {try{return typeof source==='string'&&source.length<=65536?readQubitInput(strictJSON(source)):null;}catch{return null;}}
export function parseQubitStoredInput(source:string):QubitInput|null {
  try{if(typeof source!=='string'||source.length>65536)return null;const d=qubitData(strictJSON(source),['version','value']);return d?.version===1?readQubitInput(d.value):null;}catch{return null;}
}
