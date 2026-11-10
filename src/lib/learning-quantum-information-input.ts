/** Small reader-safe wire ownership. Mathematical engines are Worker-side imports. */
import type {QubitWireScalar} from './learning-qubit-input';
export type InformationBloch=readonly[string,string,string];
export type InformationState=readonly[QubitWireScalar,QubitWireScalar];
export const quantumInformationKinds=Object.freeze(['discrimination','entropy','ensemble','relative-entropy','bb84','teleportation','error-correction','majorization','capacity'] as const);
export type QuantumInformationKind=typeof quantumInformationKinds[number];
export type QuantumInformationInput=
 |Readonly<{kind:'discrimination';bloch0:InformationBloch;bloch1:InformationBloch;prior:string}>
 |Readonly<{kind:'entropy';correlationWeight:string;coherence:string;probabilityOne:string;copies:number;retainedRank:string}>
 |Readonly<{kind:'ensemble';bloch0:InformationBloch;bloch1:InformationBloch;prior:string;measurement:'helstrom'|'z'}>
 |Readonly<{kind:'relative-entropy';rho:InformationBloch;sigma:InformationBloch}>
 |Readonly<{kind:'bb84';input:InformationState;comparison:InformationState;attackFraction:string;eveZProbability:string;sampleSize:number}>
 |Readonly<{kind:'teleportation';input:InformationState;a:number;b:number;order:'x-then-z'|'z-then-x'}>
 |Readonly<{kind:'error-correction';input:InformationState;selectedError:string;physicalXProbability:string;errors:readonly string[]}>
 |Readonly<{kind:'majorization';mode:'compare';source:readonly string[];target:readonly string[]}>
 |Readonly<{kind:'majorization';mode:'bell-dilution';a:string;b:string}>
 |Readonly<{kind:'capacity';probability:string;input:InformationBloch}>;
function field(value:unknown,key:string):unknown{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype)throw new RangeError('Plain owned record required');
 const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))throw new RangeError('Owned data field required');return d.value;
}
function record(value:unknown,keys:readonly string[]):Record<string,unknown>{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)throw new RangeError('Exact record fields required');
 const out:Record<string,unknown>={};for(const key of keys)out[key]=field(value,key);return out;
}
function array(value:unknown,min:number,max=min):unknown[]{
 if(!Array.isArray(value)||Object.getPrototypeOf(value)!==Array.prototype)throw new RangeError('Plain dense array required');
 const length=Object.getOwnPropertyDescriptor(value,'length');
 if(!length||!Object.hasOwn(length,'value')||!Number.isSafeInteger(length.value)||length.value<min||length.value>max||Reflect.ownKeys(value).length!==length.value+1)throw new RangeError('Bounded dense array required');
 return Array.from({length:length.value},(_,i)=>{const d=Object.getOwnPropertyDescriptor(value,String(i));if(!d||!Object.hasOwn(d,'value'))throw new RangeError('Owned array entry required');return d.value;});
}
function rational(value:unknown):string{
 if(typeof value!=='string'||value.length>66||!/^(?:0|-?[1-9]\d{0,31})(?:\/[1-9]\d{0,31})?$/.test(value))throw new RangeError('Exact integer/fraction string with at most32 digits required');return value;
}
const decimalLE=(a:string,b:string)=>a.length<b.length||(a.length===b.length&&a<=b);
function probability(value:unknown):string{
 const text=rational(value),[n,d='1']=text.split('/');if(n.startsWith('-')||!decimalLE(n,d))throw new RangeError('Probability must lie in[0,1]');return text;
}
function bloch(value:unknown):InformationBloch{
 return Object.freeze(array(value,3).map(x=>{const text=rational(x),[n,d='1']=text.replace(/^-/,'').split('/');if(!decimalLE(n,d))throw new RangeError('Bloch components lie in[-1,1]');return text;}))as InformationBloch;
}
function scalar(value:unknown):QubitWireScalar{
 const d=record(value,['real','imaginary']);
 const pair=(v:unknown)=>Object.freeze(array(v,2).map(rational))as readonly[string,string];
 return Object.freeze({real:pair(d.real),imaginary:pair(d.imaginary)});
}
const state=(value:unknown)=>Object.freeze(array(value,2).map(scalar))as InformationState;
function integer(value:unknown,min:number,max:number):number{
 if(typeof value!=='number'||!Number.isSafeInteger(value)||Object.is(value,-0)||value<min||value>max)throw new RangeError('Bounded nonnegative integer required');return value;
}
function choice<T extends string>(value:unknown,choices:readonly T[]):T{
 if(typeof value!=='string'||!choices.includes(value as T))throw new RangeError('Unsupported choice');return value as T;
}
function pauli(value:unknown):string{if(typeof value!=='string'||!/^[IXYZ]{3}$/.test(value))throw new RangeError('Three-site Pauli word required');return value;}
function rank(value:unknown,copies:number):string{
 if(typeof value!=='string'||!/^(?:0|[1-9]\d{0,19})$/.test(value))throw new RangeError('Exact rank string required');
 // Decimal doubling avoids Number rounding and keeps BigInt off the reader path.
 const digits=[1];for(let step=0;step<copies;step++){let carry=0;for(let i=0;i<digits.length;i++){const v=2*digits[i]+carry;digits[i]=v%10;carry=Math.floor(v/10);}if(carry)digits.push(carry);}
 if(!decimalLE(value,digits.reverse().join('')))throw new RangeError('Rank exceeds2^copies');return value;
}
export function readQuantumInformationInput(value:unknown):QuantumInformationInput|null{
 try{
  const kind=field(value,'kind');
  if(kind==='discrimination'||kind==='ensemble'){
   const d=record(value,kind==='ensemble'?['kind','bloch0','bloch1','prior','measurement']:['kind','bloch0','bloch1','prior']);
   const common={bloch0:bloch(d.bloch0),bloch1:bloch(d.bloch1),prior:probability(d.prior)};
   return kind==='ensemble'?Object.freeze({kind,...common,measurement:choice(d.measurement,['helstrom','z'])}):Object.freeze({kind,...common});
  }
  if(kind==='entropy'){
   const d=record(value,['kind','correlationWeight','coherence','probabilityOne','copies','retainedRank']),copies=integer(d.copies,1,64);
   return Object.freeze({kind,correlationWeight:probability(d.correlationWeight),coherence:probability(d.coherence),probabilityOne:probability(d.probabilityOne),copies,retainedRank:rank(d.retainedRank,copies)});
  }
  if(kind==='relative-entropy'){const d=record(value,['kind','rho','sigma']);return Object.freeze({kind,rho:bloch(d.rho),sigma:bloch(d.sigma)});}
  if(kind==='bb84'){
   const d=record(value,['kind','input','comparison','attackFraction','eveZProbability','sampleSize']);
   return Object.freeze({kind,input:state(d.input),comparison:state(d.comparison),attackFraction:probability(d.attackFraction),eveZProbability:probability(d.eveZProbability),sampleSize:integer(d.sampleSize,0,256)});
  }
  if(kind==='teleportation'){
   const d=record(value,['kind','input','a','b','order']);return Object.freeze({kind,input:state(d.input),a:integer(d.a,0,1),b:integer(d.b,0,1),order:choice(d.order,['x-then-z','z-then-x'])});
  }
  if(kind==='error-correction'){
   const d=record(value,['kind','input','selectedError','physicalXProbability','errors']);
   return Object.freeze({kind,input:state(d.input),selectedError:pauli(d.selectedError),physicalXProbability:probability(d.physicalXProbability),errors:Object.freeze(array(d.errors,1,16).map(pauli))});
  }
  if(kind==='majorization'){
   const mode=field(value,'mode');
   if(mode==='compare'){const d=record(value,['kind','mode','source','target']);return Object.freeze({kind,mode,source:Object.freeze(array(d.source,1,16).map(probability)),target:Object.freeze(array(d.target,1,16).map(probability))});}
   if(mode==='bell-dilution'){const d=record(value,['kind','mode','a','b']);return Object.freeze({kind,mode,a:probability(d.a),b:probability(d.b)});}
  }
  if(kind==='capacity'){const d=record(value,['kind','probability','input']);return Object.freeze({kind,probability:probability(d.probability),input:bloch(d.input)});}
  return null;
 }catch{return null;}
}
export function snapshotQuantumInformationInput(value:unknown):QuantumInformationInput{
 const input=readQuantumInformationInput(value);if(!input)throw new RangeError('Invalid quantum-information wire input');return input;
}
export const quantumInformationInputKey=(value:unknown)=>JSON.stringify(snapshotQuantumInformationInput(value));
/** Reject duplicate decoded JSON keys and non-plain integer number tokens before ownership. */
function strictJSON(source:string):unknown{
 if(typeof source!=='string'||source.length>32768)throw new RangeError('Draft text budget exceeded');
 const value:unknown=JSON.parse(source),stack:{object:boolean;key:boolean;seen:Set<string>}[]=[];
 for(const token of source.match(/"(?:\\.|[^"\\])*"|[{}\[\],:]|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/g)??[]){
  if(token==='{'||token==='['){if(stack.length>=16)throw new RangeError('Draft nesting budget exceeded');stack.push({object:token==='{',key:token==='{',seen:new Set()});continue;}
  if(token==='}'||token===']'){stack.pop();continue;}
  const top=stack.at(-1);if(token===','){if(top?.object)top.key=true;continue;}if(token===':'){if(top)top.key=false;continue;}
  if(token.startsWith('"')){if(top?.object&&top.key){const key=JSON.parse(token)as string;if(top.seen.has(key))throw new RangeError('Duplicate decoded key');top.seen.add(key);}continue;}
  if(!/^(?:0|[1-9]\d{0,2})$/.test(token))throw new RangeError('Small plain integer number tokens required');
 }
 return value;
}
export function parseQuantumInformationDraft(source:string):QuantumInformationInput|null{try{return readQuantumInformationInput(strictJSON(source));}catch{return null;}}
export function parseQuantumInformationStoredInput(source:string):QuantumInformationInput|null{
 try{const d=record(strictJSON(source),['version','value']);return d.version===1?readQuantumInformationInput(d.value):null;}catch{return null;}
}
