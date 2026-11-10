/** Owned, reader-safe inputs for the finite spatial scalar-field model. */
export type ScalarFieldBoundary='periodic'|'dirichlet-zero'|'neumann-zero';
export type ScalarFieldInput=Readonly<{boundary:ScalarFieldBoundary;spacing:string;speed:string;quadratic:string;quartic:string;position:readonly string[];velocity:readonly string[]}>;
const keys=['boundary','spacing','speed','quadratic','quartic','position','velocity']as const;
function data(value:unknown,key:PropertyKey):unknown{const d=value&&typeof value==='object'?Object.getOwnPropertyDescriptor(value,key):undefined;if(!d||!Object.hasOwn(d,'value'))throw Error('Owned data required');return d.value;}
function rational(value:unknown,sign:'any'|'positive'|'nonnegative'='any'):string{
 if(typeof value!=='string'||value.length>40||!/^(?:0|-?[1-9]\d{0,17})(?:\/[1-9]\d{0,17})?$/.test(value))throw Error('Integer or fraction with at most18 digits required');
 const n=value.split('/')[0];if(sign!=='any'&&(n.startsWith('-')||(sign==='positive'&&n==='0')))throw Error('Coefficient sign invalid');return value;
}
function vector(value:unknown):readonly string[]{
 if(!Array.isArray(value)||Object.getPrototypeOf(value)!==Array.prototype)throw Error('Plain vector required');
 const length=data(value,'length');if(typeof length!=='number'||!Number.isSafeInteger(length)||length<3||length>32||Reflect.ownKeys(value).length!==length+1)throw Error('Dense vector of3–32 sites required');
 return Object.freeze(Array.from({length},(_,i)=>rational(data(value,String(i)))));
}
export function readScalarFieldInput(value:unknown):ScalarFieldInput|null{
 try{
  if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return null;
  const boundary=data(value,'boundary');if(boundary!=='periodic'&&boundary!=='dirichlet-zero'&&boundary!=='neumann-zero')return null;
  const position=vector(data(value,'position')),velocity=vector(data(value,'velocity'));if(position.length!==velocity.length)return null;
  return Object.freeze({boundary,spacing:rational(data(value,'spacing'),'positive'),speed:rational(data(value,'speed'),'positive'),quadratic:rational(data(value,'quadratic')),quartic:rational(data(value,'quartic'),'nonnegative'),position,velocity});
 }catch{return null;}
}
/** Input identity preserves entered rational spelling while fixing property order. */
export function scalarFieldInputKey(input:ScalarFieldInput):string{return JSON.stringify([input.boundary,input.spacing,input.speed,input.quadratic,input.quartic,input.position,input.velocity]);}
