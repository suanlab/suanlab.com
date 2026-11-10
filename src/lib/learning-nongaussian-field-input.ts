/** Reader-safe Boltzmann factors, not physical energies. */
export interface NongaussianFieldInput {sites:number;boundary:'open'|'periodic';pairFactor:string;fourBodyFactor:string;temperature:string;sourceFactors:string[];probes:[number,number,number,number];}
const fields=['sites','boundary','pairFactor','fourBodyFactor','temperature','sourceFactors','probes']as const;
function array(value:unknown,n:number):unknown[]|null{
 if(!Array.isArray(value))return null;const d=Object.getOwnPropertyDescriptors(value)as unknown as Record<PropertyKey,PropertyDescriptor>;if(Reflect.ownKeys(d).length!==n+1||!d.length||!('value'in d.length)||d.length.value!==n)return null;const out=[];for(let i=0;i<n;i++){const p=d[String(i)];if(!p||!('value'in p))return null;out.push(p.value);}return out;
}
function positive(value:unknown,factor:boolean):value is string{
 if(typeof value!=='string'||!/^[1-9]\d{0,11}(?:\/[1-9]\d{0,11})?$/.test(value))return false;const[a,b='1']=value.split('/');const n=Number(a),d=Number(b);return n<=16*d&&(!factor||16*n>=d);
}
export function readNongaussianFieldInput(value:unknown):Readonly<NongaussianFieldInput>|null{
 try{if(!value||typeof value!=='object'||Array.isArray(value))return null;const proto=Object.getPrototypeOf(value);if(proto!==Object.prototype&&proto!==null)return null;const d=Object.getOwnPropertyDescriptors(value);if(Reflect.ownKeys(d).length!==fields.length||fields.some(k=>!d[k]||!('value'in d[k])))return null;const own=Object.fromEntries(fields.map(k=>[k,d[k].value])),n=own.sites;
 if(!Number.isInteger(n)||n<4||n>8||!['open','periodic'].includes(own.boundary)||!positive(own.pairFactor,true)||!positive(own.fourBodyFactor,true)||!positive(own.temperature,false))return null;
 const source=array(own.sourceFactors,n),probes=array(own.probes,4);if(!source||!source.every(s=>positive(s,true))||!probes||!probes.every(i=>Number.isInteger(i)&&Number(i)>=0&&Number(i)<n))return null;
 return Object.freeze({sites:n,boundary:own.boundary,pairFactor:own.pairFactor,fourBodyFactor:own.fourBodyFactor,temperature:own.temperature,sourceFactors:Object.freeze(source.slice())as unknown as string[],probes:Object.freeze(probes.slice())as unknown as NongaussianFieldInput['probes']});
 }catch{return null;}
}
export const nongaussianFieldDefault:Readonly<NongaussianFieldInput>=Object.freeze({sites:4,boundary:'periodic',pairFactor:'2',fourBodyFactor:'1',temperature:'1',sourceFactors:Object.freeze(['1','1','1','1'])as unknown as string[],probes:Object.freeze([0,1,2,3])as unknown as NongaussianFieldInput['probes']});
const example=(id:string,patch:Partial<NongaussianFieldInput>)=>{const input=readNongaussianFieldInput({...nongaussianFieldDefault,...patch});if(!input)throw Error('Invalid non-Gaussian preparation '+id);return Object.freeze({id,input});};
export const nongaussianFieldExamples=Object.freeze([
 example('pair-ring',{}),example('independent',{pairFactor:'1'}),example('independent-repeated',{pairFactor:'1',probes:[0,0,0,0]}),example('biased-repeated',{pairFactor:'1',sourceFactors:['2','1','1','1'],probes:[0,0,0,0]}),
 example('localized-source',{sourceFactors:['2','1','1','1']}),example('uniform-source',{sourceFactors:['2','2','2','2']}),example('open-chain',{boundary:'open'}),example('antiferromagnetic',{pairFactor:'1/2'}),example('odd-frustration',{sites:5,pairFactor:'1/2',sourceFactors:['1','1','1','1','1'],probes:[0,1,2,4]}),
 example('pure-four-body',{pairFactor:'1',fourBodyFactor:'2'}),example('negative-four-body',{pairFactor:'1',fourBodyFactor:'1/2'}),example('mixed-interactions',{pairFactor:'3/2',fourBodyFactor:'5/3',sourceFactors:['2','1/2','3/2','1']}),example('same-factors-hotter',{temperature:'2'}),
 example('fixed-energy-reference',{pairFactor:'4',fourBodyFactor:'9',sourceFactors:['4','1','1','1']}),example('hotter-fixed-energy',{pairFactor:'2',fourBodyFactor:'3',sourceFactors:['2','1','1','1'],temperature:'2'}),example('equivalent-spelling',{pairFactor:'4/2',fourBodyFactor:'2/2',temperature:'3/3',sourceFactors:['2/2','3/3','4/4','5/5']}),
 example('maximum-coprime',{sites:8,pairFactor:'999999999983/62499999999',fourBodyFactor:'62499999999/999999999979',temperature:'1/999999999989',sourceFactors:['16','1/16','999999999983/999999999979','999999999961/999999999959','999999999953/999999999929','999999999893/999999999883','999999999863/999999999859','999999999847/999999999821'],probes:[0,7,7,0]}),
]);
