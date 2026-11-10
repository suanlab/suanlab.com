/** Reader-safe inputs for a finite classical Gaussian field; no arithmetic engine. */
export interface GaussianFieldInput {sites:number;boundary:'open'|'periodic'|'dirichlet';constraint:'none'|'mean-zero';massSquared:string;stiffness:string;temperature:string;source:string[];probes:[number,number,number,number];}
const fields=['sites','boundary','constraint','massSquared','stiffness','temperature','source','probes']as const;
function ownedArray(value:unknown,length:number):unknown[]|null{
 if(!Array.isArray(value))return null;const d=Object.getOwnPropertyDescriptors(value) as unknown as Record<PropertyKey,PropertyDescriptor>;if(Reflect.ownKeys(d).length!==length+1||!d.length||!('value'in d.length)||d.length.value!==length)return null;const result=[];for(let i=0;i<length;i++){const item=d[String(i)];if(!item||!('value'in item))return null;result.push(item.value);}return result;
}
function fraction(value:unknown,signed:boolean,positive=false):value is string{
 if(typeof value!=='string'||!(signed?/^-?(?:0|[1-9]\d{0,11})(?:\/[1-9]\d{0,11})?$/:/^(?:0|[1-9]\d{0,11})(?:\/[1-9]\d{0,11})?$/).test(value)||/^-0(?:\/|$)/.test(value))return false;
 const[n,d='1']=value.split('/');return Math.abs(Number(n))<=16*Number(d)&&(!positive||Number(n)>0);
}
export function readGaussianFieldInput(value:unknown):Readonly<GaussianFieldInput>|null{
 try{if(!value||typeof value!=='object'||Array.isArray(value))return null;const d=Object.getOwnPropertyDescriptors(value);if(Reflect.ownKeys(d).length!==fields.length||fields.some(k=>!d[k]||!('value'in d[k])))return null;const own=Object.fromEntries(fields.map(k=>[k,d[k].value]));const n=own.sites;
 if(!Number.isInteger(n)||n<3||n>8||!['open','periodic','dirichlet'].includes(own.boundary)||!['none','mean-zero'].includes(own.constraint))return null;
 if(!fraction(own.massSquared,false)||!fraction(own.stiffness,false)||!fraction(own.temperature,false,true))return null;
 const massZero=/^0(?:\/|$)/.test(own.massSquared),stiffZero=/^0(?:\/|$)/.test(own.stiffness);if(massZero&&(stiffZero||(own.boundary!=='dirichlet'&&own.constraint==='none')))return null;
 const source=ownedArray(own.source,n),probes=ownedArray(own.probes,4);if(!source||!source.every(s=>fraction(s,true))||!probes||!probes.every(i=>Number.isInteger(i)&&Number(i)>=0&&Number(i)<n))return null;
 return Object.freeze({sites:n,boundary:own.boundary,constraint:own.constraint,massSquared:own.massSquared,stiffness:own.stiffness,temperature:own.temperature,source:Object.freeze(source.slice())as unknown as string[],probes:Object.freeze(probes.slice())as unknown as GaussianFieldInput['probes']});
 }catch{return null;}
}
export const gaussianFieldDefault:Readonly<GaussianFieldInput>=Object.freeze({sites:4,boundary:'periodic',constraint:'none',massSquared:'1',stiffness:'1',temperature:'1',source:Object.freeze(['1','0','0','0'])as unknown as string[],probes:Object.freeze([0,1,2,3])as unknown as GaussianFieldInput['probes']});
const example=(id:string,patch:Partial<GaussianFieldInput>)=>{const value=readGaussianFieldInput({...gaussianFieldDefault,...patch});if(!value)throw Error('Invalid Gaussian preparation '+id);return Object.freeze({id,input:value});};
export const gaussianFieldExamples=Object.freeze([
 example('localized-source',{}),example('zero-source',{source:['0','0','0','0']}),example('uniform-source',{source:['1','1','1','1']}),example('hotter',{temperature:'2'}),example('independent-sites',{stiffness:'0'}),example('open-boundary',{boundary:'open'}),example('dirichlet-boundary',{boundary:'dirichlet'}),
 example('massless-dirichlet',{sites:3,boundary:'dirichlet',massSquared:'0',source:['1','0','0'],probes:[0,1,1,2]}),example('massless-periodic-constrained',{massSquared:'0',constraint:'mean-zero'}),example('massless-open-constrained',{massSquared:'0',boundary:'open',constraint:'mean-zero'}),example('massive-constrained',{constraint:'mean-zero'}),example('dirichlet-constrained',{sites:3,boundary:'dirichlet',constraint:'mean-zero',massSquared:'0',source:['1','0','0'],probes:[0,1,1,2]}),example('shifted-constrained-source',{constraint:'mean-zero',massSquared:'0',source:['3','2','2','2']}),
 example('soft-mass',{sites:8,massSquared:'1/999999999989',source:['1','0','0','0','0','0','0','0'],probes:[0,7,0,7]}),example('stiff-coupling',{stiffness:'16',probes:[0,0,0,0]}),example('equivalent-spelling',{massSquared:'2/2',stiffness:'3/3',temperature:'4/4'}),example('maximum-coprime',{sites:8,massSquared:'999999999983/62499999999',stiffness:'999999999979/62499999999',temperature:'999999999961/62499999999',source:['16','-16','16','-16','16','-16','16','-16'],probes:[0,7,7,0]}),
]);
