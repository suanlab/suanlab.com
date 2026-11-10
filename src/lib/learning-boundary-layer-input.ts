/** Reader-safe input for εu″+au′=f, u(0)=A, u(1)=B. No numerical engine. */
export interface BoundaryLayerInput { epsilon:string; drift:string; forcing:string; left:string; right:string; probe:string; panels:number; }
export function readBoundaryLayerInput(value:unknown):Readonly<BoundaryLayerInput>|null{
 try{
  if(!value||typeof value!=='object'||Array.isArray(value))return null;
  const fields=['epsilon','drift','forcing','left','right','probe'] as const,d=Object.getOwnPropertyDescriptors(value);
  if(Reflect.ownKeys(d).length!==7||![...fields,'panels'].every(k=>d[k]&&'value'in d[k]))return null;
  const output={} as BoundaryLayerInput;
  for(const k of fields){const v=d[k].value;if(typeof v!=='string'||! /^-?(?:0|[1-9]\d{0,11})(?:\/[1-9]\d{0,11})?$/.test(v)||/^-0(?:\/|$)/.test(v))return null;
   const[n,den='1']=v.split('/'),numerator=Number(n),denominator=Number(den),bound=k==='probe'?1:16;
   if(Math.abs(numerator)>bound*denominator||(k==='epsilon'&&numerator<=0)||(k==='probe'&&numerator<0))return null;
   output[k]=v;
  }
  const panels=d.panels.value;if(!Number.isInteger(panels)||panels<4||panels>256)return null;output.panels=panels;return Object.freeze(output);
 }catch{return null;}
}
export const boundaryLayerDefault=Object.freeze({epsilon:'1/10',drift:'1',forcing:'0',left:'0',right:'1',probe:'1/20',panels:32});
export const boundaryLayerExamples=Object.freeze([
 {id:'left-layer',input:boundaryLayerDefault},
 {id:'thinner-left-layer',input:{...boundaryLayerDefault,epsilon:'1/100'}},
 {id:'right-layer',input:{...boundaryLayerDefault,drift:'-1',probe:'19/20'}},
 {id:'compatible-boundaries',input:{...boundaryLayerDefault,forcing:'1'}},
 {id:'forced-layer',input:{...boundaryLayerDefault,forcing:'1',right:'0'}},
 {id:'negative-mismatch',input:{...boundaryLayerDefault,left:'2',right:'0'}},
 {id:'no-thin-layer',input:{...boundaryLayerDefault,drift:'1/1000'}},
 {id:'zero-drift-forced',input:{...boundaryLayerDefault,drift:'0',forcing:'1'}},
 {id:'zero-drift-unforced',input:{...boundaryLayerDefault,drift:'0'}},
 {id:'tiny-diffusion',input:{...boundaryLayerDefault,epsilon:'1/999999999989'}},
 {id:'tiny-nonzero-drift',input:{...boundaryLayerDefault,drift:'1/999999999989'}},
 {id:'large-diffusion',input:{...boundaryLayerDefault,epsilon:'16'}},
 {id:'maximum-drift',input:{...boundaryLayerDefault,epsilon:'1/100',drift:'16'}},
 {id:'equivalent-spelling',input:{...boundaryLayerDefault,epsilon:'2/20'}},
 {id:'coarse-mesh',input:{...boundaryLayerDefault,epsilon:'1/100',panels:4}},
 {id:'fine-mesh',input:{...boundaryLayerDefault,epsilon:'1/100',panels:256}},
 {id:'maximum-coprime',input:{epsilon:'1/999999999989',drift:'-999999999983/62499999999',forcing:'999999999979/62499999999',left:'-999999999961/62499999999',right:'999999999959/62499999999',probe:'999999999953/999999999989',panels:256}},
].map(c=>Object.freeze({id:c.id,input:Object.freeze(c.input)})));
