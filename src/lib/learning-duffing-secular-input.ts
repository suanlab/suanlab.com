/** Reader-safe inputs for x″+x+λx³=0 at a nonnegative turning amplitude. */
export interface DuffingSecularInput {lambda:string;amplitude:string;horizon:string;probeFraction:string;}
export function readDuffingSecularInput(value:unknown):Readonly<DuffingSecularInput>|null{
 try{if(!value||typeof value!=='object'||Array.isArray(value))return null;const keys=['lambda','amplitude','horizon','probeFraction']as const,d=Object.getOwnPropertyDescriptors(value);if(Reflect.ownKeys(d).length!==4)return null;const out={}as DuffingSecularInput;
  for(const k of keys){const p=d[k];if(!p||!('value'in p)||typeof p.value!=='string'||!/^(?:0|[1-9]\d{0,11})(?:\/[1-9]\d{0,11})?$/.test(p.value))return null;const[n,den='1']=p.value.split('/'),limit=k==='horizon'?999999999999:k==='probeFraction'?1:16;if(Number(n)>limit*Number(den))return null;out[k]=p.value;}return Object.freeze(out);
 }catch{return null;}
}
export const duffingSecularDefault=Object.freeze({lambda:'1/100',amplitude:'1',horizon:'100',probeFraction:'1'});
export const duffingSecularExamples=Object.freeze([
 {id:'weak-short',input:{...duffingSecularDefault,horizon:'10'}},
 {id:'secular-growth',input:duffingSecularDefault},
 {id:'naive-breakdown',input:{...duffingSecularDefault,horizon:'1000'}},
 {id:'first-frequency-drift',input:{...duffingSecularDefault,horizon:'10000'}},
 {id:'second-frequency-drift',input:{...duffingSecularDefault,horizon:'1000000'}},
 {id:'harmonic',input:{...duffingSecularDefault,lambda:'0'}},
 {id:'equilibrium',input:{...duffingSecularDefault,lambda:'1',amplitude:'0'}},
 {id:'amplitude-matters',input:{...duffingSecularDefault,amplitude:'4'}},
 {id:'small-amplitude',input:{...duffingSecularDefault,amplitude:'1/10',horizon:'10000'}},
 {id:'tiny-coupling-long-time',input:{...duffingSecularDefault,lambda:'1/999999999989',horizon:'999999999989'}},
 {id:'strong-coupling',input:{...duffingSecularDefault,lambda:'1'}},
 {id:'maximum-coupling',input:{...duffingSecularDefault,lambda:'16',amplitude:'16'}},
 {id:'negative-truncated-frequency',input:{...duffingSecularDefault,lambda:'1',amplitude:'4'}},
 {id:'initial-time',input:{...duffingSecularDefault,horizon:'0'}},
 {id:'equivalent-spelling',input:{...duffingSecularDefault,lambda:'2/200'}},
 {id:'fractional-time',input:{...duffingSecularDefault,horizon:'101/7',probeFraction:'1/4'}},
 {id:'maximum-coprime',input:{lambda:'999999999983/62499999999',amplitude:'999999999979/62499999999',horizon:'999999999961',probeFraction:'999999999953/999999999989'}},
].map(c=>Object.freeze({id:c.id,input:Object.freeze(c.input)})));
