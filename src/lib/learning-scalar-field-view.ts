/** Reader-only drawing projections. Exact result strings remain the authority. */
import type {ScalarFieldResult} from './learning-scalar-field';
export type ScalarFieldQuantity='position'|'velocity'|'acceleration';
export function scalarFieldProjection(text:string):number|null{
 const [numerator,denominator='1']=text.split('/'),negative=numerator.startsWith('-'),digits=negative?numerator.slice(1):numerator;
 if(!/^\d+$/.test(digits)||!/^\d+$/.test(denominator))return null;
 const ns=digits.slice(0,16),ds=denominator.slice(0,16),value=(negative?-1:1)*Number(ns)/Number(ds)*10**(digits.length-ns.length-denominator.length+ds.length);
 return Number.isFinite(value)&&(value!==0||/^0+$/.test(digits))?value:null;
}
export function scalarFieldView(result:ScalarFieldResult,quantity:ScalarFieldQuantity){
 const values=quantity==='position'?result.input.position:quantity==='velocity'?result.input.velocity:result.acceleration,projected=values.map(scalarFieldProjection);
 if(projected.some(v=>v===null))return null;
 const numbers=projected as number[],maximum=Math.max(...numbers.map(Math.abs)),denominator=result.input.boundary==='dirichlet-zero'?result.sites+1:result.sites,offset=result.input.boundary==='neumann-zero'?.5:result.input.boundary==='dirichlet-zero'?1:0;
 const points=numbers.map((y,i)=>({x:(i+offset)/denominator,y:maximum===0?0:y/maximum,site:i}));
 const curve=result.input.boundary==='periodic'?[...points,{x:1,y:points[0].y,site:-1}]:result.input.boundary==='dirichlet-zero'?[{x:0,y:0,site:-1},...points,{x:1,y:0,site:-1}]:points;
 return {points,curve,maximum,exactZero:values.every(x=>x.split('/')[0]==='0'),quantity};
}
