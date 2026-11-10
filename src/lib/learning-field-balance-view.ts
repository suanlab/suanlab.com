/** Approximate drawing coordinates only; exact strings remain authoritative. */
import type {FieldBalanceResult} from './learning-field-balance';
import {scalarFieldProjection} from './learning-scalar-field-view';
export type FieldBalanceQuantity='energy'|'energyDerivative'|'shiftSource';
export function fieldBalanceView(result:FieldBalanceResult,quantity:FieldBalanceQuantity){
 const values=result.cells.map(c=>c[quantity]),numbers=values.map(scalarFieldProjection);if(numbers.some(v=>v===null))return null;
 const maximum=Math.max(...(numbers as number[]).map(Math.abs)),boundary=result.input.field.boundary,n=result.sites,denominator=boundary==='dirichlet-zero'?n+1:n,offset=boundary==='dirichlet-zero'?1:boundary==='neumann-zero'?.5:0;
 const points=(numbers as number[]).map((value,site)=>({x:(site+offset)/denominator,y:maximum===0?0:value/maximum,site}));
 return{points,curve:boundary==='periodic'?[...points,{x:1,y:points[0].y,site:-1}]:points,maximum,exactZero:values.every(x=>x==='0'),quantity:'position' as const};
}
