/** Drawing projections and interval labels only; no Fourier or validation engine. */
import type {FieldModesResult,FieldModeRange} from './learning-field-modes';
import {scalarFieldProjection} from './learning-scalar-field-view';
export type FieldModesQuantity='shape'|'position'|'velocity';
export type FieldModesModalQuantity='coordinate'|'velocity'|'quadraticEnergy'|'eigenvalue';
export function fieldModeMidpoint(x:FieldModeRange):number|null{const a=scalarFieldProjection(x.lower),b=scalarFieldProjection(x.upper);if(a===null||b===null)return null;const middle=a/2+b/2;return Number.isFinite(middle)?middle:null;}
export function fieldModeSummary(x:FieldModeRange){if(x.lower===x.upper)return x.lower;const middle=fieldModeMidpoint(x);if(middle===null)return null;return `≈ ${middle.toExponential(5)}`;}
export function fieldModesView(r:FieldModesResult,quantity:FieldModesQuantity){
 const numbers=quantity==='shape'?r.selected.shape.map(fieldModeMidpoint):r.input[quantity].map(scalarFieldProjection);if(numbers.some(x=>x===null))return null;
 const maximum=Math.max(...(numbers as number[]).map(Math.abs)),boundary=r.input.boundary,n=r.sites,denominator=boundary==='dirichlet-zero'?n+1:n,offset=boundary==='dirichlet-zero'?1:boundary==='neumann-zero'?.5:0,points=(numbers as number[]).map((y,site)=>({x:(site+offset)/denominator,y:maximum===0?0:y/maximum,site}));
 const curve=boundary==='periodic'?[...points,{x:1,y:points[0].y,site:-1}]:boundary==='dirichlet-zero'?[{x:0,y:0,site:-1},...points,{x:1,y:0,site:-1}]:points;
 const exactZero=quantity==='shape'?r.selected.shape.every(x=>x.lower==='0'&&x.upper==='0'):r.input[quantity].every(x=>x.split('/')[0]==='0');return{points,curve,maximum,exactZero,quantity:'position' as const};
}
export function fieldModesSeries(r:FieldModesResult,quantity:FieldModesModalQuantity){const values=r.modes.map(m=>fieldModeMidpoint(m[quantity]));if(values.some(x=>x===null))return null;const maximum=Math.max(...(values as number[]).map(Math.abs));return{maximum,points:(values as number[]).map((y,index)=>({index,y:maximum===0?0:y/maximum}))};}
