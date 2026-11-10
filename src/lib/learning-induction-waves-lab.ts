import type {FoundationResult} from './learning-foundations';
import {uniformInductionState as field,inductionMovingLoop as loop,inductionLoopInterval as interval,transverseGaussianState as pulse} from './learning-induction-waves';
import {teachingMaxwellPermittivity as eps,teachingMaxwellPermeability as mu} from './learning-maxwell';
import type {StaticVector} from './learning-electro-magnetostatics';
export const inductionWavesKinds=['faraday-induction-fixed-moving-loop','electromagnetic-wave-equation-transverse-initial-data'] as const;
export type InductionWavesKind=typeof inductionWavesKinds[number];
const scale=(x:number|null,f:number)=>x===null||!Number.isFinite(x*f)||x!==0&&x*f===0?null:x*f;
function validate(v:number[],induction:boolean){
 const bounds=induction?[[-4,4],[-4,4],[-4,4],[1,2],[-.25,.25],[0,4],[-2,2],[-2,2],[0,1],[-4,4],[-4,4],[-4,4],[.5,4]]:[[-4,4],[.1,4],[0,1],[-4,4],[-4,4],[0,1],[.5,4],[.25,4]],selector=induction?8:5;
 if(v.length!==bounds.length)throw new RangeError('Wrong induction or pulse control count');
 for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||i===selector&&!Number.isInteger(v[i]))throw new RangeError(`Invalid induction/pulse control ${i}`);
}
function inductionLab(v:number[]){
 const system={magneticOffset:v[0]*1e-6,magneticRate:v[1]*1e-6,magneticAcceleration:v[2]*1e-6,initialWidth:v[3],widthRate:v[4],height:v[5],orientation:(v[8]===1?1:-1) as 1|-1,permittivity:eps*v[12]},position=v.slice(9,12) as StaticVector,time=v[6],startTime=v[7];
 const query=field(system,position,time),path=loop(system,time),balance=interval(system,startTime,time);
 const temporal=Array.from({length:81},(_,i)=>{const t=-2+4*i/80;return{time:t,field:field(system,position,t),path:loop(system,t),balance:interval(system,startTime,t)};});
 return{system,position,time,startTime,query,path,balance,temporal};
}
function pulseLab(v:number[]){
 const system={amplitude:v[0],width:v[1],rightFraction:v[2],magneticMode:v[5]===0?'compatible' as const:'zero-magnetic-negative-control' as const,permittivity:eps*v[6],permeability:mu*v[7]},time=v[3]*1e-9,z=v[4],query=pulse(system,z,time);
 const spatial=Array.from({length:257},(_,i)=>{const x=-4+8*i/256;return{z:x,value:pulse(system,x,time),initial:pulse(system,x,0)};}),temporal=Array.from({length:129},(_,i)=>{const ns=-4+8*i/128;return{ns,value:pulse(system,z,ns*1e-9)};});
 return{system,time,z,query,spatial,temporal};
}
export interface InductionWavesResult extends FoundationResult{kind:InductionWavesKind;electromagnetism:true;unavailable?:'input-representation';inductionWave?:ReturnType<typeof inductionLab>;transversePulse?:ReturnType<typeof pulseLab>;}
export function evaluateInductionWaves(kind:InductionWavesKind,v:number[]):InductionWavesResult{
 if(!(inductionWavesKinds as readonly string[]).includes(kind))throw new RangeError('Unknown induction/pulse lesson');const induction=kind===inductionWavesKinds[0];validate(v,induction);
 if((induction?[0,1,2]:[3]).some(i=>v[i]!==0&&v[i]*(induction?1e-6:1e-9)===0))return{kind,electromagnetism:true,unavailable:'input-representation',curves:[],stats:{'SI observation':null}};
 if(induction){const r=inductionLab(v);return{kind,electromagnetism:true,inductionWave:r,curves:[],stats:{'E y · µV/m':scale(r.query.electricField?.[1]??null,1e6),'total emf · µV':scale(r.path.totalEmf,1e6),'motional emf · µV':scale(r.path.motionalCirculation,1e6),'Delta flux · µWb':scale(r.balance.stableFluxChange,1e6)}};}
 const r=pulseLab(v),q=r.query;return{kind,electromagnetism:true,transversePulse:r,curves:[],stats:{'E x · V/m':q.electricValue,'B y · nT':scale(q.magneticValue,1e9),'displacement · m':q.displacement,'Faraday residual · V/m²':q.faradayResidualY,'compatible cB · V/m':q.stableCB}};
}
