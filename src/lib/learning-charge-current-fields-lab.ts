import type {FoundationResult} from './learning-foundations';
import type {Vec2} from './learning-simulations';
import {pointChargeField,transportedChargeObservation,gaussianChargeInterval,transportedChargeBalance,directedSurfaceCurrent,type PointCharge,type ChargeVector} from './learning-charge-current-fields';
export const chargeCurrentFieldKinds=['charge-superposition-test-field','charge-continuity-surface-current'] as const;
export type ChargeCurrentFieldKind=typeof chargeCurrentFieldKinds[number];
const scaled=(x:number|null,f:number)=>x===null||!Number.isFinite(x*f)?null:x*f;
function validate(v:number[],bounds:number[][]){if(v.length!==bounds.length)throw new RangeError('Wrong charge/current control count');for(let i=0;i<v.length;i++){const x=v[i];if(!Number.isFinite(x)||x<bounds[i][0]||x>bounds[i][1])throw new RangeError(`Invalid charge/current control ${i}`);}}
function chargeLab(v:number[]){
 validate(v,[[-2,2],[-2,2],[0,4],[-4,4],[-4,4],[-2,2],[1,100]]);
 const sources:PointCharge[]=[{position:[-v[2]/2,0],charge:v[0]*1e-9},{position:[v[2]/2,0],charge:v[1]*1e-9}],observation:ChargeVector=[v[3],v[4]],testCharge=v[5]*1e-9;
 const at=(x:number,y:number)=>pointChargeField(sources,[x,y],testCharge),query=at(...observation);
 const grid=Array.from({length:81},(_,i)=>{const x=i%9-4,y=Math.floor(i/9)-4;return{x,y,value:at(x,y)};});
 const slice=Array.from({length:161},(_,i)=>{const x=-4+i/20,value=at(x,v[4]);const displayExcluded=v[4]===0&&sources.some(s=>s.charge!==0&&Math.abs(x-s.position[0])<.0125);return{x,value,displayExcluded,forceNano:value.force?.map(x=>scaled(x,1e9))??[null,null]};});
 const radial=Array.from({length:151},(_,i)=>{const radius=.25+3.75*i/150,q=pointChargeField([{position:[0,0],charge:sources[0].charge}],[radius,0],testCharge);return{radius,fieldMagnitude:q.fieldMagnitude,potential:q.potential};});
 const sourceFields=query.contributions.map(c=>c.fieldOverK?.map(x=>scaled(x,query.constant))??[null,null]);
 return{sources,observation,testCharge,arrowReference:v[6],query,grid,slice,radial,sourceFields};
}
function currentLab(v:number[]){
 validate(v,[[-2,2],[.1,2],[-2,2],[-2,2],[.1,4],[-2,2],[1e-12,2],[-2,2],[-4,4],[-Math.PI,Math.PI],[0,4]]);
 const system={totalCharge:v[0]*1e-6,sigma:v[1],initialCenter:v[2],velocity:v[3],area:v[4]},left=v[5]-v[6],right=v[5]+v[6],time=v[7],queryX=v[8];
 const query=transportedChargeObservation(system,queryX,time),interval=gaussianChargeInterval(system,left,right,time),balance=transportedChargeBalance(system,left,right,0,time);
 const patchAt=(angle:number)=>query.currentDensity===null?{normal:[Math.cos(angle),Math.sin(angle)]as ChargeVector,densityThroughNormal:null,current:v[10]===0?0:null,area:v[10],status:'numerical-range' as const}:directedSurfaceCurrent([query.currentDensity,0],v[10],angle);
 const surface=patchAt(v[9]);
 const spatialIntervals=Math.max(256,Math.ceil(128/system.sigma));
 const spatial=Array.from({length:spatialIntervals+1},(_,i)=>{const x=-8+16*i/spatialIntervals;return{x,current:transportedChargeObservation(system,x,time),initial:transportedChargeObservation(system,x,0)};});
 const temporalIntervals=Math.max(128,Math.ceil(32*Math.abs(system.velocity)/system.sigma));
 const temporal=Array.from({length:temporalIntervals+1},(_,i)=>{const t=-2+4*i/temporalIntervals;return gaussianChargeInterval(system,left,right,t);});
 const angular=Array.from({length:129},(_,i)=>{const angle=-Math.PI+2*Math.PI*i/128;return{angle,...patchAt(angle)};});
 return{system,left,right,time,queryX,query,interval,balance,surface,normalAngle:v[9],patchArea:v[10],spatial,temporal,angular};
}
export interface ChargeCurrentFieldResult extends FoundationResult{kind:ChargeCurrentFieldKind;electromagnetism:true;unavailable?:'input-representation';charges?:ReturnType<typeof chargeLab>;transport?:ReturnType<typeof currentLab>;}
export function evaluateChargeCurrentFields(kind:ChargeCurrentFieldKind,v:number[]):ChargeCurrentFieldResult{
 const fieldKind=kind===chargeCurrentFieldKinds[0];if(!fieldKind&&kind!==chargeCurrentFieldKinds[1])throw new RangeError('Unknown charge/current lesson');
 validate(v,fieldKind?[[-2,2],[-2,2],[0,4],[-4,4],[-4,4],[-2,2],[1,100]]:[[-2,2],[.1,2],[-2,2],[-2,2],[.1,4],[-2,2],[1e-12,2],[-2,2],[-4,4],[-Math.PI,Math.PI],[0,4]]);
 if((fieldKind?[0,1,5]:[0]).some(i=>v[i]!==0&&v[i]*(fieldKind?1e-9:1e-6)===0)||fieldKind&&v[2]!==0&&v[2]/2===0)return{kind,electromagnetism:true,unavailable:'input-representation',curves:[],stats:{'SI observation':null}};
 if(fieldKind){const r=chargeLab(v),q=r.query;return{kind,electromagnetism:true,charges:r,curves:[r.slice.filter(p=>p.value.field!==null).map(p=>[p.x,p.value.field![0]]as Vec2)],stats:{'E x · N/C':q.field?.[0]??null,'E y · N/C':q.field?.[1]??null,'|E| · N/C':q.fieldMagnitude,'V · V':q.potential,'F x · nN':scaled(q.force?.[0]??null,1e9),'F y · nN':scaled(q.force?.[1]??null,1e9),'test charge · nC':v[5],'source sum · nC':v[0]+v[1],'distance 1 · m':q.contributions[0].distance,'distance 2 · m':q.contributions[1].distance}};}
 if(kind!==chargeCurrentFieldKinds[1])throw new RangeError('Unknown charge/current lesson');const r=currentLab(v),q=r.query,b=r.balance;
 return{kind,electromagnetism:true,transport:r,curves:[],stats:{'Q inside · µC':scaled(r.interval.chargeInside,1e6),'net inflow · µA':scaled(r.interval.netInflow,1e6),'rho query · µC/m³':scaled(q.density,1e6),'J query · µA/m²':scaled(q.currentDensity,1e6),'patch current · µA':scaled(r.surface.current,1e6),'charge change · µC':scaled(b.chargeChange,1e6),'integrated inflow · µC':scaled(b.integratedNetInflow,1e6),'balance residual · C':b.balanceResidual,'mass fraction · 1':r.interval.fraction,'local residual · C/(m³ s)':q.localContinuityResidual}};
}
