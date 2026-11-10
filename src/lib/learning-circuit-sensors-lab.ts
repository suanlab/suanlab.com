import type{FoundationResult}from './learning-foundations';
import {rcPlateState,finiteAreaCoilState,coilFieldState,inferFixedCoilRates,type CoilField,type CoilProbe}from './learning-circuit-sensors';
import {teachingMaxwellPermittivity as eps,teachingMaxwellPermeability as mu}from './learning-maxwell';
export const circuitSensorsKinds=['rc-circuit-field-closure-approximation','finite-area-coil-sensor-observability']as const;
export type CircuitSensorsKind=typeof circuitSensorsKinds[number];
function validate(v:number[],first:boolean){const bounds=first?[[1,1e10],[.05,2],[.01,1],[-4,4],[-4,4],[0,16],[0,1],[.5,4],[.5,4]]:Array.from({length:19},(_,i)=>i===6||i===13?[0,2]:i===7?[0,64]:i===16?[0,1]:i===17?[0,2]:[-4,4]);if(v.length!==bounds.length)throw new RangeError('Wrong circuit/sensor control count');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||!first&&(i===7||i===17)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid circuit/sensor control ${i}`);}
function rcLab(v:number[]){const system={resistance:v[0],plateRadius:v[1],gap:v[2],supplyVoltage:v[3],initialVoltage:v[4],permittivity:eps*v[7],permeability:mu*v[8]},theta=v[5],fraction=v[6],query=rcPlateState(system,theta,fraction),temporal=Array.from({length:129},(_,i)=>{const theta=i/8;return{theta,value:rcPlateState(system,theta,fraction)};}),radial=Array.from({length:81},(_,i)=>{const fraction=i/80;return{fraction,value:rcPlateState(system,theta,fraction)};}),resistanceSweep=Array.from({length:81},(_,i)=>{const exponent=i/8;return{exponent,value:rcPlateState({...system,resistance:10**exponent},theta,fraction)};});return{system,theta,fraction,query,temporal,radial,resistanceSweep};}
const sum=(a:number,b:number)=>{const q=a+b;return Number.isFinite(q)?q||0:null;};
function coilLab(v:number[]){
 const system:CoilField={uniformField:v[0]*1e-6,uniformRate:v[1]*1e-6,uniformAcceleration:v[2]*1e-6,curvature:v[3]*1e-6,curvatureRate:v[4]*1e-6,curvatureAcceleration:v[5]*1e-6},probe:CoilProbe={radius:v[6],turns:v[7],center:[v[8],v[9]],velocity:[v[10],v[11]]},time=v[12],query=finiteAreaCoilState(system,probe,time);
 const fixedProbes:CoilProbe[]=[{...probe,velocity:[0,0]},{radius:v[13],turns:v[7],center:[v[14],v[15]],velocity:[0,0]}],calibration=fixedProbes.map(p=>finiteAreaCoilState(system,p,time)),error=v[16]*1e-6,bias=v[18]*1e-6,noise=v[17]===0?[0,0]:v[17]===1?[-error,error]:[error,error];
 const measured=calibration.map((q,i)=>q.emf===null?null:sum(q.emf,noise[i]+(i?bias:0)));
 // Fixed calibration coils stay at their declared initial centers;they do not follow the moving probe.
 const inference=measured.includes(null)?null:inferFixedCoilRates(fixedProbes.map((probe,i)=>({probe,emf:measured[i]!})),1,error);
 const temporal=Array.from({length:81},(_,i)=>{const time=-4+i/10;return{time,value:finiteAreaCoilState(system,probe,time)};});
 const radii=Array.from({length:81},(_,i)=>{const radius=i/40;return{radius,value:finiteAreaCoilState(system,{...probe,radius},time)};});
 const spatial=query.center[1]===null?[]:Array.from({length:81},(_,i)=>{const x=-4+i/10;return{x,value:coilFieldState(system,[x,query.center[1]!,0],time)};});
 const actualField=query.center.includes(null)?null:coilFieldState(system,[query.center[0]!,query.center[1]!,0],time);
 return{system,probe,time,query,fixedProbes,calibration,measured,error,bias,noise,inference,temporal,radii,spatial,actualField,trueUniformRate:query.B0t,trueCurvatureRate:query.gt};
}
export interface CircuitSensorsResult extends FoundationResult{kind:CircuitSensorsKind;electromagnetism:true;unavailable?:'input-representation';rcCircuit?:ReturnType<typeof rcLab>;coilSensor?:ReturnType<typeof coilLab>;}
export function evaluateCircuitSensors(kind:CircuitSensorsKind,v:number[]):CircuitSensorsResult{
 if(!(circuitSensorsKinds as readonly string[]).includes(kind))throw new RangeError('Unknown circuit/sensor lesson');const first=kind===circuitSensorsKinds[0];validate(v,first);
 if(first){const r=rcLab(v),q=r.query;return{kind,electromagnetism:true,rcCircuit:r,curves:[],stats:{'stable voltage change · V':q.stableVoltageChange,'series current · A':q.current,'gap Faraday phi residual · T/s':q.field.faradayPhi,'retardation parameter a/(cRC)':q.retardationParameter}};}
 if([0,1,2,3,4,5,16,18].some(i=>v[i]!==0&&v[i]*1e-6===0))return{kind,electromagnetism:true,unavailable:'input-representation',curves:[],stats:{'SI observation':null}};
 const r=coilLab(v),q=r.query;return{kind,electromagnetism:true,coilSensor:r,curves:[],stats:{'moving-probe emf · V':q.emf,'finite-area transformer correction · V':q.finiteAreaCorrection,'fixed-calibration structural rank':r.inference?.structuralRank??null,'scaled inference condition infinity':r.inference?.conditionInfinity??null}};
}
