import type {FoundationResult} from './learning-foundations';
import {quadraticPotentialObservation as observe,quadraticPotentialPathWork as work,cylinderVectorPotential as vectorPotential,cylinderPotentialRectangle as rectangle,type QuadraticPotential,type CylinderPotential}from './learning-potentials';
import {magnetostaticCylinder,type StaticVector}from './learning-electro-magnetostatics';
export const potentialsKinds=['electrostatic-potential-reference-path-work','magnetic-vector-potential-curl-gauge']as const;
export type PotentialsKind=typeof potentialsKinds[number];
const mul=(x:number|null,f:number)=>f===0?0:x===null||!Number.isFinite(x*f)||x!==0&&x*f===0?null:x*f;
function validate(v:number[],scalar:boolean){const bounds=scalar?[...Array.from({length:9},()=>[-4,4]),[-100,100],...Array.from({length:10},()=>[-4,4])]:[[.25,4],...Array.from({length:12},()=>[-4,4])];if(v.length!==bounds.length)throw new RangeError('Wrong potential control count');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1])throw new RangeError(`Invalid potential control ${i}`);}
function scalarLab(v:number[]){
 const system:QuadraticPotential={hessian:[[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]],linear:v.slice(6,9)as StaticVector,reference:v[9]},start=v.slice(10,13)as StaticVector,end=v.slice(13,16)as StaticVector,control=v.slice(16,19)as StaticVector,charge=v[19]*1e-9;
 const query=work(system,start,end,control,charge),lerp=(a:number,b:number,t:number)=>a+(b-a)*t;
 const samples=Array.from({length:65},(_,i)=>{const t=i/64,straight=start.map((x,j)=>lerp(x,end[j],t))as StaticVector,subControl=start.map((x,j)=>lerp(x,control[j],t))as StaticVector,curve=start.map((x,j)=>lerp(subControl[j],lerp(control[j],end[j],t),t))as StaticVector;
  return{t,straight,curve,straightObservation:observe(system,straight,charge),curveObservation:observe(system,curve,charge),straightWork:work(system,start,straight,start,charge),curveWork:work(system,start,curve,subControl,charge)};
 });
 return{system,start,end,control,charge,query,samples};
}
const n16=[.09501250983763744,.2816035507792589,.4580167776572274,.6178762444026438,.755404408355003,.8656312023878318,.9445750230732326,.9894009349916499],w16=[.1894506104550685,.1826034150449236,.1691565193950025,.1495959888165767,.1246289712555339,.0951585116824928,.0622535239386479,.0271524594117541];
const n8=[.1834346424956498,.525532409916329,.7966664774136267,.9602898564975363],w8=[.362683783378362,.3137066458778873,.2223810344533745,.1012285362903763];
function sum(values:(number|null)[]):number|null{if(values.some(x=>x===null))return null;let value=0,c=0;for(const x of values){const y=x!-c,t=value+y;c=(t-value)-y;value=t;}return Number.isFinite(value)?value:null;}
/** Integrate B from the preceding Ampere model,not A endpoint values. Split at
 * physical interfaces; order differences are diagnostics,not certified bounds. */
export function potentialRectangleFlux(s:CylinderPotential,left:number,right:number,bottom:number,top:number,y:number){
 for(const x of[left,right,bottom,top,y])if(!Number.isFinite(x)||Math.abs(x)>4)throw new RangeError('Flux rectangle outside teaching domain');
 magnetostaticCylinder({radius:s.radius,totalCurrent:s.current,permeability:s.permeability},[0,y,0]);
 const height=top-bottom,area=-(right-left)*height||0;
 if(left===right||height===0)return{flux:0,orderDifference:0,orientedAreaY:0,meanNormalField:null,meanYField:null,normalizationStatus:'zero-area' as const};
 const low=Math.min(left,right),high=Math.max(left,right),orientation=right>left?1:-1,cross=Math.abs(y)<s.radius?Math.sqrt(s.radius*s.radius-y*y):null;
 const breaks=[low,...(cross===null?[]:[-cross,cross]).filter(x=>x>low&&x<high),high].sort((a,b)=>a-b);
 const integrate=(nodes:number[],weights:number[])=>{const terms:(number|null)[]=[];for(let j=1;j<breaks.length;j++){const width=(breaks[j]-breaks[j-1])/8;for(let k=0;k<8;k++){const center=breaks[j-1]+(k+.5)*width;for(let i=0;i<nodes.length;i++)for(const sign of[-1,1]){const x=center+sign*width/2*nodes[i],b=magnetostaticCylinder({radius:s.radius,totalCurrent:s.current,permeability:s.permeability},[x,y,0]).sourceField?.[1]??null;terms.push(mul(mul(mul(b,width/2),weights[i]),-height*orientation));}}}return sum(terms);};
 const flux=integrate(n16,w16),other=integrate(n8,w8);
 const ratio=(denominator:number)=>{if(flux===null||denominator===0)return null;const value=flux/denominator;return !Number.isFinite(value)||flux!==0&&value===0?null:value;};
 const mean=ratio(Math.abs(area));
 return{flux,orderDifference:flux===null||other===null?null:flux-other,orientedAreaY:area,meanNormalField:mean,meanYField:ratio(area),normalizationStatus:area===0?'area-underflow' as const:mean===null?'arithmetic-range' as const:'finite' as const};
}
function vectorLab(v:number[]){
 const system:CylinderPotential={radius:v[0],current:v[1],gaugeQuadratic:v[5]*1e-6,gaugeCubic:v[6]*1e-6,gaugeAxialConstant:v[7]*1e-6},observation=v.slice(2,5)as StaticVector,query=vectorPotential(system,observation),loop=rectangle(system,...v.slice(8,13)as[number,number,number,number,number]),sourceLoop=rectangle({...system,gaugeQuadratic:0,gaugeCubic:0,gaugeAxialConstant:0},...v.slice(8,13)as[number,number,number,number,number]),flux=potentialRectangleFlux(system,...v.slice(8,13)as[number,number,number,number,number]);
 const radii=[...new Set([...Array.from({length:161},(_,i)=>i/40),system.radius,...(query.radialDistance<=4?[query.radialDistance]:[])])].sort((a,b)=>a-b);
 return{system,observation,query,loop,sourceLoop,flux,radial:radii.map(radius=>({radius,value:vectorPotential(system,[radius,0,0])})),rectangleCoordinates:v.slice(8,13)};
}
export interface PotentialsResult extends FoundationResult{kind:PotentialsKind;electromagnetism:true;unavailable?:'input-representation';scalarPotential?:ReturnType<typeof scalarLab>;vectorPotential?:ReturnType<typeof vectorLab>;}
export function evaluatePotentials(kind:PotentialsKind,v:number[]):PotentialsResult{
 if(!(potentialsKinds as readonly string[]).includes(kind))throw new RangeError('Unknown potential lesson');const scalar=kind===potentialsKinds[0];validate(v,scalar);
 if((scalar?[19]:[5,6,7]).some(i=>v[i]!==0&&v[i]*(scalar?1e-9:1e-6)===0))return{kind,electromagnetism:true,unavailable:'input-representation',curves:[],stats:{'SI observation':null}};
 if(scalar){const r=scalarLab(v),q=r.query;return{kind,electromagnetism:true,scalarPotential:r,curves:[],stats:{'Delta V · V':q.potentialDifference,'field work · nJ':mul(q.fieldWorkBezier,1e9),'naive Delta V · V':q.naiveEndpointDifference,'charge · nC':v[19]}};}
 const r=vectorLab(v),q=r.query;return{kind,electromagnetism:true,vectorPotential:r,curves:[],stats:{'A z · µT m':mul(q.vectorPotential?.[2]??null,1e6),'B y · µT':mul(q.magneticField?.[1]??null,1e6),'div A · µT':q.divergenceA*1e6,'closed A integral · µWb':mul(r.loop.circulation,1e6),'magnetic flux · µWb':mul(r.flux.flux,1e6)}};
}
