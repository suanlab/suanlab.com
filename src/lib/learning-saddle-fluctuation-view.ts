/** Reader-only projections/state. Exact certificates stay in the Workers.
 * Decimal plots show interval midpoints or explicit analytic approximations.
 */
import {readSaddleFluctuationInput,type SaddleFluctuationInput} from './learning-saddle-fluctuation-input';
import type {SaddleFluctuationResult,SaddleRange} from './learning-saddle-fluctuation';
import {scalarFieldProjection as project} from './learning-scalar-field-view';
export type SaddleChainResult=Extract<SaddleFluctuationResult,{kind:'chain'}>;
export type SaddleWellResult=Extract<SaddleFluctuationResult,{kind:'well'}>;
export type SaddleMatrixKind='covariance'|'inverse'|'hessian';
export interface SaddleFluctuationDisplay {kind:SaddleMatrixKind;scale:'linear'|'signed-log';selectedTime:number;selectedCell:number;selectedWell:'left'|'right';}
export interface SaddleFluctuationLearning {step:number;experiment:number;explanationOpen:boolean;answers:readonly number[];}
export const saddleFluctuationDefaultLearning:Readonly<SaddleFluctuationLearning>=Object.freeze({step:0,experiment:0,explanationOpen:false,answers:Object.freeze(Array(8).fill(-1) as number[])});
export const saddleFluctuationDefaultDisplay:Readonly<SaddleFluctuationDisplay>=Object.freeze({kind:'covariance',scale:'linear',selectedTime:1,selectedCell:4,selectedWell:'right'});
function fields(value:unknown,keys:readonly string[]):Record<string,unknown>|null {
 try{
  if(!value||typeof value!=='object'||Array.isArray(value))return null;
  const d=Object.getOwnPropertyDescriptors(value);
  if(Reflect.ownKeys(d).length!==keys.length||keys.some(k=>!d[k]||!Object.hasOwn(d[k],'value')))return null;
  return Object.fromEntries(keys.map(k=>[k,d[k].value]));
 }catch{return null;}
}
export function readSaddleFluctuationLearning(value:unknown):Readonly<SaddleFluctuationLearning>|null {
 const d=fields(value,['step','experiment','explanationOpen','answers']);
 if(!d||!Number.isInteger(d.step)||Number(d.step)<0||Number(d.step)>=10||!Number.isInteger(d.experiment)||Number(d.experiment)<0||Number(d.experiment)>=7||typeof d.explanationOpen!=='boolean'||!Array.isArray(d.answers))return null;
 try{
  const descriptors=Object.getOwnPropertyDescriptors(d.answers) as unknown as Record<string,PropertyDescriptor>,answers:number[]=[];
  if(Reflect.ownKeys(descriptors).length!==9||descriptors.length?.value!==8)return null;
  for(let i=0;i<8;i++){const v=descriptors[String(i)];if(!v||!Object.hasOwn(v,'value')||!Number.isInteger(v.value)||v.value< -1||v.value>2)return null;answers.push(v.value);}
  return Object.freeze({step:Number(d.step),experiment:Number(d.experiment),explanationOpen:d.explanationOpen,answers:Object.freeze(answers)});
 }catch{return null;}
}
export const saddleMatrixSize=(input:SaddleFluctuationInput,kind:SaddleMatrixKind)=>input.mode==='well'?1:input.intervals+(kind==='hessian'?-1:1);
export function saddleFluctuationAlignDisplay(input:SaddleFluctuationInput,display:SaddleFluctuationDisplay):SaddleFluctuationDisplay {
 if(input.mode==='well')return {...display,selectedTime:0,selectedCell:0};
 const n=saddleMatrixSize(input,display.kind),i=display.kind==='hessian'?Math.max(0,Math.min(n-1,input.probe-1)):input.probe;
 return {...display,selectedTime:input.probe,selectedCell:i*(n+1)};
}
export function saddleFluctuationChangeMatrix(input:SaddleFluctuationInput,display:SaddleFluctuationDisplay,kind:SaddleMatrixKind):SaddleFluctuationDisplay {
 const oldN=saddleMatrixSize(input,display.kind),n=saddleMatrixSize(input,kind),offset=display.kind==='hessian'?1:0,nextOffset=kind==='hessian'?1:0;
 const i=Math.max(0,Math.min(n-1,Math.floor(display.selectedCell/oldN)+offset-nextOffset)),j=Math.max(0,Math.min(n-1,display.selectedCell%oldN+offset-nextOffset));
 return {...display,kind,selectedCell:i*n+j};
}
export function readSaddleFluctuationDisplay(value:unknown,input:SaddleFluctuationInput):Readonly<SaddleFluctuationDisplay>|null {
 const d=fields(value,['kind','scale','selectedTime','selectedCell','selectedWell']);
 if(!d||typeof d.kind!=='string'||typeof d.scale!=='string'||typeof d.selectedWell!=='string'||!['covariance','inverse','hessian'].includes(d.kind)||!['linear','signed-log'].includes(d.scale)||!['left','right'].includes(d.selectedWell))return null;
 const kind=d.kind as SaddleMatrixKind,n=saddleMatrixSize(input,kind),last=input.mode==='chain'?input.intervals:0;
 if(!Number.isInteger(d.selectedTime)||Number(d.selectedTime)<0||Number(d.selectedTime)>last||!Number.isInteger(d.selectedCell)||Number(d.selectedCell)<0||Number(d.selectedCell)>=n*n)return null;
 return Object.freeze({kind,scale:d.scale as 'linear'|'signed-log',selectedTime:Number(d.selectedTime),selectedCell:Number(d.selectedCell),selectedWell:d.selectedWell as 'left'|'right'});
}
export type SaddleFluctuationDraft={mode:'chain';intervals:string;probe:string;time:string;mass:string;hbar:string;omegaSquared:string;quartic:string;start:string;end:string}|{mode:'well';hbar:string;kappa:string};
export function saddleFluctuationDraft(input:SaddleFluctuationInput):SaddleFluctuationDraft {
 return input.mode==='chain'?{...input,intervals:String(input.intervals),probe:String(input.probe)}:{...input};
}
export function parseSaddleFluctuationDraft(draft:SaddleFluctuationDraft):SaddleFluctuationInput|null {
 if(draft.mode==='well')return readSaddleFluctuationInput(draft);
 if(!/^(?:[2-9]|[12]\d|3[0-2])$/.test(draft.intervals)||!/^(?:0|[1-9]\d?)$/.test(draft.probe))return null;
 return readSaddleFluctuationInput({...draft,intervals:Number(draft.intervals),probe:Number(draft.probe)});
}
export const saddleProjection=project;
export function saddleBandProjection(value:SaddleRange):number|null {
 const a=project(value.lower),b=project(value.upper);return a===null||b===null?null:a/2+b/2;
}
export function saddleFluctuationMatrixView(result:SaddleChainResult,display:Pick<SaddleFluctuationDisplay,'kind'|'scale'|'selectedCell'>) {
 const source=display.kind==='hessian'?result.gaussian.hessian:display.kind==='inverse'?result.gaussian.inverseCurvature:result.gaussian.covariance;
 const n=source.length,selected=Math.min(n*n-1,Math.max(0,display.selectedCell)),row=Math.floor(selected/n),column=selected%n,projections=source.map(r=>r.map(saddleBandProjection));
 const projectionAvailable=projections.every(r=>r.every(x=>x!==null)),values=projections.map(r=>r.map(x=>x??0));
 const transform=(x:number)=>display.scale==='signed-log'?Math.sign(x)*Math.log1p(Math.abs(x))/Math.LN10:x;
 const heights=values.map(r=>r.map(transform)),maximum=Math.max(...heights.flat().map(Math.abs))||1;
 const unresolved=source.flat().filter(v=>v.lower!==v.upper&&(v.lower==='0'||v.upper==='0')).length;
 return {n,selected,row,column,source,values,heights,maximum,projectionAvailable,unresolved,kind:display.kind,scale:display.scale,labels:source.map((_,i)=>i+(display.kind==='hessian'?1:0)),points:heights.flatMap((r,i)=>r.map((value,j)=>({x:n===1?0:6*i/(n-1)-3,z:n===1?0:6*j/(n-1)-3,y:2.5*value/maximum,value,selected:i===row&&j===column})))};
}
export type SaddleFluctuationMatrixView=ReturnType<typeof saddleFluctuationMatrixView>;
export function saddleFluctuationPathView(result:SaddleChainResult) {
 const input=result.input,a=project(input.start),b=project(input.end),tau=project(input.time),q=result.stationary.position.map(saddleBandProjection),variances=result.gaussian.covariance.map((r,i)=>saddleBandProjection(r[i]));
 if(a===null||b===null||tau===null||q.some(x=>x===null)||variances.some(x=>x===null||x<0))return null;
 return {q:q as number[],sigma:variances.map(x=>Math.sqrt(x!)),times:q.map((_,i)=>tau*i/input.intervals),straight:q.map((_,i)=>a+(b-a)*i/input.intervals)};
}
/** One-coordinate slice at the mode, not a marginal over the other variables.
 * u uses the smaller of the Gaussian and quartic characteristic widths, so
 * a flat quadratic approximation cannot hide a narrow quartic profile.
 */
export function saddleFluctuationSlice(result:SaddleChainResult,selectedTime:number) {
 if(!Number.isInteger(selectedTime)||selectedTime<=0||selectedTime>=result.input.intervals)return null;
 const i=selectedTime,curvature=saddleBandProjection(result.gaussian.hessian[i-1][i-1]),q=saddleBandProjection(result.stationary.position[i]),h=project(result.input.hbar),delta=project(result.timeStep),lambda=project(result.input.quartic),gradient=project(result.stationary.gradient[i-1]);
 if([curvature,q,h,delta,lambda,gradient].some(x=>x===null)||curvature!<=0)return null;
 const gaussianWidth=Math.sqrt(h!/curvature!),quarticWidth=lambda!>0?(h!/(delta!*lambda!))**.25:Infinity,width=Math.min(gaussianWidth,quarticWidth);
 if(!Number.isFinite(width)||width<=0)return null;
 const rows=Array.from({length:161},(_,j)=>{
  const u=-4+8*j/160,dx=width*u,quadratic=curvature!*dx*dx/(2*h!),cubic=delta!*lambda!*q!*dx*dx*dx/h!,quartic=delta!*lambda!*dx**4/(4*h!),linear=gradient!*dx/h!;
  const cost=quadratic+cubic+quartic+linear,weight=Math.exp(-cost),gaussian=Math.exp(-quadratic);
  return {u,x:q!+dx,weight,gaussian};
 });
 return rows.every(r=>Number.isFinite(r.weight))?{i,center:q!,width,gaussianWidth,rows}:null;
}
/** Relative scalar well weights, evaluated analytically for drawing only.
 * No numerical quadrature, probability normalization or tunnelling model.
 */
export function saddleFluctuationWellView(result:SaddleWellResult,selectedWell:'left'|'right') {
 const k=project(result.input.kappa),h=project(result.input.hbar);
 if(k===null||h===null)return null;
 const center=k>0?(selectedWell==='left'?-1:1)*Math.sqrt(k):0,curvature=k>0?2*k:-k,criticalWidth=h**.25,gaussianWidth=k===0?null:Math.sqrt(h/curvature),width=gaussianWidth===null?criticalWidth:Math.min(gaussianWidth,criticalWidth),extent=Math.max(Math.sqrt(Math.max(0,k))*1.7,h**.25*2.5);
 const potential=Array.from({length:181},(_,j)=>{
  const x=extent*(-1+2*j/180),cost=k>0?(x*x-k)**2/4:x**4/4-k*x*x/2;
  return {x,cost};
 });
 const local=Array.from({length:161},(_,j)=>{
  const u=-4+8*j/160,dx=width*u,x=center+dx,cost=k>0?(2*center*dx+dx*dx)**2/4:x**4/4-k*x*x/2;
  return {u,x,weight:Math.exp(-cost/h),gaussian:k===0?null:Math.exp(-curvature*dx*dx/(2*h))};
 });
 return {center,curvature,width,gaussianWidth,criticalWidth,extent,potential,local,degenerate:k===0,double:k>0};
}
/** Fixed-N first approximation/bounds as functions of hbar. Floating diagnostic
 * curves are separate from the current input's certified ratio endpoints.
 */
export function saddleFluctuationCorrectionView(result:SaddleChainResult) {
 const c1=saddleBandProjection(result.correction.first),m1=saddleBandProjection(result.correction.gaussianFourthSum),m2=saddleBandProjection(result.correction.gaussianFourthSumSquared),delta=project(result.timeStep),lambda=project(result.input.quartic),h=project(result.input.hbar);
 if([c1,m1,m2,delta,lambda,h].some(x=>x===null))return null;
 const coefficient=delta!*lambda!/4;
 return {hbar:h!,rows:Array.from({length:61},(_,j)=>{const hbar=h!*j/60,epsilon=hbar*coefficient;return {hbar,first:1+hbar*c1!,firstContribution:hbar*c1!,lowerDifference:result.ratio.status==='not-applicable-shifted-cubic'?null:Math.max(-1,-epsilon*m1!),upperDifference:result.ratio.status==='not-applicable-shifted-cubic'?null:Math.min(0,-epsilon*m1!+epsilon**2*m2!/2),lower:result.ratio.status==='not-applicable-shifted-cubic'?null:Math.max(0,1-epsilon*m1!),upper:result.ratio.status==='not-applicable-shifted-cubic'?null:Math.min(1,1-epsilon*m1!+epsilon**2*m2!/2)};})};
}
