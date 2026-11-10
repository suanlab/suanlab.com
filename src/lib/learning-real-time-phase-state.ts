/** Reader-safe drafts and persisted preferences. Never store numerical receipts. */
import {readRealTimePhaseInput,realTimeChainDefault,type RealTimePhaseInput} from './learning-real-time-phase-input';
import {realTimePhaseData} from './learning-real-time-phase-protocol';
import {emptyRealTimeLesson,readRealTimeLesson,type RealTimeLessonState,type RealTimeLessonLimits} from './learning-real-time-phase-lesson-state';
import {defaultRealTimeMatrixDisplay,type RealTimeMatrixDisplay} from './learning-real-time-phase-matrix-view';
export interface RealTimeLearningLimits extends RealTimeLessonLimits {steps:number;}
export interface RealTimeDraft {readonly mode:RealTimePhaseInput['mode'];readonly fields:Readonly<Record<string,string>>;}
export interface RealTimeWaveDisplay {mode:number;shift:number;sample:number;comparisonHbar:string;comparisonLog:boolean;foldQuantity:'scaled'|'raw'|'log';}
export interface RealTimeSavedMatrix extends RealTimeMatrixDisplay {tableOpen:boolean;flat:boolean;}
export interface RealTimeSavedState {readonly input:RealTimePhaseInput;readonly draft:RealTimeDraft;readonly step:number;readonly lesson:RealTimeLessonState;readonly matrix:Readonly<RealTimeSavedMatrix>;readonly wave:Readonly<RealTimeWaveDisplay>;}
export const realTimePhaseStorageKey='suanlab-real-time-phase-v1';
export const defaultRealTimeWave:Readonly<RealTimeWaveDisplay>=Object.freeze({mode:1,shift:0,sample:32,comparisonHbar:'',comparisonLog:false,foldQuantity:'scaled'});
export const defaultRealTimeSavedMatrix:Readonly<RealTimeSavedMatrix>=Object.freeze({...defaultRealTimeMatrixDisplay,tableOpen:false,flat:true});
const fields={chain:['intervals','time','mass','hbar','omegaSquared','start','end','damping','probe'],packet:['intervals','time','mass','hbar','omegaSquared','position','momentum','chirp','width','widthMode'],fold:['control','hbar']} as const;
const integer=(value:unknown,min:number,max:number):value is number=>typeof value==='number'&&Number.isInteger(value)&&!Object.is(value,-0)&&value>=min&&value<=max;
export function realTimeDraft(input:RealTimePhaseInput):RealTimeDraft {return Object.freeze({mode:input.mode,fields:Object.freeze(Object.fromEntries(Object.entries(input).filter(([key])=>key!=='mode').map(([key,value])=>[key,String(value)])))});}
export function readRealTimeDraft(value:unknown):RealTimeDraft|null {
  const d=realTimePhaseData(value,['mode','fields']);if(!d||!['chain','packet','fold'].includes(d.mode as string))return null;
  const mode=d.mode as RealTimePhaseInput['mode'],f=realTimePhaseData(d.fields,fields[mode]);
  if(!f||Object.values(f).some(v=>typeof v!=='string'||v.length>32))return null;
  return Object.freeze({mode,fields:Object.freeze({...f} as Record<string,string>)});
}
export function parseRealTimeDraft(value:unknown):RealTimePhaseInput|null {
  const draft=readRealTimeDraft(value);if(!draft)return null;
  const input:Record<string,unknown>={mode:draft.mode,...draft.fields};
  for(const key of draft.mode==='chain'?['intervals','probe']:draft.mode==='packet'?['intervals']:[]){if(!/^(?:0|[1-9]\d{0,2})$/.test(draft.fields[key]))return null;input[key]=Number(draft.fields[key]);}
  return readRealTimePhaseInput(input);
}
export function readRealTimeWave(value:unknown):Readonly<RealTimeWaveDisplay>|null {
  const d=realTimePhaseData(value,['mode','shift','sample','comparisonHbar','comparisonLog','foldQuantity']);
  if(!d||!integer(d.mode,1,31)||typeof d.shift!=='number'||!Number.isFinite(d.shift)||d.shift< -2||d.shift>2||!Number.isInteger(d.shift*4)||!integer(d.sample,0,64)||!['','1/2','1/8','1/32'].includes(d.comparisonHbar as string)||typeof d.comparisonLog!=='boolean'||!['scaled','raw','log'].includes(d.foldQuantity as string))return null;
  return Object.freeze({...d}) as unknown as Readonly<RealTimeWaveDisplay>;
}
export function readRealTimeMatrix(value:unknown):Readonly<RealTimeSavedMatrix>|null {
  const d=realTimePhaseData(value,['kind','component','scale','selectedCell','tableOpen','flat']);
  if(!d||!['hessian','inverse','formal-connected','regulated-connected','map','initial-covariance','final-covariance'].includes(d.kind as string)||!['real','imaginary'].includes(d.component as string)||!['linear','signed-log'].includes(d.scale as string)||!integer(d.selectedCell,0,960)||typeof d.tableOpen!=='boolean'||typeof d.flat!=='boolean')return null;
  return Object.freeze({...d}) as unknown as Readonly<RealTimeSavedMatrix>;
}
export function emptyRealTimeState(limits:RealTimeLearningLimits,input:RealTimePhaseInput=realTimeChainDefault):RealTimeSavedState {return Object.freeze({input:readRealTimePhaseInput(input)!,draft:realTimeDraft(input),step:0,lesson:emptyRealTimeLesson(limits),matrix:defaultRealTimeSavedMatrix,wave:defaultRealTimeWave});}
export function readRealTimeState(value:unknown,limits:RealTimeLearningLimits):RealTimeSavedState|null {
  const d=realTimePhaseData(value,['input','draft','step','lesson','matrix','wave']);if(!d||!integer(d.step,0,limits.steps-1))return null;
  const input=readRealTimePhaseInput(d.input),draft=readRealTimeDraft(d.draft),lesson=readRealTimeLesson(d.lesson,limits),matrix=readRealTimeMatrix(d.matrix),wave=readRealTimeWave(d.wave);
  return input&&draft&&lesson&&matrix&&wave?Object.freeze({input,draft,step:d.step,lesson,matrix,wave}):null;
}
export function parseStoredRealTimeState(raw:string,limits:RealTimeLearningLimits):RealTimeSavedState|null {
  try{if(typeof raw!=='string'||raw.length>100000)return null;const d=realTimePhaseData(JSON.parse(raw),['version','model']);return d?.version===1?readRealTimeState(d.model,limits):null;}catch{return null;}
}
