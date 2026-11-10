'use client';
import {useEffect,useState} from 'react';
import {useRealTimePhaseCalculation} from './use-real-time-phase-calculation';
import type {RealTimePhaseInput} from '@/lib/learning-real-time-phase-input';
import {realTimePhaseInputKey,type RealTimePhaseValidatedSuccess} from '@/lib/learning-real-time-phase-protocol';
/** One owned calculation/validation pair at a time. All samples retain their
 * inputs; changed configurations hide the old series during render.
 */
export function useRealTimePhaseSeries(inputs:readonly RealTimePhaseInput[],enabled:boolean){
  if(inputs.length<1||inputs.length>33)throw new RangeError('Series requires1..33inputs');
  const key=JSON.stringify(inputs.map(realTimePhaseInputKey));
  const [saved,setSaved]=useState<{key:string;results:RealTimePhaseValidatedSuccess[]}>({key,results:[]});
  const results=saved.key===key?saved.results:[],index=results.length,complete=index===inputs.length;
  const input=inputs[Math.min(index,inputs.length-1)],expected=realTimePhaseInputKey(input);
  const worker=useRealTimePhaseCalculation(input,enabled&&!complete);
  useEffect(()=>{
    if(saved.key!==key){setSaved({key,results:[]});return;}
    if(enabled&&!complete&&worker.response?.inputKey===expected){const response=worker.response;setSaved(current=>current.key===key&&current.results.length===index?{key,results:[...current.results,response]}:current);}
  },[key,saved.key,index,expected,complete,enabled,worker.response]);
  return {results,complete,status:complete?'ready':!enabled?'idle':worker.status,error:worker.error,retry:worker.retry};
}
