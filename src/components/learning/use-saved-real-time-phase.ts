'use client';
import {useEffect,useState} from 'react';
import {realTimePhaseSteps} from '@/data/real-time-phase-teaching';
import {realTimeLessonLimits} from './real-time-phase-lessons';
import {emptyRealTimeState,parseStoredRealTimeState,readRealTimeState,realTimePhaseStorageKey,type RealTimeSavedState} from '@/lib/learning-real-time-phase-state';
export const realTimeLearningLimits={...realTimeLessonLimits,steps:realTimePhaseSteps.length};
/** Restore before starting a Worker or writing. Store no response or run flag. */
export function useSavedRealTimePhase(){
  const[model,setModel]=useState(()=>emptyRealTimeState(realTimeLearningLimits)),[ready,setReady]=useState(false),[storage,setStorage]=useState<'available'|'corrupt'|'unavailable'>('available');
  useEffect(()=>{let next=emptyRealTimeState(realTimeLearningLimits);try{const raw=localStorage.getItem(realTimePhaseStorageKey);if(raw!==null){const parsed=parseStoredRealTimeState(raw,realTimeLearningLimits);if(parsed)next=parsed;else setStorage('corrupt');}}catch{setStorage('unavailable');}setModel(next);setReady(true);},[]);
  useEffect(()=>{if(ready)try{localStorage.setItem(realTimePhaseStorageKey,JSON.stringify({version:1,model}));}catch{setStorage('unavailable');}},[ready,model]);
  const update=(change:(previous:RealTimeSavedState)=>RealTimeSavedState)=>setModel(previous=>readRealTimeState(change(previous),realTimeLearningLimits)??previous);
  return{model,ready,storage,update};
}
