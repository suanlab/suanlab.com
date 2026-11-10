'use client';
import {useCallback,useEffect,useRef,useState} from 'react';
import {RealTimePhaseCalculationController,idleRealTimePhaseState} from '@/lib/learning-real-time-phase-controller';
import {readRealTimePhaseInput,type RealTimePhaseInput} from '@/lib/learning-real-time-phase-input';
import {realTimePhaseInputKey} from '@/lib/learning-real-time-phase-protocol';
import {createRealTimePhaseWorker,createRealTimePhaseValidationWorker} from '@/lib/learning-real-time-phase-worker-factory';
/** Hide stale results during render,before replacement effects dispatch work. */
export function useRealTimePhaseCalculation(input:RealTimePhaseInput,enabled:boolean) {
  const[state,setState]=useState(idleRealTimePhaseState),[retryToken,setRetryToken]=useState(0),[visible,setVisible]=useState(true);
  const controller=useRef<RealTimePhaseCalculationController|null>(null),latest=useRef(input);latest.current=input;
  const owned=readRealTimePhaseInput(input),key=owned?realTimePhaseInputKey(owned):null;
  useEffect(()=>{
    const update=()=>{const shown=document.visibilityState!=='hidden';if(!shown)controller.current?.cancel();setVisible(shown);};
    update();document.addEventListener('visibilitychange',update);return()=>document.removeEventListener('visibilitychange',update);
  },[]);
  useEffect(()=>{
    const instance=new RealTimePhaseCalculationController({calculate:createRealTimePhaseWorker,validate:createRealTimePhaseValidationWorker},setState);
    controller.current=instance;return()=>{controller.current=null;instance.dispose();};
  },[]);
  useEffect(()=>{
    const instance=controller.current;if(!instance)return;
    if(enabled&&visible&&document.visibilityState!=='hidden')instance.run(latest.current);else instance.cancel();
    return()=>instance.cancel();
  },[key,enabled,visible,retryToken]);
  const retry=useCallback(()=>setRetryToken(n=>n+1),[]),cancel=useCallback(()=>controller.current?.cancel(),[]);
  const matches=enabled&&visible&&state.inputKey===key;
  return{status:!enabled||!visible?'idle':state.status!=='idle'&&!matches?'pending':state.status,
    phase:matches&&state.status==='pending'?state.phase:null,response:matches&&state.status==='ready'?state.response:null,
    error:matches&&state.status==='error'?state.error:null,retry,cancel};
}
