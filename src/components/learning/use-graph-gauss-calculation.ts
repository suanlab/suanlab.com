'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { GraphGaussCalculationController, idleGraphGaussState } from '@/lib/learning-graph-gauss-controller';
import { graphGaussInputKey, type GraphGaussInput } from '@/lib/learning-graph-gauss-input';
import { createGraphGaussWorker, createGraphGaussValidationWorker } from '@/lib/learning-graph-gauss-worker-factory';

/** Render-time attribution hides old results before the replacement effect. */
export function useGraphGaussCalculation(input: GraphGaussInput, enabled: boolean) {
  const [state, setState] = useState(idleGraphGaussState);
  const [retryToken, setRetryToken] = useState(0);
  const controller = useRef<GraphGaussCalculationController | null>(null);
  const latest = useRef(input); latest.current = input;
  const key = graphGaussInputKey(input);
  const [visible,setVisible]=useState(true);
  useEffect(()=>{
    const update=()=>{const next=document.visibilityState!=='hidden';if(!next)controller.current?.cancel();setVisible(next);};
    update();document.addEventListener('visibilitychange',update);return()=>document.removeEventListener('visibilitychange',update);
  },[]);
  useEffect(() => {
    const instance = new GraphGaussCalculationController({ calculate: createGraphGaussWorker, validate: createGraphGaussValidationWorker }, setState);
    controller.current = instance;
    return () => { controller.current = null; instance.dispose(); };
  }, []);
  useEffect(() => {
    const instance = controller.current;
    if (!instance) return;
    if (enabled && visible && document.visibilityState !== 'hidden') instance.run(latest.current); else instance.cancel();
    return () => instance.cancel();
  }, [key, enabled, visible, retryToken]);
  const retry = useCallback(() => setRetryToken(n => n + 1), []);
  const cancel = useCallback(() => controller.current?.cancel(), []);
  const matches = enabled && visible && state.inputKey === key;
  return {
    status: !enabled || !visible ? 'idle' : state.status !== 'idle' && !matches ? 'pending' : state.status,
    phase: matches && state.status === 'pending' ? state.phase : null,
    response: matches && state.status === 'ready' ? state.response : null,
    error: matches && state.status === 'error' ? state.error : null,
    retry, cancel,
  };
}
