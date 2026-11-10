'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { FokkerCalculationController, idleFokkerState } from '@/lib/learning-fokker-planck-controller';
import { isFokkerInput, fokkerInputKey, type FokkerInput } from '@/lib/learning-fokker-planck-protocol';
import { createFokkerWorker } from '@/lib/learning-fokker-planck-worker-factory';
/** Inputs may be recreated during render; their canonical key drives work.
 * A ready result is never exposed under newly rendered input parameters, even
 * in the render before the replacement effect runs. No synchronous fallback.
 */
export function useFokkerCalculation(input: FokkerInput | null, enabled = true) {
  const [state, setState] = useState(idleFokkerState);
  const [retryToken, setRetryToken] = useState(0);
  const controller = useRef<FokkerCalculationController | null>(null);
  const inputRef = useRef(input); inputRef.current = input;
  const inputKey = input === null ? null : isFokkerInput(input) ? fokkerInputKey(input) : 'invalid-input';
  useEffect(() => {
    const instance = new FokkerCalculationController(createFokkerWorker, setState);
    controller.current = instance;
    return () => { controller.current = null; instance.dispose(); };
  }, []);
  useEffect(() => {
    const instance = controller.current;
    if (!instance) return;
    if (!enabled || inputRef.current === null) instance.cancel(); else instance.run(inputRef.current);
    return () => { instance.cancel(); };
  }, [inputKey, enabled, retryToken]);
  const retry = useCallback(() => setRetryToken(value => value + 1), []);
  const cancel = useCallback(() => controller.current?.cancel(), []);
  const active = enabled && input !== null, matches = active && state.inputKey === inputKey;
  return { status: !active ? 'idle' as const : state.status !== 'idle' && !matches ? 'pending' as const : state.status,
    phase: matches && state.status === 'pending' ? state.phase : null,
    requestId: matches ? state.requestId : null,
    response: matches && state.status === 'ready' ? state.response : null,
    error: active && state.status === 'error' && matches ? state.error : null,
    retry, cancel };
}
