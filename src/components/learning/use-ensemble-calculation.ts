'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { EnsembleCalculationController, idleEnsembleState } from '@/lib/learning-ensemble-controller';
import { isEnsembleInput, ensembleInputKey, type EnsembleInput } from '@/lib/learning-ensemble-protocol';
import { createEnsembleWorker } from '@/lib/learning-ensemble-worker-factory';
/** Inputs may be recreated during render; their canonical key drives work.
 * A ready result is never exposed under newly rendered input parameters, even
 * in the render before the replacement effect runs. No synchronous fallback.
 */
export function useEnsembleCalculation(input: EnsembleInput | null, enabled = true) {
  const [state, setState] = useState(idleEnsembleState);
  const [retryToken, setRetryToken] = useState(0);
  const controller = useRef<EnsembleCalculationController | null>(null);
  const inputRef = useRef(input); inputRef.current = input;
  const inputKey = input === null ? null : isEnsembleInput(input) ? ensembleInputKey(input) : 'invalid-input';
  useEffect(() => {
    const instance = new EnsembleCalculationController(createEnsembleWorker, setState);
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
