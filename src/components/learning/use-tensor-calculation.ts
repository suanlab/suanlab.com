'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { TensorCalculationController, idleTensorState } from '@/lib/learning-tensor-controller';
import { isTensorInput, tensorInputKey, type TensorInput } from '@/lib/learning-tensor-protocol';
import { createTensorWorker } from '@/lib/learning-tensor-worker-factory';
/** Inputs may be recreated during render; their canonical key drives work.
 * A ready result is never exposed under newly rendered input parameters, even
 * in the render before the replacement effect runs. No synchronous fallback.
 */
export function useTensorCalculation(input: TensorInput | null, enabled = true) {
  const [state, setState] = useState(idleTensorState);
  const [retryToken, setRetryToken] = useState(0);
  const controller = useRef<TensorCalculationController | null>(null);
  const inputRef = useRef(input); inputRef.current = input;
  const inputKey = input === null ? null : isTensorInput(input) ? tensorInputKey(input) : 'invalid-input';
  useEffect(() => {
    const instance = new TensorCalculationController(createTensorWorker, setState);
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
