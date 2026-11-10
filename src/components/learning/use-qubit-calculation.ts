'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { QubitCalculationController, idleQubitState } from '@/lib/learning-qubit-controller';
import { qubitInputKey, type QubitInput } from '@/lib/learning-qubit-input';
import { createQubitWorker, createQubitValidationWorker } from '@/lib/learning-qubit-worker-factory';

/** Render-time attribution hides old results before the replacement effect. */
export function useQubitCalculation(input: QubitInput, enabled: boolean) {
  const [state, setState] = useState(idleQubitState);
  const [retryToken, setRetryToken] = useState(0);
  const controller = useRef<QubitCalculationController | null>(null);
  const latest = useRef(input); latest.current = input;
  const key = qubitInputKey(input);
  useEffect(() => {
    const instance = new QubitCalculationController({ calculate: createQubitWorker, validate: createQubitValidationWorker }, setState);
    controller.current = instance;
    return () => { controller.current = null; instance.dispose(); };
  }, []);
  useEffect(() => {
    const instance = controller.current;
    if (!instance) return;
    if (enabled) instance.run(latest.current); else instance.cancel();
    return () => instance.cancel();
  }, [key, enabled, retryToken]);
  const retry = useCallback(() => setRetryToken(n => n + 1), []);
  const cancel = useCallback(() => controller.current?.cancel(), []);
  const matches = enabled && state.inputKey === key;
  return {
    status: !enabled ? 'idle' : state.status !== 'idle' && !matches ? 'pending' : state.status,
    phase: matches && state.status === 'pending' ? state.phase : null,
    response: matches && state.status === 'ready' ? state.response : null,
    error: matches && state.status === 'error' ? state.error : null,
    retry, cancel,
  };
}
