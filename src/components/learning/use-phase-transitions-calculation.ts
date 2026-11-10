'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { PhaseCalculationController, idlePhaseState } from '@/lib/learning-phase-transitions-controller';
import { isPhaseInput, phaseInputKey, type PhaseInput } from '@/lib/learning-phase-transitions-protocol';
import { createPhaseWorker } from '@/lib/learning-phase-transitions-worker-factory';
/** Inputs may be recreated during render; their canonical key drives work.
 * A ready result is never exposed under newly rendered input parameters, even
 * in the render before the replacement effect runs. No synchronous fallback.
 */
export function usePhaseCalculation(input: PhaseInput | null, enabled = true) {
  const [state, setState] = useState(idlePhaseState);
  const [retryToken, setRetryToken] = useState(0);
  const controller = useRef<PhaseCalculationController | null>(null);
  const inputRef = useRef(input); inputRef.current = input;
  const inputKey = input === null ? null : isPhaseInput(input) ? phaseInputKey(input) : 'invalid-input';
  useEffect(() => {
    const instance = new PhaseCalculationController(createPhaseWorker, setState);
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
