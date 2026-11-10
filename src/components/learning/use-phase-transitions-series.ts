'use client';
import { useEffect, useRef, useState } from 'react';
import { PhaseCalculationController } from '@/lib/learning-phase-transitions-controller';
import { createPhaseWorker } from '@/lib/learning-phase-transitions-worker-factory';
import { isPhaseInput, phaseInputKey, type PhaseInput, type PhaseSuccess } from '@/lib/learning-phase-transitions-protocol';
/** Calculate fixed samples sequentially, always in workers. Never expose
 * another step's responses while its replacement effect is being installed. */
export function usePhaseSeries(inputs: readonly PhaseInput[]) {
  const key = JSON.stringify(inputs.map(v => isPhaseInput(v) ? phaseInputKey(v) : 'invalid-input')), ref = useRef(inputs); ref.current = inputs;
  const [retry, setRetry] = useState(0);
  const [state, setState] = useState<{ key: string; status: 'pending' | 'ready' | 'error'; responses: PhaseSuccess[] }>({ key: '', status: 'pending', responses: [] });
  useEffect(() => {
    const samples = ref.current, responses: PhaseSuccess[] = [];
    let live = true, index = 0;
    setState({ key, status: 'pending', responses: [] });
    const controller = new PhaseCalculationController(createPhaseWorker, next => {
      if (!live) return;
      if (next.status === 'error') setState({ key, status: 'error', responses: [] });
      if (next.status !== 'ready') return;
      responses.push(next.response); index++;
      if (index === samples.length) setState({ key, status: 'ready', responses: [...responses] });
      else queueMicrotask(() => { if (live) controller.run(samples[index]); });
    });
    if (samples.length) controller.run(samples[0]);
    else setState({ key, status: 'ready', responses: [] });
    return () => { live = false; controller.dispose(); };
  }, [key, retry]);
  return { status: state.key === key ? state.status : 'pending' as const, responses: state.key === key && state.status === 'ready' ? state.responses : [], retry: () => setRetry(n => n + 1) };
}
