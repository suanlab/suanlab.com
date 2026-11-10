'use client';
import { useEffect, useRef, useState } from 'react';
import { CorrelationCalculationController } from '@/lib/learning-correlations-controller';
import { createCorrelationWorker } from '@/lib/learning-correlations-worker-factory';
import { isCorrelationInput, correlationInputKey, type CorrelationInput, type CorrelationSuccess } from '@/lib/learning-correlations-protocol';
/** Calculate fixed samples sequentially, always in workers. Never expose
 * another step's responses while its replacement effect is being installed. */
export function useCorrelationSeries(inputs: readonly CorrelationInput[]) {
  const key = JSON.stringify(inputs.map(v => isCorrelationInput(v) ? correlationInputKey(v) : 'invalid-input')), ref = useRef(inputs); ref.current = inputs;
  const [retry, setRetry] = useState(0);
  const [state, setState] = useState<{ key: string; status: 'pending' | 'ready' | 'error'; responses: CorrelationSuccess[] }>({ key: '', status: 'pending', responses: [] });
  useEffect(() => {
    const samples = ref.current, responses: CorrelationSuccess[] = [];
    let live = true, index = 0;
    setState({ key, status: 'pending', responses: [] });
    const controller = new CorrelationCalculationController(createCorrelationWorker, next => {
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
