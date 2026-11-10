'use client';
import { useEffect, useRef, useState } from 'react';
import { FdtCalculationController } from '@/lib/learning-fluctuation-dissipation-controller';
import { createFdtWorker } from '@/lib/learning-fluctuation-dissipation-worker-factory';
import { isFdtInput, snapshotFdtInput, fdtInputKey, type FdtInput, type FdtSuccess } from '@/lib/learning-fluctuation-dissipation-protocol';
/** Calculate fixed samples sequentially, always in workers. Never expose
 * another step's responses while its replacement effect is being installed. */
export function useFdtSeries(inputs: readonly FdtInput[]) {
  const key = JSON.stringify(Array.from(inputs, v => isFdtInput(v) ? fdtInputKey(v) : 'invalid-input')), ref = useRef(inputs); ref.current = inputs;
  const [retry, setRetry] = useState(0);
  const [state, setState] = useState<{ key: string; status: 'pending' | 'ready' | 'error'; responses: FdtSuccess[] }>({ key: '', status: 'pending', responses: [] });
  useEffect(() => {
    if (!Array.from(ref.current).every(isFdtInput)) {
      setState({ key, status: 'error', responses: [] });
      return;
    }
    const samples = Array.from(ref.current, snapshotFdtInput), responses: FdtSuccess[] = [];
    let live = true, index = 0;
    setState({ key, status: 'pending', responses: [] });
    const controller = new FdtCalculationController(createFdtWorker, next => {
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
