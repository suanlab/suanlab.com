'use client';
import { useEffect, useRef, useState } from 'react';
import { ObservableCalculationController } from '@/lib/learning-observable-controller';
import { createObservableWorker } from '@/lib/learning-observable-worker-factory';
import { isObservableInput, snapshotObservableInput, observableInputKey, type ObservableInput, type ObservableSuccess } from '@/lib/learning-observable-protocol';
/** Calculate fixed samples sequentially, always in workers. Never expose
 * another step's responses while its replacement effect is being installed. */
export function useObservableSeries(inputs: readonly ObservableInput[]) {
  const key = JSON.stringify(Array.from(inputs, v => isObservableInput(v) ? observableInputKey(v) : 'invalid-input')), ref = useRef(inputs); ref.current = inputs;
  const [retry, setRetry] = useState(0);
  const [state, setState] = useState<{ key: string; status: 'pending' | 'ready' | 'error'; responses: ObservableSuccess[] }>({ key: '', status: 'pending', responses: [] });
  useEffect(() => {
    if (!Array.from(ref.current).every(isObservableInput)) {
      setState({ key, status: 'error', responses: [] });
      return;
    }
    const samples = Array.from(ref.current, snapshotObservableInput), responses: ObservableSuccess[] = [];
    let live = true, index = 0;
    setState({ key, status: 'pending', responses: [] });
    const controller = new ObservableCalculationController(createObservableWorker, next => {
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
