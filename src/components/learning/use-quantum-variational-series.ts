'use client';
import { useEffect, useRef, useState } from 'react';
import { QuantumVariationalCalculationController } from '@/lib/learning-quantum-variational-controller';
import { createQuantumVariationalWorker } from '@/lib/learning-quantum-variational-worker-factory';
import { isQuantumVariationalInput, snapshotQuantumVariationalInput, quantumVariationalInputKey, type QuantumVariationalInput, type QuantumVariationalSuccess } from '@/lib/learning-quantum-variational-protocol';
/** Calculate fixed samples sequentially, always in workers. Never expose
 * another step's responses while its replacement effect is being installed. */
export function useQuantumVariationalSeries(inputs: readonly QuantumVariationalInput[]) {
  const key = JSON.stringify(Array.from(inputs, v => isQuantumVariationalInput(v) ? quantumVariationalInputKey(v) : 'invalid-input')), ref = useRef(inputs); ref.current = inputs;
  const [retry, setRetry] = useState(0);
  const [state, setState] = useState<{ key: string; status: 'pending' | 'ready' | 'error'; responses: QuantumVariationalSuccess[] }>({ key: '', status: 'pending', responses: [] });
  useEffect(() => {
    if (!Array.from(ref.current).every(isQuantumVariationalInput)) {
      setState({ key, status: 'error', responses: [] });
      return;
    }
    const samples = Array.from(ref.current, snapshotQuantumVariationalInput), responses: QuantumVariationalSuccess[] = [];
    let live = true, index = 0;
    setState({ key, status: 'pending', responses: [] });
    const controller = new QuantumVariationalCalculationController(createQuantumVariationalWorker, next => {
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
