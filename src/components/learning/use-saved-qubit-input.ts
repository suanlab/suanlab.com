'use client';
import { useEffect, useState } from 'react';
import { qubitTeachingInput, readQubitTeachingInput, parseQubitTeachingStoredInput, type QubitTeachingId } from '@/lib/learning-qubit-teaching-input';
import type { QubitInput } from '@/lib/learning-qubit-input';

/** Mount this hook in a component keyed by teaching ID. */
export function useSavedQubitInput(kind: QubitTeachingId) {
  const key = `suanlab-qubit-${kind}`;
  const [input, setInput] = useState(() => qubitTeachingInput(kind));
  const [ready, setReady] = useState(false);
  const [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => {
    let next = qubitTeachingInput(kind);
    try {
      const raw = localStorage.getItem(key);
      if (raw !== null) {
        const saved = parseQubitTeachingStoredInput(kind, raw);
        if (saved) next = saved; else setStorage('corrupt');
      }
    } catch { setStorage('unavailable'); }
    setInput(next); setReady(true);
  }, [kind, key]);
  useEffect(() => {
    if (!ready) return;
    try { localStorage.setItem(key, JSON.stringify({ version: 1, value: input })); }
    catch { setStorage('unavailable'); }
  }, [key, ready, input]);
  return { input, ready, storage, setInput(next: QubitInput) {
    const owned = readQubitTeachingInput(kind, next);
    if (owned) setInput(owned);
  } };
}
