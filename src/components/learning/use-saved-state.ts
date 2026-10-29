'use client';
import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';

/** Read before the first write; reject old or malformed browser state. */
export function useSavedState<T>(key: string, initial: T, valid: (value: unknown) => value is T): [T, Dispatch<SetStateAction<T>>, boolean, boolean] {
  const [value, setValue] = useState(initial); const [ready, setReady] = useState(false); const [available, setAvailable] = useState(true);
  useEffect(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw) {
        try {
          const saved = JSON.parse(raw);
          if (saved && saved.version === 1 && valid(saved.value)) setValue(saved.value);
        } catch { /* Malformed saved JSON is replaced by validated defaults. */ }
      }
    } catch { setAvailable(false); }
    setReady(true);
  }, [key, valid]);
  useEffect(() => {
    if (!ready) return;
    try { localStorage.setItem(key, JSON.stringify({ version: 1, value })); }
    catch { setAvailable(false); }
  }, [key, ready, value]);
  return [value, setValue, ready, available];
}
