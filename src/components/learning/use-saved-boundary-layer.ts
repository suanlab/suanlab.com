'use client';
import { useEffect, useState } from 'react';
import { readBoundaryLayerInput, type BoundaryLayerInput } from '@/lib/learning-boundary-layer-input';
export const boundaryLayerStorageKey = 'suanlab-boundary-layer-v1';
export function parseStoredBoundaryLayer(raw: string): BoundaryLayerInput | null {
  try { if (raw.length > 4096) return null; const data = JSON.parse(raw); return data && typeof data === 'object' && !Array.isArray(data) && Object.keys(data).length === 2 && data.version === 1 ? readBoundaryLayerInput(data.value) : null; } catch { return null; }
}
export function useSavedBoundaryLayer(initial: BoundaryLayerInput) {
  const [input, setInput] = useState(initial), [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => { let next = initial; try { const raw = localStorage.getItem(boundaryLayerStorageKey); if (raw !== null) { const parsed = parseStoredBoundaryLayer(raw); if (parsed) next = parsed; else setStorage('corrupt'); } } catch { setStorage('unavailable'); } setInput(next); setReady(true); }, [initial]);
  useEffect(() => { if (!ready) return; try { localStorage.setItem(boundaryLayerStorageKey, JSON.stringify({ version: 1, value: input })); } catch { setStorage('unavailable'); } }, [input, ready]);
  return { input, ready, storage, setInput(value: BoundaryLayerInput) { const next = readBoundaryLayerInput(value); if (next) setInput(next); } };
}
