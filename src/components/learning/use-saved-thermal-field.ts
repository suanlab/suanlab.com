'use client';
import { useEffect, useState } from 'react';
import { thermalFieldSteps, thermalFieldExperiments, thermalFieldQuestions } from '@/data/thermal-field-teaching';
import { emptyThermalState, parseStoredThermalState, readThermalState, thermalFieldStorageKey, type ThermalSavedState } from '@/lib/learning-thermal-field-state';
export const thermalLearningLimits = { steps: thermalFieldSteps.length, guides: thermalFieldExperiments.length, questions: thermalFieldQuestions.length, options: 3 };
export function useSavedThermalField() {
  const [model, setModel] = useState(() => emptyThermalState(thermalLearningLimits)), [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => {
    let next = emptyThermalState(thermalLearningLimits);
    try { const raw = localStorage.getItem(thermalFieldStorageKey); if (raw !== null) { const read = parseStoredThermalState(raw, thermalLearningLimits); if (read) next = read; else setStorage('corrupt'); } } catch { setStorage('unavailable'); }
    setModel(next); setReady(true);
  }, []);
  useEffect(() => { if (ready) try { localStorage.setItem(thermalFieldStorageKey, JSON.stringify({ version: 1, model })); } catch { setStorage('unavailable'); } }, [ready, model]);
  const update = (change: (previous: ThermalSavedState) => ThermalSavedState) => { if (ready) setModel(p => readThermalState(change(p), thermalLearningLimits) ?? p); };
  return { model, ready, storage, update };
}
