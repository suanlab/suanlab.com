'use client';
import { blockOrderData } from '@/lib/learning-block-order-input';
import BlockOrderConcept from './block-order-concept';
import { useSavedState } from './use-saved-state';
import '@/styles/block-order-lab.css';
function valid(value: unknown): value is { step: number; motion: boolean } {
  const d = blockOrderData(value,['step','motion']);
  return !!d && typeof d.step === 'number' && Number.isInteger(d.step) && d.step >= 0 && d.step < 6 && typeof d.motion === 'boolean';
}
export default function BlockOrderReaderConcept({ language }: { language: 'ko' | 'en' }) {
  const [saved,setSaved,ready] = useSavedState('suanlab-learning:block-order-concept-v1',{ step: 0,motion: true },valid);
  return <div className="block-order-lab"><BlockOrderConcept language={language} step={saved.step} motion={saved.motion} ready={ready}
    onStep={step => setSaved(p => ({ ...p,step }))} onMotion={motion => setSaved(p => ({ ...p,motion }))}/></div>;
}
