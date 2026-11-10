'use client';
import dynamic from 'next/dynamic';
import { useLanguage } from '@/components/language-provider';
import type { QubitTeachingId } from '@/lib/learning-qubit-teaching-input';
import '@/styles/qubit-lab.css';
function Loading() { const { language } = useLanguage(); return <p role="status">{language === 'ko' ? '실습 준비 중입니다.' : 'Preparing the experiment.'}</p>; }
const Lab = dynamic(() => import('./qubit-lab'), { loading: Loading });
export default function QubitReaderLab({ kind }: { kind: QubitTeachingId }) {
  const { language } = useLanguage();
  return <div id="experiment" data-qubit-reader-lab={kind}><Lab kind={kind} language={language} /></div>;
}
