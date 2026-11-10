'use client';
import dynamic from 'next/dynamic';
import {useLanguage} from '@/components/language-provider';
import type {InformationTeachingId} from '@/lib/learning-information-teaching';
import '@/styles/qubit-lab.css';
import '@/styles/information-lab.css';
function Loading(){const{language}=useLanguage();return <p role="status">{language==='ko'?'실험을 준비하고 있습니다.':'Preparing the experiment.'}</p>;}
const labs={
 'binary-qubit-state-discrimination':dynamic(()=>import('./information-discrimination-lab'),{loading:Loading}),
 'quantum-entropy-compression-conditional-information':dynamic(()=>import('./information-entropy-lab'),{loading:Loading}),
 'classical-quantum-ensembles-holevo-information':dynamic(()=>import('./information-ensemble-lab'),{loading:Loading}),
 'quantum-relative-entropy-data-processing':dynamic(()=>import('./information-relative-lab'),{loading:Loading}),
 'no-cloning-bb84-information-disturbance':dynamic(()=>import('./information-bb84-lab'),{loading:Loading}),
 'teleportation-superdense-coding-resources':dynamic(()=>import('./information-teleportation-lab'),{loading:Loading}),
 'quantum-error-correction-syndrome-knill-laflamme':dynamic(()=>import('./information-error-correction-lab'),{loading:Loading}),
 'pure-state-entanglement-majorization-resources':dynamic(()=>import('./information-majorization-lab'),{loading:Loading}),
 'quantum-channel-capacities-coding-limits':dynamic(()=>import('./information-capacity-lab'),{loading:Loading}),
}as const;
export default function InformationReaderLab({kind}:{kind:InformationTeachingId}){const{language}=useLanguage(),Lab=labs[kind];return <div id="experiment" data-information-reader-lab={kind}><Lab language={language} embedded/></div>;}
