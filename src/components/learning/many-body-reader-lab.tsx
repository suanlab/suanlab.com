'use client';
import dynamic from 'next/dynamic';
import {useLanguage} from '@/components/language-provider';
import type {ManyBodyTeachingId} from '@/lib/learning-many-body-teaching';
function Loading(){const{language}=useLanguage();return <p role="status">{language==='ko'?'실습 준비 중입니다.':'Preparing the experiment.'}</p>;}
const labs={
 'finite-spin-chain-local-hamiltonians':dynamic(()=>import('./many-body-spin-lab'),{loading:Loading}),
 'fermion-fock-parity-operator-algebra':dynamic(()=>import('./many-body-fermion-lab'),{loading:Loading}),
 'boson-fock-cutoff-boundary-algebra':dynamic(()=>import('./many-body-boson-lab'),{loading:Loading}),
 'matrix-product-state-contraction-gauge':dynamic(()=>import('./many-body-mps-lab'),{loading:Loading}),
 'schmidt-cuts-bond-rank-truncation':dynamic(()=>import('./many-body-schmidt-lab'),{loading:Loading}),
};
export default function ManyBodyReaderLab({kind}:{kind:ManyBodyTeachingId}){const Lab=labs[kind];return <div data-many-body-reader-lab={kind}><Lab/></div>;}
