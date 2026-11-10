'use client';
import dynamic from 'next/dynamic';
import {useLanguage} from '@/components/language-provider';
import '@/styles/qubit-lab.css';
import '@/styles/information-lab.css';
import '@/styles/scalar-field-lab.css';
function Loading(){const{language}=useLanguage();return <p role="status">{language==='ko'?'실험을 준비하고 있습니다.':'Preparing the experiment.'}</p>;}
const Lab=dynamic(()=>import('./thermal-field-lab'),{loading:Loading});
export default function ThermalFieldReaderLab(){const{language}=useLanguage();return <div id="experiment" data-thermal-field-reader-lab><Lab language={language} embedded/></div>;}
