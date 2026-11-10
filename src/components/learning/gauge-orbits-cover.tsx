'use client';
import {useLanguage} from '@/components/language-provider';
import GaugeOrbitGraphic from './gauge-orbits-graphic';
export default function GaugeOrbitCover(){const{language}=useLanguage();return <GaugeOrbitGraphic language={language} compact/>;}
