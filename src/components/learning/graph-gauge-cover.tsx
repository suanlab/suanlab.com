'use client';
import {useLanguage} from '@/components/language-provider';
import GraphGaugeGraphic from './graph-gauge-graphic';
export default function GraphGaugeCover(){const{language}=useLanguage();return <GraphGaugeGraphic language={language} compact/>;}
