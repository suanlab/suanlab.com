'use client';
import {useLanguage} from '@/components/language-provider';
import GraphGaussGraphic from './graph-gauss-graphic';
export default function GraphGaussCover(){const{language}=useLanguage();return <GraphGaussGraphic language={language} compact/>;}
