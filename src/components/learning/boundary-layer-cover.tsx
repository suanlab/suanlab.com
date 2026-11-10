'use client';
import {useLanguage} from '@/components/language-provider';
import BoundaryLayerGraphic from './boundary-layer-graphic';
export default function BoundaryLayerCover(){const{language}=useLanguage();return <BoundaryLayerGraphic language={language} compact/>;}
