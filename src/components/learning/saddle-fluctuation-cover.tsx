'use client';
import {useLanguage} from '@/components/language-provider';
import SaddleFluctuationGraphic from './saddle-fluctuation-graphic';
export default function SaddleFluctuationCover(){const {language}=useLanguage();return <SaddleFluctuationGraphic language={language} compact/>;}
