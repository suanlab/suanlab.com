'use client';
import {useLanguage} from '@/components/language-provider';
import NongaussianFieldGraphic from './nongaussian-field-graphic';
export default function NongaussianFieldCover(){const{language}=useLanguage();return <NongaussianFieldGraphic language={language} compact/>;}
