'use client';
import {useLanguage} from '@/components/language-provider';
import ScalarFieldGraphic from './scalar-field-graphic';
export default function ScalarFieldCover(){const{language}=useLanguage();return <ScalarFieldGraphic language={language} compact/>;}
