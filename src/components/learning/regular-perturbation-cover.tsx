'use client';
import {useLanguage} from '@/components/language-provider';
import RegularPerturbationGraphic from './regular-perturbation-graphic';
export default function RegularPerturbationCover(){const{language}=useLanguage();return <RegularPerturbationGraphic language={language} compact/>;}
