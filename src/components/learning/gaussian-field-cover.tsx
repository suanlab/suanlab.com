'use client';
import {useLanguage} from '@/components/language-provider';
import GaussianFieldGraphic from './gaussian-field-graphic';
export default function GaussianFieldCover(){const{language}=useLanguage();return <GaussianFieldGraphic language={language} compact/>;}
