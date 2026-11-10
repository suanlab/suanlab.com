'use client';
import { useLanguage } from '@/components/language-provider';
import EuclideanKernelGraphic from './euclidean-kernel-graphic';
export default function EuclideanKernelCover() { const { language } = useLanguage(); return <EuclideanKernelGraphic language={language} compact />; }
