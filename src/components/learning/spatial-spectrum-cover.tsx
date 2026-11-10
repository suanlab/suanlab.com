'use client';
import { useLanguage } from '@/components/language-provider';
import SpatialSpectrumGraphic from './spatial-spectrum-graphic';
export default function SpatialSpectrumCover() { const { language } = useLanguage(); return <SpatialSpectrumGraphic language={language} compact />; }
