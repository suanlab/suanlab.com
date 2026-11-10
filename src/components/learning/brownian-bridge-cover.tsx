'use client';
import { useLanguage } from '@/components/language-provider';
import BrownianBridgeGraphic from './brownian-bridge-graphic';
export default function BrownianBridgeCover() { const { language } = useLanguage(); return <BrownianBridgeGraphic language={language} compact />; }
