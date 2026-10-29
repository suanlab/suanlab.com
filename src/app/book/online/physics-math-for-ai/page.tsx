import type { Metadata } from 'next';
import { site } from '@/config/site';
import { bookPath, interactiveBook } from '@/data/interactive-book';
import { InteractiveBookOverview } from '@/components/learning/interactive-book';
export const metadata: Metadata = {
  title: 'Physics and Mathematics for AI', description: interactiveBook.description.en,
  alternates: { canonical: `${site.url}${bookPath}` },
};
export default function PhysicsMathBookPage() { return <InteractiveBookOverview />; }
