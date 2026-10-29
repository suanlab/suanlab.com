import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { site } from '@/config/site';
import { bookPath, learningUnits } from '@/data/interactive-book';
import { getLearningContent } from '@/lib/learning-content';
import { InteractiveBookReader } from '@/components/learning/interactive-book';
interface Props { params: { unit: string }; }
export const dynamicParams = false;
export function generateStaticParams() { return learningUnits.map(unit => ({ unit: unit.id })); }
export function generateMetadata({ params }: Props): Metadata {
  const unit = learningUnits.find(item => item.id === params.unit);
  if (!unit) return { title: 'Unit not found' };
  return { title: `${unit.title.en} | Physics and Mathematics for AI`, description: unit.subtitle.en, alternates: { canonical: `${site.url}${bookPath}${unit.id}/` } };
}
export default async function LearningUnitPage({ params }: Props) {
  const unit = learningUnits.find(item => item.id === params.unit);
  if (!unit) notFound();
  return <InteractiveBookReader key={unit.id} unit={unit} html={await getLearningContent(unit.id)} />;
}
