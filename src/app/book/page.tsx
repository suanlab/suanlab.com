import Link from 'next/link';
import { Globe, BookOpen, ArrowRight } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

export const metadata = {
  title: 'Book | SuanLab',
  description: 'Books by Professor Suan Lee - Online books and published books',
};

const bookCategories = [
  {
    title: 'Online Book',
    description: '온라인으로 제공되는 도서 콘텐츠와 학습 자료입니다.',
    href: '/book/online',
    icon: Globe,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
  },
  {
    title: 'Published Book',
    description: '출판된 프로그래밍 및 데이터 사이언스 서적입니다.',
    href: '/book/published',
    icon: BookOpen,
    color: 'text-green-500',
    bgColor: 'bg-green-500/10',
  },
];

export default function BookPage() {
  return (
    <>
      <PageHeader
        title="Book"
        subtitle="Books by Professor Suan Lee"
        breadcrumbs={[{ label: 'Book' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
              Explore <span className="text-primary">Books</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              온라인 도서와 출판 도서를 선택하여 살펴보세요
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-2 max-w-4xl mx-auto">
            {bookCategories.map((category) => (
              <Link key={category.href} href={category.href}>
                <Card className="h-full transition-all hover:shadow-lg hover:border-primary/50 cursor-pointer group">
                  <CardContent className="p-8">
                    <div className={`inline-flex p-4 rounded-xl ${category.bgColor} mb-6`}>
                      <category.icon className={`h-8 w-8 ${category.color}`} />
                    </div>
                    <h3 className="text-2xl font-bold mb-3 group-hover:text-primary transition-colors">
                      {category.title}
                    </h3>
                    <p className="text-muted-foreground mb-6">
                      {category.description}
                    </p>
                    <Button variant="outline" className="group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                      View Books
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
