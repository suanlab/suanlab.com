'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Search, FileText, BookOpen, GraduationCap, Wand2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import type { BlogPostMeta } from '@/lib/blog';
import type { Publication } from '@/data/publications';
import type { Lecture } from '@/data/lectures';
import { promptBuilders, promptSnippets } from '@/data/prompts';

interface SearchClientProps {
  posts: BlogPostMeta[];
  publications: Publication[];
  lectures: Lecture[];
}

interface SearchResult {
  type: 'blog' | 'publication' | 'lecture' | 'prompt';
  title: string;
  description: string;
  href: string;
  badges?: string[];
}

export default function SearchClient({ posts, publications, lectures }: SearchClientProps) {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get('q') || '';
  const [searchQuery, setSearchQuery] = useState(initialQuery);
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-focus on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Debounce search (300ms)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const performSearch = useCallback((query: string): SearchResult[] => {
    if (query.length < 2) return [];

    const results: SearchResult[] = [];
    const lowerQuery = query.toLowerCase();

    // Search blog posts
    posts.forEach((post) => {
      if (
        post.title.toLowerCase().includes(lowerQuery) ||
        post.excerpt.toLowerCase().includes(lowerQuery)
      ) {
        results.push({
          type: 'blog',
          title: post.title,
          description: post.excerpt,
          href: `/blog/${post.slug}`,
          badges: [post.category, ...post.tags.slice(0, 2)],
        });
      }
    });

    // Search publications
    publications.forEach((pub) => {
      if (
        pub.title.toLowerCase().includes(lowerQuery) ||
        pub.authors.toLowerCase().includes(lowerQuery) ||
        pub.venue.toLowerCase().includes(lowerQuery)
      ) {
        results.push({
          type: 'publication',
          title: pub.title,
          description: `${pub.authors} - ${pub.venue}`,
          href: `/publication/${pub.id}`,
          badges: [pub.badge || pub.type],
        });
      }
    });

    // Search lectures
    lectures.forEach((lecture) => {
      if (
        lecture.titleKo.toLowerCase().includes(lowerQuery) ||
        lecture.titleEn.toLowerCase().includes(lowerQuery)
      ) {
        results.push({
          type: 'lecture',
          title: `${lecture.titleKo} (${lecture.titleEn})`,
          description: lecture.descriptionKo || lecture.description,
          href: `/lecture/${lecture.slug}`,
          badges: lecture.topics.slice(0, 2),
        });
      }
    });

    // Search prompt builders (titles/descriptions/tags)
    promptBuilders.forEach((b) => {
      const hay = `${b.title.ko} ${b.title.en} ${b.description.ko} ${b.description.en} ${b.tags.join(' ')}`.toLowerCase();
      if (hay.includes(lowerQuery)) {
        results.push({
          type: 'prompt',
          title: `${b.title.ko} (${b.title.en})`,
          description: b.description.ko,
          href: '/prompts/',
          badges: ['빌더', ...b.tags.slice(0, 2)],
        });
      }
    });

    // Search prompt snippets
    promptSnippets.forEach((s) => {
      const hay = `${s.title.ko} ${s.title.en} ${s.description.ko} ${s.description.en} ${s.tags.join(' ')}`.toLowerCase();
      if (hay.includes(lowerQuery)) {
        results.push({
          type: 'prompt',
          title: `${s.title.ko} (${s.title.en})`,
          description: s.description.ko,
          href: '/prompts/',
          badges: ['라이브러리', ...s.tags.slice(0, 2)],
        });
      }
    });

    return results;
  }, [posts, publications, lectures]);

  const results = performSearch(debouncedQuery);

  // Group results by type
  const blogResults = results.filter((r) => r.type === 'blog');
  const publicationResults = results.filter((r) => r.type === 'publication');
  const lectureResults = results.filter((r) => r.type === 'lecture');
  const promptResults = results.filter((r) => r.type === 'prompt');



  const typeLabels = {
    blog: '블로그 포스트',
    publication: '논문',
    lecture: '강의',
    prompt: '프롬프트',
  };

  const renderResultGroup = (
    label: string,
    icon: React.ReactNode,
    items: SearchResult[]
  ) => {
    if (items.length === 0) return null;

    return (
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-4">
          {icon}
          <h2 className="text-lg font-semibold">{label}</h2>
          <Badge variant="secondary">{items.length}</Badge>
        </div>
        <div className="space-y-3">
          {items.map((item, index) => (
            <Link key={`${item.type}-${item.href}-${index}`} href={item.href}>
              <Card className="hover:shadow-md transition-shadow">
                <CardContent className="p-4">
                  <h3 className="font-medium mb-1 line-clamp-1">{item.title}</h3>
                  <p className="text-sm text-muted-foreground line-clamp-2 mb-2">
                    {item.description}
                  </p>
                  {item.badges && item.badges.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {item.badges.map((badge, i) => (
                        <Badge key={i} variant="outline" className="text-xs">
                          {badge}
                        </Badge>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    );
  };

  return (
    <section className="py-16 md:py-20">
      <div className="container">
        <div className="max-w-3xl mx-auto">
          {/* Search Input */}
          <div className="mb-8">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
              <Input
                ref={inputRef}
                type="text"
                placeholder="검색어를 입력하세요 (최소 2자)"
                className="h-14 pl-12 text-lg"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            {debouncedQuery.length >= 2 && (
              <p className="mt-3 text-sm text-muted-foreground">
                &quot;{debouncedQuery}&quot; 검색 결과: 총 {results.length}개
              </p>
            )}
          </div>

          {/* Results */}
          {debouncedQuery.length >= 2 && results.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">검색 결과가 없습니다.</p>
              <p className="text-sm text-muted-foreground mt-2">
                다른 검색어를 시도해 보세요.
              </p>
            </div>
          )}

          {debouncedQuery.length >= 2 && results.length > 0 && (
            <>
              {renderResultGroup(
                typeLabels.blog,
                <FileText className="h-5 w-5 text-primary" />,
                blogResults
              )}
              {renderResultGroup(
                typeLabels.publication,
                <BookOpen className="h-5 w-5 text-primary" />,
                publicationResults
              )}
              {renderResultGroup(
                typeLabels.lecture,
                <GraduationCap className="h-5 w-5 text-primary" />,
                lectureResults
              )}
              {renderResultGroup(
                typeLabels.prompt,
                <Wand2 className="h-5 w-5 text-primary" />,
                promptResults
              )}
            </>
          )}

          {/* Initial state */}
          {debouncedQuery.length < 2 && (
            <div className="text-center py-12 text-muted-foreground">
              <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>검색어를 입력하면 블로그, 논문, 강의를 검색합니다.</p>
              <p className="text-sm mt-2">최소 2자 이상 입력해주세요.</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
