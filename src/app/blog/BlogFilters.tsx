'use client';

import { Search, Folder, Tag } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';

interface BlogFiltersProps {
  categories: string[];
  tags: string[];
}

export default function BlogFilters({ categories, tags }: BlogFiltersProps) {
  return (
    <>
      {/* 검색 */}
      <Card className="mb-6">
        <CardContent className="p-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              type="text"
              placeholder="검색..."
              className="pl-10"
            />
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            * 검색 기능은 준비 중입니다
          </p>
        </CardContent>
      </Card>

      {/* 카테고리 */}
      <Card className="mb-6">
        <CardContent className="p-4">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Folder className="h-4 w-4 text-primary" />
            카테고리
          </h3>
          <div className="space-y-2">
            {categories.map((category) => (
              <div
                key={category}
                className="block w-full text-left px-3 py-2 rounded-md text-sm hover:bg-muted transition-colors"
              >
                {category}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 태그 */}
      <Card className="mb-6">
        <CardContent className="p-4">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Tag className="h-4 w-4 text-primary" />
            태그
          </h3>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <Badge
                key={tag}
                variant="secondary"
                className="cursor-pointer"
              >
                {tag}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>
    </>
  );
}
