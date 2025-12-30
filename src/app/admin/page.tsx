import Link from 'next/link';
import { PenLine, FileText, BarChart } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { getAllPosts } from '@/lib/blog';

export default function AdminDashboard() {
  const posts = getAllPosts();
  const recentPosts = posts.slice(0, 5);

  return (
    <div className="container py-8">
      <h1 className="text-3xl font-bold mb-8">Dashboard</h1>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-3 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Posts</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{posts.length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Categories</CardTitle>
            <BarChart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {new Set(posts.map((p) => p.category)).size}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-primary text-primary-foreground">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Quick Action</CardTitle>
            <PenLine className="h-4 w-4" />
          </CardHeader>
          <CardContent>
            <Button asChild variant="secondary" className="w-full">
              <Link href="/admin/blog/new">Create New Post</Link>
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Recent Posts */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Posts</CardTitle>
        </CardHeader>
        <CardContent>
          {recentPosts.length === 0 ? (
            <p className="text-muted-foreground">No posts yet.</p>
          ) : (
            <div className="space-y-4">
              {recentPosts.map((post) => (
                <div
                  key={post.slug}
                  className="flex items-center justify-between border-b pb-4 last:border-0"
                >
                  <div>
                    <Link
                      href={`/blog/${post.slug}`}
                      className="font-medium hover:text-primary transition-colors"
                    >
                      {post.title}
                    </Link>
                    <p className="text-sm text-muted-foreground">
                      {post.date} | {post.category}
                    </p>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {post.readingTime} min read
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
