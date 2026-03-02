export default function BlogLoading() {
  return (
    <div className="container py-8">
      <div className="grid gap-8 lg:grid-cols-4">
        {/* Sidebar Skeleton */}
        <aside className="lg:col-span-1 space-y-6">
          {/* Search skeleton */}
          <div className="animate-pulse space-y-2">
            <div className="h-10 bg-muted rounded" />
          </div>

          {/* Categories skeleton */}
          <div className="animate-pulse space-y-3">
            <div className="h-6 bg-muted rounded w-1/2" />
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-8 bg-muted rounded" />
            ))}
          </div>

          {/* Tags skeleton */}
          <div className="animate-pulse space-y-3">
            <div className="h-6 bg-muted rounded w-1/2" />
            <div className="flex flex-wrap gap-2">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <div key={i} className="h-6 bg-muted rounded-full w-16" />
              ))}
            </div>
          </div>
        </aside>

        {/* Main Content Skeleton */}
        <main className="lg:col-span-3">
          {/* Header skeleton */}
          <div className="animate-pulse space-y-4 mb-8">
            <div className="h-8 bg-muted rounded w-1/3" />
            <div className="h-4 bg-muted rounded w-1/2" />
          </div>

          {/* Post Grid skeleton */}
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="animate-pulse space-y-3">
                <div className="h-40 bg-muted rounded-lg" />
                <div className="h-6 bg-muted rounded w-3/4" />
                <div className="h-4 bg-muted rounded w-full" />
                <div className="h-4 bg-muted rounded w-2/3" />
                <div className="flex gap-2">
                  <div className="h-6 bg-muted rounded-full w-12" />
                  <div className="h-6 bg-muted rounded-full w-12" />
                </div>
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
