export default function BlogPostLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="max-w-4xl mx-auto animate-pulse space-y-8">
        <div className="rounded-lg border p-6">
          <div className="flex flex-wrap items-center gap-4">
            <div className="h-4 bg-muted rounded w-24" />
            <div className="h-4 bg-muted rounded w-20" />
            <div className="h-5 bg-muted rounded-full w-24" />
            <div className="flex gap-1">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-5 bg-muted rounded-full w-14" />
              ))}
            </div>
          </div>
        </div>
        <div className="aspect-video bg-muted rounded-lg" />
        <div className="space-y-4">
          <div className="h-6 bg-muted rounded w-3/4" />
          <div className="h-4 bg-muted rounded w-full" />
          <div className="h-4 bg-muted rounded w-5/6" />
          <div className="h-4 bg-muted rounded w-full" />
          <div className="h-4 bg-muted rounded w-4/5" />
          <div className="h-6 bg-muted rounded w-2/3 mt-6" />
          <div className="h-4 bg-muted rounded w-full" />
          <div className="h-4 bg-muted rounded w-5/6" />
          <div className="h-4 bg-muted rounded w-full" />
        </div>
      </div>
    </div>
  );
}
