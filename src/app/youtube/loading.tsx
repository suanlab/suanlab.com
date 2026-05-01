export default function YouTubeLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="animate-pulse grid gap-8 lg:grid-cols-4">
        <aside className="lg:col-span-1 space-y-2">
          <div className="h-6 bg-muted rounded w-24 mb-4" />
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <div key={i} className="h-12 bg-muted rounded-lg" />
          ))}
        </aside>
        <div className="lg:col-span-3">
          <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="rounded-lg border p-4 flex items-center gap-3">
                <div className="h-10 w-10 bg-muted rounded-lg shrink-0" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-muted rounded w-3/4" />
                  <div className="h-3 bg-muted rounded w-1/2" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
