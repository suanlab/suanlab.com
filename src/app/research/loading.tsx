export default function ResearchLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="animate-pulse">
        <div className="mx-auto max-w-2xl text-center mb-12">
          <div className="h-9 bg-muted rounded w-64 mx-auto" />
          <div className="h-4 bg-muted rounded w-80 mx-auto mt-4" />
        </div>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="rounded-lg border overflow-hidden">
              <div className="aspect-video bg-muted" />
              <div className="p-6 space-y-3">
                <div className="h-5 bg-muted rounded w-3/4" />
                <div className="h-4 bg-muted rounded w-1/2" />
                <div className="h-3 bg-muted rounded w-full" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
