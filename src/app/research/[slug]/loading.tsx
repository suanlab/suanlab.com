export default function ResearchDetailLoading() {
  return (
    <div className="animate-pulse">
      <div className="container py-16 md:py-20">
        <div className="grid gap-12 lg:grid-cols-2 lg:items-start">
          <div className="aspect-video bg-muted rounded-xl" />
          <div className="space-y-4">
            <div className="h-14 w-14 bg-muted rounded-xl" />
            <div className="h-8 bg-muted rounded w-3/4" />
            <div className="h-5 bg-muted rounded w-1/2" />
            <div className="mt-6 space-y-3">
              <div className="h-4 bg-muted rounded w-full" />
              <div className="h-4 bg-muted rounded w-5/6" />
              <div className="h-4 bg-muted rounded w-4/5" />
            </div>
            <div className="flex flex-wrap gap-2 mt-6">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-6 bg-muted rounded-full w-16" />
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="py-16 bg-muted/30">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <div className="h-8 bg-muted rounded w-56 mx-auto" />
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="rounded-lg border p-4 flex items-center gap-3">
                <div className="h-8 w-8 bg-muted rounded-lg shrink-0" />
                <div className="h-4 bg-muted rounded w-20" />
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="container py-16 md:py-20">
        <div className="mx-auto max-w-2xl text-center mb-12">
          <div className="h-8 bg-muted rounded w-52 mx-auto" />
        </div>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="rounded-lg border p-4 space-y-3">
              <div className="flex items-start gap-4">
                <div className="h-10 w-10 bg-muted rounded-lg shrink-0" />
                <div className="h-5 bg-muted rounded w-3/4" />
              </div>
              <div className="h-3 bg-muted rounded w-full" />
              <div className="h-3 bg-muted rounded w-5/6" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
