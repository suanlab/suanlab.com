export default function ProjectLoading() {
  return (
    <div className="animate-pulse">
      <div className="py-8 bg-muted/30">
        <div className="container">
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="rounded-lg border-0 shadow-sm p-6 text-center bg-background">
                <div className="h-8 w-8 bg-muted rounded mx-auto mb-3" />
                <div className="h-8 bg-muted rounded w-16 mx-auto" />
                <div className="h-3 bg-muted rounded w-12 mx-auto mt-2" />
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="container py-16 md:py-20">
        <div className="mx-auto max-w-2xl text-center mb-12 space-y-4">
          <div className="h-9 bg-muted rounded w-64 mx-auto" />
          <div className="h-4 bg-muted rounded w-80 mx-auto" />
        </div>
        <div className="flex gap-2 mb-8">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-9 bg-muted rounded-full w-24" />
          ))}
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="rounded-lg border p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="h-5 bg-muted rounded w-3/4" />
                <div className="h-5 bg-muted rounded-full w-16" />
              </div>
              <div className="h-3 bg-muted rounded w-full" />
              <div className="h-3 bg-muted rounded w-2/3" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
