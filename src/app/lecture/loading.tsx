export default function LectureLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="animate-pulse space-y-12">
        <div>
          <div className="flex items-center justify-between mb-6">
            <div className="h-8 bg-muted rounded w-28" />
            <div className="h-5 bg-muted rounded-full w-24" />
          </div>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="rounded-lg border overflow-hidden">
                <div className="aspect-[16/10] bg-muted" />
                <div className="p-3 space-y-2">
                  <div className="h-3 bg-muted rounded w-full" />
                  <div className="h-3 bg-muted rounded w-2/3" />
                </div>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="h-8 bg-muted rounded w-44 mb-6" />
          <div className="rounded-lg border p-6">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <div key={i} className="flex items-start gap-3">
                  <div className="h-8 w-8 bg-muted rounded-full shrink-0" />
                  <div className="space-y-1">
                    <div className="h-4 bg-muted rounded w-28" />
                    <div className="h-3 bg-muted rounded w-20" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
