export default function SuanLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="animate-pulse grid gap-12 lg:grid-cols-3">
        <div className="lg:col-span-1 space-y-6">
          <div className="rounded-lg border overflow-hidden">
            <div className="aspect-square bg-muted" />
            <div className="p-6 space-y-3">
              <div className="h-6 bg-muted rounded w-24 mx-auto" />
              <div className="h-4 bg-muted rounded w-40 mx-auto" />
              <div className="flex justify-center gap-3">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="h-9 w-9 bg-muted rounded-full" />
                ))}
              </div>
            </div>
          </div>
          <div className="rounded-lg border p-6 space-y-4">
            <div className="h-6 bg-muted rounded w-16" />
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="space-y-2">
                <div className="flex justify-between">
                  <div className="h-4 bg-muted rounded w-32" />
                  <div className="h-4 bg-muted rounded w-8" />
                </div>
                <div className="h-2 bg-muted rounded-full" />
              </div>
            ))}
          </div>
        </div>
        <div className="lg:col-span-2 space-y-8">
          <div className="space-y-4">
            <div className="h-8 bg-muted rounded w-32" />
            <div className="h-4 bg-muted rounded w-full" />
            <div className="h-4 bg-muted rounded w-5/6" />
          </div>
          <div className="rounded-lg border p-6 space-y-4">
            <div className="h-6 bg-muted rounded w-40" />
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="border-l-2 border-muted pl-4 space-y-1">
                <div className="h-4 bg-muted rounded w-48" />
                <div className="h-3 bg-muted rounded w-32" />
              </div>
            ))}
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            <div className="rounded-lg border p-6 space-y-4">
              <div className="h-6 bg-muted rounded w-28" />
              {[1, 2, 3].map((i) => (
                <div key={i} className="border-l-2 border-muted pl-4 space-y-1">
                  <div className="h-4 bg-muted rounded w-36" />
                  <div className="h-3 bg-muted rounded w-52" />
                </div>
              ))}
            </div>
            <div className="rounded-lg border p-6 space-y-3">
              <div className="h-6 bg-muted rounded w-36" />
              {[1, 2, 3, 4, 5, 6, 7].map((i) => (
                <div key={i} className="h-4 bg-muted rounded w-48" />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
