export default function CourseLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="animate-pulse space-y-6">
        <div className="flex flex-wrap gap-2">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-9 bg-muted rounded-full w-20" />
          ))}
        </div>
        <div className="flex gap-2">
          <div className="h-10 bg-muted rounded w-48" />
          <div className="h-10 bg-muted rounded w-32" />
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="rounded-lg border p-4 space-y-3">
              <div className="h-5 bg-muted rounded w-3/4" />
              <div className="h-4 bg-muted rounded w-1/2" />
              <div className="h-3 bg-muted rounded w-full" />
              <div className="h-3 bg-muted rounded w-2/3" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
