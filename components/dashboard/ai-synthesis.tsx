"use client"

import { Loader2, Sparkles } from "lucide-react"
import type { AISynthesisResponse } from "@/lib/api"

interface AISynthesisProps {
  selectedDate: string
  timeRange: string
  data?: AISynthesisResponse | null
  loading?: boolean
}

function formatDate(date: string) {
  return new Date(date).toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  })
}

function formatTimeRange(timeRange: string) {
  const labels: Record<string, string> = {
    "12h": "12 hours",
    "6h": "6 hours",
    "4h": "4 hours",
    "3h": "3 hours",
    "2h": "2 hours",
    "1h": "1 hour",
    "30m": "30 minutes",
  }
  return labels[timeRange] ?? timeRange
}

const badgeToneClasses: Record<string, string> = {
  blue: "bg-blue-500/10 text-blue-400",
  green: "bg-emerald-500/10 text-emerald-400",
  orange: "bg-orange-500/10 text-orange-400",
  purple: "bg-purple-500/10 text-purple-400",
  red: "bg-red-500/10 text-red-400",
}

export function AISynthesis({ selectedDate, timeRange, data, loading = false }: AISynthesisProps) {
  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-elevated">
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-purple-600 shadow-elevated-sm">
          <Sparkles className="h-5 w-5 text-white" />
        </div>
        <div>
          <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">ALIVE AI Synthesis</h3>
          <p className="text-sm text-muted-foreground">
            Intelligent summary for {formatDate(data?.date ?? selectedDate)} · {formatTimeRange(data?.timeRange ?? timeRange)}
          </p>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-10 text-muted-foreground">
          <Loader2 className="mr-2 h-5 w-5 animate-spin" />
          Loading AI synthesis...
        </div>
      ) : data ? (
        <div className="space-y-6">
          {data.sections.map((section, index) => (
            <section key={section.title}>
              <h4 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                {index + 1}. {section.title}
              </h4>
              <div className="rounded-2xl border border-border bg-secondary/30 p-4">
                <p className="text-sm leading-relaxed text-foreground">{section.body}</p>
                {section.badges.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {section.badges.map((badge) => (
                      <span
                        key={`${section.title}-${badge.label}`}
                        className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${badgeToneClasses[badge.tone] ?? badgeToneClasses.blue}`}
                      >
                        {badge.label}: {badge.value}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-border p-6 text-sm text-muted-foreground">
          No AI synthesis is available yet for this selection.
        </div>
      )}
    </div>
  )
}
