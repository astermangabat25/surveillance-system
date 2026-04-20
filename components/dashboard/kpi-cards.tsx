"use client"

import { Activity, Car, MapPin, Video } from "lucide-react"
import type { DashboardSummary } from "@/lib/api"

interface KPICardsProps {
  summary?: DashboardSummary | null
  loading?: boolean
  footageClipCount?: number
  averageLos?: string
}

export function KPICards({ summary, loading = false, footageClipCount = 0, averageLos = "--" }: KPICardsProps) {
  const kpis = [
    {
      label: "Tracked Vehicles",
      value: loading ? "--" : (summary?.totalUniquePedestrians ?? 0).toLocaleString(),
      hint: "All unique vehicle tracks counted for the selected date",
      icon: Car,
      color: "primary",
    },
    {
      label: "Footage Clips",
      value: loading ? "--" : `${footageClipCount}`,
      hint: "Uploaded clips available for the selected date",
      icon: Video,
      color: "accent",
    },
    {
      label: "Monitored Locations",
      value: loading ? "--" : `${summary?.monitoredLocations ?? 0}`,
      hint: "Locations contributing footage",
      icon: MapPin,
      color: "chart-3",
    },
    {
      label: "Average LOS",
      value: loading ? "--" : averageLos,
      hint: "Mean Level of Service grade across the current time window",
      icon: Activity,
      color: "chart-4",
    },
  ]

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
      {kpis.map((kpi) => (
        <KPICard key={kpi.label} {...kpi} />
      ))}
    </div>
  )
}

function KPICard({
  label,
  value,
  hint,
  icon: Icon,
  color,
}: {
  label: string
  value: string
  hint: string
  icon: React.ElementType
  color: string
}) {
  const colorClasses: Record<string, { bg: string; text: string; icon: string }> = {
    primary: { bg: "bg-primary/10", text: "text-primary", icon: "text-primary" },
    accent: { bg: "bg-accent/10", text: "text-accent", icon: "text-accent" },
    "chart-3": { bg: "bg-chart-3/10", text: "text-chart-3", icon: "text-chart-3" },
    "chart-4": { bg: "bg-chart-4/10", text: "text-chart-4", icon: "text-chart-4" },
  }

  const colors = colorClasses[color] || colorClasses.primary

  return (
    <div className="rounded-3xl border border-border bg-card p-5 shadow-elevated-sm transition-all hover:border-primary/30">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className={`rounded-2xl p-3 ${colors.bg}`}>
          <Icon className={`h-5 w-5 ${colors.icon}`} />
        </div>
        <div className="rounded-full bg-secondary px-2 py-1 text-[11px] font-medium text-muted-foreground">
          Live
        </div>
      </div>

      <p className={`mb-1 text-3xl font-bold ${colors.text}`}>{value}</p>
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-2 text-xs text-muted-foreground">{hint}</p>
    </div>
  )
}
