"use client"

import { useMemo } from "react"
import type { EventRecord, VideoSeverityBucket } from "@/lib/api"

interface PlaybackTimelineProps {
  startTime: string
  endTime: string
  durationSeconds: number
  currentTimeSeconds: number
  events: EventRecord[]
  severityBuckets?: VideoSeverityBucket[]
  searchMatchOffsets?: number[]
  onSeek?: (seconds: number) => void
}

type SeverityLevel = "neutral" | "light" | "moderate" | "heavy"

const SEVERITY_STYLES: Record<SeverityLevel, { label: string; fill: string }> = {
  neutral: { label: "LOS A-B", fill: "rgba(16, 185, 129, 0.32)" },
  light: { label: "LOS C", fill: "rgba(132, 204, 22, 0.4)" },
  moderate: { label: "LOS D", fill: "rgba(245, 158, 11, 0.5)" },
  heavy: { label: "LOS E-F", fill: "rgba(239, 68, 68, 0.58)" },
}

function formatDuration(seconds: number) {
  const totalSeconds = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const remainingSeconds = totalSeconds % 60

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`
  }

  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`
}

function parseClockMinutes(value: string) {
  const trimmed = value.trim()
  const twelveHourMatch = trimmed.match(/^([0-9]{1,2}):([0-9]{2})(?::[0-9]{2})?\s*(AM|PM)$/i)

  if (twelveHourMatch) {
    let hours = Number(twelveHourMatch[1])
    const minutes = Number(twelveHourMatch[2])
    const period = twelveHourMatch[3].toUpperCase()

    if (period === "PM" && hours < 12) hours += 12
    if (period === "AM" && hours === 12) hours = 0
    return hours * 60 + minutes
  }

  const parts = trimmed.split(":")
  if (parts.length < 2) return null

  const hours = Number(parts[0])
  const minutes = Number(parts[1])
  if (!Number.isFinite(hours) || !Number.isFinite(minutes)) return null
  return hours * 60 + minutes
}

function formatClock(minutes: number) {
  const normalizedMinutes = ((Math.round(minutes) % 1440) + 1440) % 1440
  const hours24 = Math.floor(normalizedMinutes / 60)
  const mins = normalizedMinutes % 60
  const period = hours24 >= 12 ? "PM" : "AM"
  const hours12 = hours24 % 12 || 12
  return `${hours12}:${mins.toString().padStart(2, "0")} ${period}`
}

function clampOffset(offset: number, durationSeconds: number) {
  return Math.max(0, Math.min(offset, durationSeconds))
}

function formatRangeLabel(start: number, end: number) {
  if (Math.abs(end - start) < 1) {
    return formatDuration(start)
  }

  return `${formatDuration(start)}–${formatDuration(end)}`
}

function severityFromOcclusion(occlusionClass?: number | null): SeverityLevel {
  if (occlusionClass === 2) return "heavy"
  if (occlusionClass === 1) return "moderate"
  if (occlusionClass === 0) return "light"
  return "neutral"
}

function severityRank(level: SeverityLevel) {
  if (level === "heavy") return 3
  if (level === "moderate") return 2
  if (level === "light") return 1
  return 0
}

export function PlaybackTimeline({
  startTime,
  endTime,
  durationSeconds,
  currentTimeSeconds,
  events,
  severityBuckets: backendSeverityBuckets = [],
  searchMatchOffsets = [],
  onSeek,
}: PlaybackTimelineProps) {
  const safeDuration = Number.isFinite(durationSeconds) && durationSeconds > 0 ? durationSeconds : 0
  const safeCurrentTime = Math.max(0, Math.min(currentTimeSeconds, safeDuration || currentTimeSeconds))

  const timedEvents = useMemo(
    () =>
      events
        .filter((event): event is EventRecord & { offsetSeconds: number } => typeof event.offsetSeconds === "number")
        .map((event) => ({
          ...event,
          offsetSeconds: clampOffset(event.offsetSeconds, safeDuration || event.offsetSeconds),
        }))
        .sort((left, right) => left.offsetSeconds - right.offsetSeconds),
    [events, safeDuration],
  )

  const summarizedSeverityBuckets = useMemo(() => {
    if (!safeDuration || backendSeverityBuckets.length === 0) return []

    return backendSeverityBuckets
      .map((bucket) => {
        const startOffset = clampOffset(bucket.startOffsetSeconds, safeDuration)
        const endOffset = clampOffset(bucket.endOffsetSeconds, safeDuration)
        if (endOffset <= startOffset) return null

        const scoreLabel = typeof bucket.score === "number" ? ` • score ${bucket.score.toFixed(1)}` : ""
        return {
          left: (startOffset / safeDuration) * 100,
          width: ((endOffset - startOffset) / safeDuration) * 100,
          severity: bucket.severity,
          title: `${SEVERITY_STYLES[bucket.severity].label}${scoreLabel} • ${formatRangeLabel(startOffset, endOffset)}`,
        }
      })
      .filter((bucket): bucket is { left: number; width: number; severity: SeverityLevel; title: string } => bucket !== null)
  }, [backendSeverityBuckets, safeDuration])

  const fallbackSeverityBuckets = useMemo(() => {
    if (!safeDuration) return []

    const classifiedEvents = timedEvents.filter(
      (event) => event.occlusionClass === 0 || event.occlusionClass === 1 || event.occlusionClass === 2,
    )

    if (classifiedEvents.length === 0) {
      return [
        {
          left: 0,
          width: 100,
          severity: "neutral" as const,
          title: `No LOS samples • ${formatRangeLabel(0, safeDuration)}`,
        },
      ]
    }

    const rawSegments = classifiedEvents
      .map((event, index) => {
        const previous = classifiedEvents[index - 1]
        const next = classifiedEvents[index + 1]
        const startOffset = previous ? (previous.offsetSeconds + event.offsetSeconds) / 2 : 0
        const endOffset = next ? (event.offsetSeconds + next.offsetSeconds) / 2 : safeDuration

        return {
          startOffset: clampOffset(startOffset, safeDuration),
          endOffset: clampOffset(endOffset, safeDuration),
          severity: severityFromOcclusion(event.occlusionClass),
        }
      })
      .filter((segment) => segment.endOffset > segment.startOffset)

    const mergedSegments: Array<{ startOffset: number; endOffset: number; severity: SeverityLevel }> = []

    for (const segment of rawSegments) {
      const previous = mergedSegments[mergedSegments.length - 1]

      if (previous && previous.severity === segment.severity) {
        previous.endOffset = segment.endOffset
        continue
      }

      mergedSegments.push({ ...segment })
    }

    return mergedSegments.map((segment) => ({
      left: (segment.startOffset / safeDuration) * 100,
      width: ((segment.endOffset - segment.startOffset) / safeDuration) * 100,
      severity: segment.severity,
      title: `${SEVERITY_STYLES[segment.severity].label} • ${formatRangeLabel(segment.startOffset, segment.endOffset)}`,
    }))
  }, [safeDuration, timedEvents])

  const severityBuckets = summarizedSeverityBuckets.length > 0 ? summarizedSeverityBuckets : fallbackSeverityBuckets

  const clusteredMarkers = useMemo(() => {
    if (!safeDuration || timedEvents.length === 0) return []

    const mergeWindowSeconds = Math.max(2.5, safeDuration * 0.012)
    const laneGap = mergeWindowSeconds * 1.2
    const lanes = [-Infinity, -Infinity, -Infinity]
    const rawClusters: Array<{ start: number; end: number; center: number; count: number; maxSeverity: SeverityLevel }> = []

    for (const event of timedEvents) {
      const severity = severityFromOcclusion(event.occlusionClass)
      const previous = rawClusters[rawClusters.length - 1]

      if (previous && event.offsetSeconds - previous.end <= mergeWindowSeconds) {
        previous.center = (previous.center * previous.count + event.offsetSeconds) / (previous.count + 1)
        previous.end = event.offsetSeconds
        previous.count += 1
        if (severityRank(severity) > severityRank(previous.maxSeverity)) {
          previous.maxSeverity = severity
        }
        continue
      }

      rawClusters.push({
        start: event.offsetSeconds,
        end: event.offsetSeconds,
        center: event.offsetSeconds,
        count: 1,
        maxSeverity: severity,
      })
    }

    return rawClusters.map((cluster, index) => {
      let lane = lanes.findIndex((lastOffset) => cluster.center - lastOffset > laneGap)
      if (lane === -1) {
        lane = index % lanes.length
      }
      lanes[lane] = cluster.center

      return {
        ...cluster,
        lane,
        left: (cluster.center / safeDuration) * 100,
        title: `${formatRangeLabel(cluster.start, cluster.end)}\n${cluster.count} ${cluster.count === 1 ? "event" : "events"}`,
      }
    })
  }, [safeDuration, timedEvents])

  const searchClusters = useMemo(() => {
    if (!safeDuration) return []

    const offsets = Array.from(
      new Set(
        searchMatchOffsets
          .filter((offset): offset is number => Number.isFinite(offset))
          .map((offset) => clampOffset(offset, safeDuration)),
      ),
    ).sort((left, right) => left - right)

    if (offsets.length === 0) return []

    const mergeWindowSeconds = Math.max(4, safeDuration * 0.015)
    const rawClusters: Array<{ start: number; end: number; center: number; count: number }> = []

    for (const offset of offsets) {
      const previous = rawClusters[rawClusters.length - 1]
      if (previous && offset - previous.end <= mergeWindowSeconds) {
        previous.center = (previous.center * previous.count + offset) / (previous.count + 1)
        previous.end = offset
        previous.count += 1
        continue
      }

      rawClusters.push({ start: offset, end: offset, center: offset, count: 1 })
    }

    return rawClusters.map((cluster) => ({
      ...cluster,
      left: (cluster.center / safeDuration) * 100,
      widthPercent: Math.max((((cluster.end - cluster.start) || mergeWindowSeconds * 0.6) / safeDuration) * 100, 0.9),
      title: `${formatRangeLabel(cluster.start, cluster.end)}\n${cluster.count} search ${cluster.count === 1 ? "match" : "matches"}`,
    }))
  }, [safeDuration, searchMatchOffsets])

  const hasSearchMatches = searchClusters.length > 0

  const markerOffsets = useMemo(() => {
    if (!safeDuration) return []

    return Array.from({ length: 5 }, (_, index) => {
      const ratio = index / 4
      const offset = safeDuration * ratio
      return {
        id: `timeline-marker-${index}-${offset.toFixed(3)}`,
        offset,
        label: (() => {
          const startMinutes = parseClockMinutes(startTime)
          if (startMinutes === null) {
            return index === 4 ? endTime : formatDuration(offset)
          }
          return formatClock(startMinutes + offset / 60)
        })(),
      }
    })
  }, [endTime, safeDuration, startTime])

  const currentPosition = safeDuration ? (safeCurrentTime / safeDuration) * 100 : 0
  const currentWallClock = (() => {
    const startMinutes = parseClockMinutes(startTime)
    if (startMinutes === null) return formatDuration(safeCurrentTime)
    return formatClock(startMinutes + safeCurrentTime / 60)
  })()

  const handleSeek = (clientX: number, target: HTMLDivElement) => {
    if (!safeDuration || !onSeek) return
    const rect = target.getBoundingClientRect()
    const relativeX = Math.max(0, Math.min(clientX - rect.left, rect.width))
    onSeek((relativeX / rect.width) * safeDuration)
  }

  return (
    <div className="rounded-xl border border-border bg-card p-4 shadow-elevated-sm">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-foreground">Playback Timeline</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Click anywhere on the bar to jump through the recording.
            {hasSearchMatches ? " Search hits are highlighted in cyan." : ""}
          </p>
        </div>
        <div className="text-right text-xs text-muted-foreground">
          <p>{startTime} - {endTime}</p>
          <p className="mt-1 text-foreground">{currentWallClock}</p>
        </div>
      </div>

      {safeDuration > 0 ? (
        <>
          <div
            className="relative mb-2 h-14 overflow-hidden rounded-xl border border-border bg-secondary/70"
            onClick={(event) => handleSeek(event.clientX, event.currentTarget)}
          >
            {severityBuckets.map((bucket) => (
              <div
                key={`${bucket.left}-${bucket.severity}`}
                title={bucket.title}
                className="absolute inset-y-0"
                style={{ left: `${bucket.left}%`, width: `${bucket.width}%`, backgroundColor: SEVERITY_STYLES[bucket.severity].fill }}
              />
            ))}

            <div className="pointer-events-none absolute inset-y-0 left-0 bg-background/10" style={{ width: `${currentPosition}%` }} />

            {searchClusters.map((cluster) => (
              <button
                key={`search-${cluster.start}-${cluster.count}`}
                type="button"
                title={cluster.title}
                aria-label={cluster.title}
                className="absolute bottom-1 z-10 h-2 rounded-full border border-cyan-200/80 bg-cyan-400/65 shadow-[0_0_0_1px_rgba(34,211,238,0.12)]"
                style={{ left: `${cluster.left}%`, width: `${cluster.widthPercent}%`, transform: "translateX(-50%)" }}
                onClick={(event) => {
                  event.stopPropagation()
                  onSeek?.(cluster.center)
                }}
              />
            ))}

            {clusteredMarkers.map((marker) => {
              const markerStyle = marker.maxSeverity === "heavy"
                ? "bg-red-500/95"
                : marker.maxSeverity === "moderate"
                  ? "bg-amber-500/95"
                  : marker.maxSeverity === "light"
                    ? "bg-lime-500/95"
                    : "bg-emerald-500/95"

              return (
                <button
                  key={`marker-${marker.start}-${marker.end}-${marker.lane}`}
                  type="button"
                  title={marker.title}
                  aria-label={marker.title}
                  className={`absolute z-20 h-4 w-1.5 -translate-x-1/2 rounded-[2px] shadow-sm transition-transform hover:scale-y-110 ${markerStyle}`}
                  style={{
                    left: `${marker.left}%`,
                    top: `${8 + marker.lane * 12}px`,
                  }}
                  onClick={(event) => {
                    event.stopPropagation()
                    onSeek?.(marker.center)
                  }}
                />
              )
            })}

            <div className="absolute inset-y-0 z-10 w-0.5 bg-accent" style={{ left: `${currentPosition}%` }}>
              <div className="absolute -top-1 left-1/2 h-3 w-3 -translate-x-1/2 rounded-full border border-background bg-accent" />
            </div>
          </div>

          <div
            className="relative h-2 cursor-pointer rounded-full bg-secondary"
            onClick={(event) => handleSeek(event.clientX, event.currentTarget)}
          >
            <div className="h-full rounded-full bg-primary/50" style={{ width: `${currentPosition}%` }} />
          </div>

          <div className="mt-3 flex justify-between gap-2 text-[10px] text-muted-foreground">
            {markerOffsets.map((marker) => (
              <span key={marker.id}>{marker.label}</span>
            ))}
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3 text-xs text-muted-foreground">
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" />
                LOS A-B
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-lime-500" />
                LOS C
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-amber-500" />
                LOS D
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-red-500" />
                LOS E-F
              </span>
            </div>
            <span>
              {formatDuration(safeCurrentTime)} / {formatDuration(safeDuration)}
            </span>
          </div>
        </>
      ) : (
        <div className="rounded-xl border border-dashed border-border bg-secondary/40 px-4 py-6 text-sm text-muted-foreground">
          Timeline controls will activate once the video metadata loads.
        </div>
      )}
    </div>
  )
}
