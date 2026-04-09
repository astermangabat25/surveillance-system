"use client"

import { useEffect, useRef } from "react"
import { AlertCircle, MapPin, Users } from "lucide-react"
import type { ROIConfiguration } from "@/lib/api"

interface VideoPlayerProps {
  videoId: string
  location: string
  src?: string | null
  pedestrianCount: number
  timestamp: string
  date: string
  isProcessed: boolean
  videoRef?: { current: HTMLVideoElement | null }
  requestedSeek?: { seconds: number; token: number } | null
  roiCoordinates?: ROIConfiguration | null
  showROI?: boolean
  onTimeUpdate?: (seconds: number) => void
  onDurationChange?: (seconds: number) => void
}

function applySeek(video: HTMLVideoElement, seconds: number) {
  const nextTime = Number.isFinite(video.duration) ? Math.min(Math.max(seconds, 0), video.duration) : Math.max(seconds, 0)
  video.currentTime = nextTime
  return nextTime
}

export function VideoPlayer({
  videoId,
  location,
  src,
  pedestrianCount,
  timestamp,
  date,
  isProcessed,
  videoRef,
  requestedSeek,
  roiCoordinates,
  showROI = false,
  onTimeUpdate,
  onDurationChange,
}: VideoPlayerProps) {
  const fallbackRef = useRef<HTMLVideoElement | null>(null)
  const resolvedRef = videoRef ?? fallbackRef
  const roiPolygons = roiCoordinates?.includePolygonsNorm ?? []

  useEffect(() => {
    if (!src) {
      onDurationChange?.(0)
      onTimeUpdate?.(0)
    }
  }, [onDurationChange, onTimeUpdate, src])

  useEffect(() => {
    if (!requestedSeek || !resolvedRef.current) {
      return
    }

    const video = resolvedRef.current
    const seekToRequestedTime = () => onTimeUpdate?.(applySeek(video, requestedSeek.seconds))

    if (video.readyState >= 1) {
      seekToRequestedTime()
      return
    }

    video.addEventListener("loadedmetadata", seekToRequestedTime, { once: true })
    return () => video.removeEventListener("loadedmetadata", seekToRequestedTime)
  }, [requestedSeek, resolvedRef, src])

  useEffect(() => {
    const video = resolvedRef.current
    if (!video || !src) {
      return
    }

    let frameId: number | null = null

    const publishCurrentTime = () => onTimeUpdate?.(video.currentTime)
    const publishDuration = () => onDurationChange?.(Number.isFinite(video.duration) ? video.duration : 0)
    const stopFrameLoop = () => {
      if (frameId !== null) {
        cancelAnimationFrame(frameId)
        frameId = null
      }
    }
    const frameLoop = () => {
      publishCurrentTime()
      if (!video.paused && !video.ended) {
        frameId = requestAnimationFrame(frameLoop)
      } else {
        frameId = null
      }
    }
    const startFrameLoop = () => {
      stopFrameLoop()
      frameLoop()
    }
    const handleLoadedMetadata = () => {
      publishDuration()
      publishCurrentTime()
    }

    video.addEventListener("loadedmetadata", handleLoadedMetadata)
    video.addEventListener("play", startFrameLoop)
    video.addEventListener("pause", publishCurrentTime)
    video.addEventListener("ended", publishCurrentTime)
    video.addEventListener("seeking", publishCurrentTime)
    video.addEventListener("seeked", publishCurrentTime)

    publishDuration()
    publishCurrentTime()
    if (!video.paused && !video.ended) {
      startFrameLoop()
    }

    return () => {
      stopFrameLoop()
      video.removeEventListener("loadedmetadata", handleLoadedMetadata)
      video.removeEventListener("play", startFrameLoop)
      video.removeEventListener("pause", publishCurrentTime)
      video.removeEventListener("ended", publishCurrentTime)
      video.removeEventListener("seeking", publishCurrentTime)
      video.removeEventListener("seeked", publishCurrentTime)
    }
  }, [onDurationChange, onTimeUpdate, resolvedRef, src])

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card shadow-elevated-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className={`rounded-full px-2.5 py-1 font-medium ${isProcessed ? "bg-primary/15 text-primary" : src ? "bg-accent/15 text-accent" : "bg-muted text-muted-foreground"}`}>
            {isProcessed ? "Annotated output" : src ? "Original upload" : "No media"}
          </span>
          <span className="rounded-full bg-secondary px-2.5 py-1 text-muted-foreground">Feed #{videoId}</span>
        </div>
        <span className="text-xs text-muted-foreground">
          {date} • {timestamp}
        </span>
      </div>

      {src ? (
        <div className="bg-black">
          <div className="relative aspect-video w-full bg-black">
            <video
              key={src}
              ref={resolvedRef}
              src={src}
              controls
              playsInline
              preload="metadata"
              className="aspect-video w-full bg-black"
            />
            {showROI && roiPolygons.length > 0 && (
              <svg
                aria-hidden="true"
                viewBox="0 0 1 1"
                preserveAspectRatio="none"
                className="pointer-events-none absolute inset-0 z-10 h-full w-full"
              >
                {roiPolygons.map((polygon, index) => (
                  <polygon
                    key={`roi-${index}`}
                    points={polygon.map(([x, y]) => `${x},${y}`).join(" ")}
                    fill="none"
                    stroke="#39FF14"
                    strokeWidth={0.004}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    style={{ filter: "drop-shadow(0 0 6px rgba(57, 255, 20, 0.9))" }}
                  />
                ))}
              </svg>
            )}
          </div>
        </div>
      ) : (
        <div className="flex aspect-video items-center justify-center gap-3 bg-secondary px-6 text-center text-muted-foreground">
          <AlertCircle className="h-5 w-5 shrink-0" />
          <p className="text-sm">No uploaded media is available for this video yet.</p>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 px-4 py-3 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-2.5 py-1">
          <MapPin className="h-3.5 w-3.5" />
          {location}
        </span>
        <span className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-2.5 py-1">
          <Users className="h-3.5 w-3.5" />
          {pedestrianCount} pedestrians
        </span>
      </div>
    </div>
  )
}
