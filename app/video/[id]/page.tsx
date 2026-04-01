"use client"

import { Suspense, use, useEffect, useMemo, useRef, useState } from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { VideoPlayer } from "@/components/video/video-player"
import { VideoMetadata } from "@/components/video/video-metadata"
import { PlaybackTimeline } from "@/components/video/playback-timeline"
import { EventFeed } from "@/components/surveillance/event-feed"
import { AISearchBar } from "@/components/surveillance/ai-search-bar"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { AlertCircle, ArrowLeft, Download, Loader2, Share2, Trash2 } from "lucide-react"
import { deleteVideo, getEvents, getLocations, getMediaUrl, getVideo, getVideoPlaybackPath, type EventRecord, type LocationRecord, type VideoRecord } from "@/lib/api"

function getDetectionStatus(event: EventRecord) {
  if (event.type === "alert") return "Requires Review"
  if (event.type === "motion") return "Motion Event"
  return "Tracked"
}

function VideoDetailContent({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const router = useRouter()
  const searchParams = useSearchParams()
  const [video, setVideo] = useState<VideoRecord | null>(null)
  const [videoLocation, setVideoLocation] = useState<LocationRecord | null>(null)
  const [events, setEvents] = useState<EventRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [selectedEventId, setSelectedEventId] = useState<string | undefined>(undefined)
  const [requestedSeek, setRequestedSeek] = useState<{ seconds: number; token: number } | null>(null)
  const [currentTimeSeconds, setCurrentTimeSeconds] = useState(0)
  const [durationSeconds, setDurationSeconds] = useState(0)
  const [showAllDetections, setShowAllDetections] = useState(false)
  const [showROI, setShowROI] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const videoElementRef = useRef<HTMLVideoElement | null>(null)
  const seekTokenRef = useRef(0)
  const appliedQuerySeekRef = useRef("")

  useEffect(() => {
    let cancelled = false

    const loadVideo = async () => {
      setLoading(true)
      try {
        const [videoResponse, eventsResponse, locationsResponse] = await Promise.all([
          getVideo(id),
          getEvents(id),
          getLocations().catch(() => null),
        ])

        if (!cancelled) {
          setVideo(videoResponse)
          setVideoLocation((locationsResponse ?? []).find((location) => location.id === videoResponse.locationId) ?? null)
          setEvents(eventsResponse)
          setError(null)
          setActionError(null)
        }
      } catch (error) {
        if (!cancelled) {
          setError(error instanceof Error ? error.message : "Failed to load video details.")
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadVideo()

    return () => {
      cancelled = true
    }
  }, [id])

  useEffect(() => {
    setShowAllDetections(false)
    setShowROI(false)
    setCurrentTimeSeconds(0)
    setDurationSeconds(0)
  }, [id])

  useEffect(() => {
    const eventId = searchParams.get("eventId") ?? undefined
    setSelectedEventId(eventId)

    const seekValue = searchParams.get("seek")
    const seekKey = `${id}:${eventId ?? ""}:${seekValue ?? ""}`

    if (!seekValue || appliedQuerySeekRef.current === seekKey) {
      return
    }

    const seconds = Number(seekValue)
    if (!Number.isFinite(seconds)) {
      return
    }

    appliedQuerySeekRef.current = seekKey
    seekTokenRef.current += 1
    setRequestedSeek({ seconds, token: seekTokenRef.current })
  }, [id, searchParams])

  const orderedEvents = useMemo(
    () =>
      [...events].sort((left, right) => {
        const leftOffset = typeof left.offsetSeconds === "number" ? left.offsetSeconds : Number.POSITIVE_INFINITY
        const rightOffset = typeof right.offsetSeconds === "number" ? right.offsetSeconds : Number.POSITIVE_INFINITY
        return leftOffset - rightOffset
      }),
    [events],
  )

  const detectionDetails = useMemo(() => {
    const seen = new Set<number>()

    return orderedEvents
      .filter((event): event is EventRecord & { pedestrianId: number } => typeof event.pedestrianId === "number")
      .filter((event) => {
        if (seen.has(event.pedestrianId)) return false
        seen.add(event.pedestrianId)
        return true
      })
      .map((event) => ({ id: event.pedestrianId, status: getDetectionStatus(event) }))
  }, [orderedEvents])

  const trackedPedestriansSoFar = useMemo(() => {
    const seen = new Set<number>()

    for (const event of orderedEvents) {
      if (typeof event.pedestrianId !== "number" || typeof event.offsetSeconds !== "number") {
        continue
      }
      if (event.offsetSeconds <= currentTimeSeconds) {
        seen.add(event.pedestrianId)
      }
    }

    return seen.size
  }, [currentTimeSeconds, orderedEvents])

  const visibleDetectionDetails = showAllDetections ? detectionDetails : detectionDetails.slice(0, 15)
  const hasCollapsedDetections = detectionDetails.length > 15
  const hasLocationROI = Boolean(videoLocation?.roiCoordinates?.includePolygonsNorm?.length)

  const mediaUrl = video ? getMediaUrl(getVideoPlaybackPath(video)) : null

  const requestSeek = (seconds: number) => {
    seekTokenRef.current += 1
    setRequestedSeek({ seconds, token: seekTokenRef.current })
  }

  const handleEventSelect = (event: EventRecord) => {
    setSelectedEventId(event.id)
    setActionError(null)

    if (typeof event.offsetSeconds === "number") {
      requestSeek(event.offsetSeconds)
    }

    const params = new URLSearchParams(searchParams.toString())
    params.set("eventId", event.id)
    if (typeof event.offsetSeconds === "number") {
      params.set("seek", String(event.offsetSeconds))
    } else {
      params.delete("seek")
    }

    const query = params.toString()
    router.replace(query ? `/video/${id}?${query}` : `/video/${id}`, { scroll: false })
  }

  const handleDelete = async () => {
    if (deleting || !video) return

    const confirmed = typeof window === "undefined"
      ? false
      : window.confirm(`Delete the recording for ${video.location} on ${video.date}? This also removes the saved media files.`)

    if (!confirmed) return

    setDeleting(true)
    setActionError(null)

    try {
      await deleteVideo(video.id)
      router.push("/")
      router.refresh()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to delete this video.")
      setDeleting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        <Loader2 className="mr-2 h-6 w-6 animate-spin" />
        Loading video details...
      </div>
    )
  }

  if (error || !video) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="max-w-md rounded-3xl border border-destructive/30 bg-card p-6 text-center shadow-elevated-sm">
          <AlertCircle className="mx-auto mb-4 h-10 w-10 text-destructive" />
          <h1 className="text-xl font-semibold text-foreground">Unable to load this video</h1>
          <p className="mt-2 text-sm text-muted-foreground">{error ?? "The requested video could not be found."}</p>
          <Link href="/" className="mt-4 inline-block">
            <Button variant="outline" className="border-border text-foreground hover:bg-secondary">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to overview
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full">
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-card">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <div>
              <h1 className="text-xl font-semibold text-foreground">{video.location}</h1>
              <p className="text-sm text-muted-foreground">Video Feed #{id}</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="destructive"
              className="rounded-2xl"
              onClick={() => void handleDelete()}
              disabled={deleting}
            >
              {deleting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Trash2 className="w-4 h-4 mr-2" />}
              Delete
            </Button>
            <Button
              variant="outline"
              className="border-border text-foreground hover:bg-secondary rounded-2xl"
              onClick={() => {
                if (typeof window !== "undefined") {
                  void navigator.clipboard.writeText(window.location.href)
                }
              }}
            >
              <Share2 className="w-4 h-4 mr-2" />
              Copy Link
            </Button>
            {mediaUrl ? (
              <Button asChild className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-2xl">
                <a href={mediaUrl} download>
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </a>
              </Button>
            ) : (
              <Button className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-2xl" disabled>
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            )}
          </div>
        </header>

        {/* Video Player and Controls */}
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-5xl mx-auto space-y-6">
            {actionError && (
              <div className="flex items-start gap-3 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{actionError}</span>
              </div>
            )}

            {hasLocationROI && (
              <div className="flex items-center justify-between gap-4 rounded-2xl border border-border/70 bg-card/70 px-4 py-3 shadow-elevated-sm">
                <div>
                  <p className="text-sm font-medium text-foreground">Show ROI Outline</p>
                  <p className="text-xs text-muted-foreground">Display the stored walkable ROI polygons over the video for alignment debugging.</p>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs font-medium text-muted-foreground">{showROI ? "ON" : "OFF"}</span>
                  <Switch checked={showROI} onCheckedChange={setShowROI} aria-label="Show ROI Outline" />
                </div>
              </div>
            )}

            {/* Video Player with Bounding Boxes */}
            <VideoPlayer
              videoId={video.id}
              location={video.location}
              src={mediaUrl}
              pedestrianCount={video.pedestrianCount}
              timestamp={video.timestamp}
              date={video.date}
              isProcessed={Boolean(video.processedPath)}
              videoRef={videoElementRef}
              requestedSeek={requestedSeek}
              roiCoordinates={videoLocation?.roiCoordinates ?? null}
              showROI={showROI}
              onTimeUpdate={setCurrentTimeSeconds}
              onDurationChange={setDurationSeconds}
            />
            
            {/* Playback Timeline */}
            <PlaybackTimeline
              startTime={video.startTime}
              endTime={video.endTime}
              durationSeconds={durationSeconds}
              currentTimeSeconds={currentTimeSeconds}
              events={orderedEvents}
              onSeek={requestSeek}
            />
            
            {/* Metadata Section */}
            <VideoMetadata 
              date={video.date}
              startTime={video.startTime}
              endTime={video.endTime}
              gpsLat={video.gpsLat}
              gpsLng={video.gpsLng}
              trackedPedestriansSoFar={trackedPedestriansSoFar}
              pedestrianCount={video.pedestrianCount}
            />
          </div>
        </div>
      </div>

      {/* Right Sidebar - Filtered Event Feed */}
      <aside className="w-80 border-l border-border bg-card flex flex-col h-full">
        <AISearchBar />
        <EventFeed
          filteredVideoId={id}
          events={orderedEvents}
          loading={loading}
          selectedEventId={selectedEventId}
          onEventSelect={handleEventSelect}
        />
        
        {/* Detection Details */}
        <div className="border-t border-border p-4">
          <h4 className="text-sm font-medium text-foreground mb-3">Detection Details</h4>
          {detectionDetails.length > 0 ? (
            <div className="space-y-2">
              {visibleDetectionDetails.map((detail) => (
                <DetectionDetail key={detail.id} id={detail.id} status={detail.status} />
              ))}
              {hasCollapsedDetections && (
                <Button
                  variant="outline"
                  className="w-full rounded-2xl border-border text-foreground hover:bg-secondary"
                  onClick={() => setShowAllDetections((current) => !current)}
                >
                  {showAllDetections ? "Show less" : `View ${detectionDetails.length - 15} more`}
                </Button>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No tracked pedestrian IDs are available for this video yet.</p>
          )}
        </div>
      </aside>
    </div>
  )
}

export default function VideoDetailPage({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={
      <div className="flex h-full items-center justify-center text-muted-foreground">
        <Loader2 className="mr-2 h-6 w-6 animate-spin" />
        Loading video details...
      </div>
    }>
      <VideoDetailContent params={params} />
    </Suspense>
  )
}

function DetectionDetail({ id, status }: { id: number; status: string }) {
  const statusColor = status === "Tracked" ? "text-primary" : status === "Requires Review" ? "text-destructive" : "text-accent"
  
  return (
    <div className="flex items-center justify-between p-2 rounded-lg bg-secondary/50 border border-border">
      <span className="text-sm text-foreground">Pedestrian ID #{id}</span>
      <span className={`text-xs ${statusColor}`}>{status}</span>
    </div>
  )
}
