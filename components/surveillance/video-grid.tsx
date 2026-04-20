"use client"

import { getMediaUrl, getVideoPlaybackPath, type LocationRecord } from "@/lib/api"
import { VideoThumbnail } from "./video-thumbnail"
import { Plus } from "lucide-react"

interface VideoGridProps {
  locations: LocationRecord[]
  onAddVideoClick?: (locationId?: string) => void
}

export function VideoGrid({ locations, onAddVideoClick }: VideoGridProps) {
  return (
    <div className="space-y-8">
      {locations.map((location) => {
        const playableVideos = location.videos.filter((video) => Boolean(video.rawPath || video.processedPath))
        const hiddenVideos = location.videos.length - playableVideos.length
        const placeholderCount = playableVideos.length === 0 ? 2 : 1

        return (
          <div key={location.id}>
            <h2 className="mb-4 flex flex-wrap items-center gap-2 text-base font-semibold text-foreground">
              {location.name}
              <span className="text-xs font-normal text-muted-foreground">({playableVideos.length} playable videos)</span>
              {hiddenVideos > 0 && <span className="text-xs font-normal text-muted-foreground">• {hiddenVideos} hidden without media</span>}
            </h2>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
              {playableVideos.map((video) => (
                <VideoThumbnail
                  key={video.id}
                  id={video.id}
                  location={location.name}
                  timestamp={video.timestamp}
                  date={video.date}
                  pedestrianCount={video.pedestrianCount}
                  rawPath={video.rawPath}
                  processedPath={video.processedPath}
                  mediaUrl={getMediaUrl(getVideoPlaybackPath(video))}
                />
              ))}

              {Array.from({ length: placeholderCount }, (_, index) => (
                <button
                  key={`placeholder-${location.id}-${index}`}
                  type="button"
                  onClick={() => onAddVideoClick?.(location.id)}
                  className="group flex flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-border bg-secondary/30 transition-all hover:border-primary/50 hover:bg-secondary/50"
                  style={{ aspectRatio: "16/10" }}
                >
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-muted transition-colors group-hover:bg-primary/20">
                    <Plus className="h-5 w-5 text-muted-foreground transition-colors group-hover:text-primary" />
                  </div>
                  <div className="text-center">
                    <span className="text-xs text-muted-foreground transition-colors group-hover:text-foreground">Add Video</span>
                    <p className="mt-1 text-[11px] text-muted-foreground">Attach footage to {location.name}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}
