"use client"

import { VideoThumbnail } from "./video-thumbnail"
import { Plus } from "lucide-react"

interface Video {
  id: string
  timestamp: string
  date: string
  startTime?: string
  endTime?: string
}

interface Location {
  id: string
  name: string
  latitude: number
  longitude: number
  videos: Video[]
}

interface VideoGridProps {
  locations: Location[]
  detectionMode: boolean
  onAddVideoClick?: () => void
}

export function VideoGrid({ locations, detectionMode, onAddVideoClick }: VideoGridProps) {
  return (
    <div className="space-y-8">
      {locations.map((location, locIndex) => (
        <div key={location.id}>
          <h2 className="text-base font-semibold text-foreground mb-4 flex items-center gap-2">
            {location.name}
            <span className="text-xs font-normal text-muted-foreground">
              ({location.videos.length} videos)
            </span>
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {/* Video Thumbnails */}
            {location.videos.map((video, videoIndex) => (
              <VideoThumbnail
                key={video.id}
                id={video.id}
                location={location.name}
                timestamp={video.timestamp}
                date={video.date}
                thumbnailIndex={(locIndex * 10) + videoIndex + 1}
                detectionMode={detectionMode}
              />
            ))}
            
            {/* Empty Placeholder Boxes */}
            {[1, 2].map((placeholder) => (
              <button
                key={`placeholder-${location.id}-${placeholder}`}
                onClick={onAddVideoClick}
                className="rounded-lg border-2 border-dashed border-border bg-muted/50 flex flex-col items-center justify-center gap-2 hover:border-primary/50 hover:bg-muted transition-all cursor-pointer group"
                style={{ aspectRatio: '16/10' }}
              >
                <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                  <Plus className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                </div>
                <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">
                  Add Video
                </span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
