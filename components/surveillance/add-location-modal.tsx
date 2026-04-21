"use client"

import { useState, useEffect, useCallback } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Info, Loader2, MapPin, Search } from "lucide-react"
import { searchLocations, type LocationPayload, type ROIConfiguration } from "@/lib/api"

const ROI_PROMPT_TEMPLATE = `I am going to paste CVAT polygon coordinates for one camera/location. Please convert them into the exact JSON format required by my application.

Rules:
1. Normalize x by image width.
2. Normalize y by image height.
3. Keep 6 decimal places.
4. Output valid JSON only.
5. Do not add explanations, markdown, comments, or extra text.
6. Treat all polygons I provide as walkable include polygons unless I explicitly say otherwise.
7. The output format must be exactly:

{
  "referenceSize": [WIDTH, HEIGHT],
  "includePolygonsNorm": [
    [[x1, y1], [x2, y2], [x3, y3]],
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  ]
}

Image size:
WIDTH = 1920
HEIGHT = 1080

Here are the raw CVAT coordinates:
[PASTE CVAT POLYGON COORDINATES HERE]`

const SEARCHABLE_LOCATIONS: Array<{
  aliases: string[]
  lat: number
  lng: number
  address: string
}> = [
  {
    aliases: ["edsa sec walk", "edsa sec walkway", "xavier hall", "xavier"],
    lat: 14.6358,
    lng: 121.07469,
    address: "Gate 2, Ateneo de Manila University",
  },
  {
    aliases: ["kostka walk", "kostka walkway", "kostka hall", "kostka"],
    lat: 14.63667,
    lng: 121.07472,
    address: "Gate 2.5, Ateneo de Manila University",
  },
  {
    aliases: ["gate 1", "gate 1 walkway", "gate one"],
    lat: 14.6354,
    lng: 121.0747,
    address: "Gate 1 Walkway, Ateneo de Manila University",
  },
  {
    aliases: ["gate 3", "gate 3 walkway", "gate three"],
    lat: 14.64028,
    lng: 121.07472,
    address: "Gate 3 Walkway, Ateneo de Manila University",
  },
  {
    aliases: ["manila"],
    lat: 14.5995,
    lng: 120.9842,
    address: "Manila, Philippines",
  },
  {
    aliases: ["tokyo"],
    lat: 35.6762,
    lng: 139.6503,
    address: "Tokyo, Japan",
  },
]

function normalizeSearchQuery(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim()
}

function findFallbackMatch(normalizedQuery: string) {
  return SEARCHABLE_LOCATIONS.find(({ aliases }) => aliases.some((alias) => normalizedQuery.includes(alias) || alias.includes(normalizedQuery)))
}

function parseROIConfiguration(value: string): ROIConfiguration | null {
  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    throw new Error("ROI JSON must be valid JSON.")
  }

  if (!parsed || typeof parsed !== "object") {
    throw new Error("ROI JSON must be an object.")
  }

  const record = parsed as Record<string, unknown>
  const referenceSize = record.referenceSize
  const includePolygonsNorm = record.includePolygonsNorm

  if (
    !Array.isArray(referenceSize) ||
    referenceSize.length !== 2 ||
    referenceSize.some((entry) => typeof entry !== "number" || !Number.isFinite(entry) || entry <= 0)
  ) {
    throw new Error("ROI JSON must include a numeric referenceSize like [1920, 1080].")
  }

  if (!Array.isArray(includePolygonsNorm) || includePolygonsNorm.length === 0) {
    throw new Error("ROI JSON must include at least one polygon in includePolygonsNorm.")
  }

  const polygons = includePolygonsNorm.map((polygon) => {
    if (!Array.isArray(polygon) || polygon.length < 3) {
      throw new Error("Each ROI polygon must contain at least 3 normalized points.")
    }

    return polygon.map((point) => {
      if (!Array.isArray(point) || point.length !== 2) {
        throw new Error("Each ROI point must be a two-number array like [0.25, 0.98].")
      }

      const [x, y] = point
      if (
        typeof x !== "number" ||
        typeof y !== "number" ||
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        x < 0 ||
        x > 1 ||
        y < 0 ||
        y > 1
      ) {
        throw new Error("ROI points must use normalized coordinates between 0 and 1.")
      }

      return [x, y] as [number, number]
    })
  })

  return {
    referenceSize: [referenceSize[0] as number, referenceSize[1] as number],
    includePolygonsNorm: polygons,
  }
}

interface AddLocationModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialData?: LocationPayload | null
  onSubmitLocation?: (data: LocationPayload) => void | Promise<void>
}

export function AddLocationModal({ open, onOpenChange, initialData = null, onSubmitLocation }: AddLocationModalProps) {
  const [name, setName] = useState("")
  const [latitude, setLatitude] = useState("")
  const [longitude, setLongitude] = useState("")
  const [description, setDescription] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const [address, setAddress] = useState("")
  const [walkableAreaM2, setWalkableAreaM2] = useState("")
  const [roadLengthM, setRoadLengthM] = useState("")
  const [laneCount, setLaneCount] = useState("")
  const [roiCoordinatesText, setRoiCoordinatesText] = useState("")
  const [searchError, setSearchError] = useState<string | null>(null)
  const [roiError, setRoiError] = useState<string | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const isEditing = Boolean(initialData)

  const resetForm = useCallback((nextData?: LocationPayload | null) => {
    setName(nextData?.name ?? "")
    setLatitude(nextData ? nextData.latitude.toString() : "")
    setLongitude(nextData ? nextData.longitude.toString() : "")
    setDescription(nextData?.description ?? "")
    setSearchQuery(nextData?.address ?? nextData?.name ?? "")
    setAddress(nextData?.address ?? "")
    setWalkableAreaM2(nextData?.walkableAreaM2 != null ? nextData.walkableAreaM2.toString() : "")
    setRoadLengthM(nextData?.roadLengthM != null ? nextData.roadLengthM.toString() : "")
    setLaneCount(nextData?.laneCount != null ? nextData.laneCount.toString() : "")
    setRoiCoordinatesText(nextData?.roiCoordinates ? JSON.stringify(nextData.roiCoordinates, null, 2) : "")
    setSearchError(null)
    setRoiError(null)
  }, [])

  useEffect(() => {
    if (!open) return
    resetForm(initialData)
  }, [initialData, open, resetForm])

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim() || isSearching) return

    setIsSearching(true)
    setSearchError(null)

    const normalizedQuery = normalizeSearchQuery(searchQuery)
    const fallbackMatch = findFallbackMatch(normalizedQuery)
    let foundResult = false
    let remoteSearchError: string | null = null

    try {
      const results = await searchLocations(searchQuery.trim())
      const firstResult = results[0]

      if (firstResult) {
        setLatitude(firstResult.latitude.toString())
        setLongitude(firstResult.longitude.toString())
        setAddress(firstResult.address)
        if (!name.trim()) {
          setName(firstResult.name || searchQuery.trim())
        }
        foundResult = true
      }
    } catch (error) {
      remoteSearchError = error instanceof Error ? error.message : "Location search is unavailable right now."
    } finally {
      if (!foundResult && fallbackMatch) {
        setLatitude(fallbackMatch.lat.toString())
        setLongitude(fallbackMatch.lng.toString())
        setAddress(fallbackMatch.address)
        if (!name.trim()) {
          setName(searchQuery.trim())
        }
        foundResult = true
      }

      if (!foundResult) {
        setSearchError(remoteSearchError ?? "Location not found. Try a more specific name or enter the coordinates manually.")
      }

      setIsSearching(false)
    }
  }, [isSearching, name, searchQuery])

  const handleSubmit = async () => {
    if (!name || !latitude || !longitude || isSubmitting) return

    let parsedROI: ROIConfiguration | null = null
    let parsedWalkableArea: number | null = null
    let parsedRoadLengthM: number | null = null
    let parsedLaneCount: number | null = null

    try {
      parsedROI = parseROIConfiguration(roiCoordinatesText)
      setRoiError(null)

      if (walkableAreaM2.trim()) {
        parsedWalkableArea = Number.parseFloat(walkableAreaM2)
        if (!Number.isFinite(parsedWalkableArea) || parsedWalkableArea <= 0) {
          throw new Error("Walkable area must be a positive number in square meters.")
        }
      }

      if (roadLengthM.trim()) {
        parsedRoadLengthM = Number.parseFloat(roadLengthM)
        if (!Number.isFinite(parsedRoadLengthM) || parsedRoadLengthM <= 0) {
          throw new Error("Road length must be a positive number in meters.")
        }
      }

      if (laneCount.trim()) {
        parsedLaneCount = Number.parseInt(laneCount, 10)
        if (!Number.isFinite(parsedLaneCount) || parsedLaneCount <= 0) {
          throw new Error("Lane count must be a positive whole number.")
        }
      }

      if ((parsedRoadLengthM == null) !== (parsedLaneCount == null)) {
        throw new Error("Provide both road length and lane count, or leave both blank.")
      }
    } catch (error) {
      setRoiError(error instanceof Error ? error.message : "Invalid ROI configuration.")
      return
    }

    setIsSubmitting(true)
    try {
      await onSubmitLocation?.({
        name,
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
        description,
        address,
        roiCoordinates: parsedROI,
        walkableAreaM2: parsedWalkableArea,
        roadLengthM: parsedRoadLengthM,
        laneCount: parsedLaneCount,
      })
      handleClose()
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    resetForm(null)
    setIsSearching(false)
    setSearchError(null)
    onOpenChange(false)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          handleClose()
        } else {
          onOpenChange(nextOpen)
        }
      }}
    >
      <DialogContent className="border-border bg-card flex max-h-[90vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-md sm:rounded-2xl">
        <DialogHeader className="shrink-0 px-6 pt-6">
          <DialogTitle className="flex items-center gap-2 text-foreground">
            <MapPin className="w-5 h-5" />
            {isEditing ? "Edit Location" : "Add New Location"}
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {isEditing
              ? "Update the location details, camera coordinates, ROI polygons, and walkable area."
              : "Search for a place or enter GPS coordinates, ROI polygons, and walkable area manually."}
          </DialogDescription>
        </DialogHeader>
        
        <div className="smooth-scrollbar min-h-0 space-y-4 overflow-y-auto overscroll-contain px-6 py-4">
          {/* Place Search */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Search Location</label>
            <div className="flex gap-2">
              <Input 
                placeholder="Search for a place..." 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault()
                    void handleSearch()
                  }
                }}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
              <Button 
                variant="outline" 
                onClick={() => void handleSearch()}
                disabled={isSearching || isSubmitting}
                className="border-border shrink-0"
              >
                {isSearching ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
              </Button>
            </div>
            {address && (
              <p className="text-xs text-muted-foreground">Found: {address}</p>
            )}
            {searchError && (
              <p className="text-xs text-destructive">{searchError}</p>
            )}
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Location Name</label>
            <Input 
              placeholder="e.g., Gate 1 Walkway" 
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Latitude</label>
              <Input 
                type="number" 
                step="0.000001"
                placeholder="40.7128" 
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Longitude</label>
              <Input 
                type="number" 
                step="0.000001"
                placeholder="-74.0060" 
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
            </div>
          </div>

          {/* Mini Map Preview */}
          {latitude && longitude && (
            <div className="rounded-lg overflow-hidden border border-border bg-muted h-32 flex items-center justify-center">
              <div className="text-center">
                <MapPin className="w-6 h-6 text-primary mx-auto mb-1" />
                <p className="text-xs text-muted-foreground">
                  {parseFloat(latitude).toFixed(4)}, {parseFloat(longitude).toFixed(4)}
                </p>
              </div>
            </div>
          )}
          
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Description (Optional)</label>
            <Input 
              placeholder="Brief description of the location" 
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Walkable Area (m²)</label>
            <Input
              type="number"
              step="0.01"
              min="0"
              placeholder="e.g., 42.5"
              value={walkableAreaM2}
              onChange={(e) => setWalkableAreaM2(e.target.value)}
              className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
            />
            <p className="text-xs text-muted-foreground">Used for the congestion part of the Pedestrian Traffic Severity Index.</p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Road Length (m)</label>
              <Input
                type="number"
                step="0.01"
                min="0"
                placeholder="e.g., 18"
                value={roadLengthM}
                onChange={(e) => setRoadLengthM(e.target.value)}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Lane Count</label>
              <Input
                type="number"
                step="1"
                min="1"
                placeholder="e.g., 2"
                value={laneCount}
                onChange={(e) => setLaneCount(e.target.value)}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">For gate LOS fallback, provide both values. They are used when walkable area is not set.</p>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-foreground">Pedestrian ROI JSON (Optional)</label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    aria-label="Show ROI JSON prompt help"
                    className="rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <Info className="h-4 w-4" />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top" align="start" sideOffset={8} className="max-w-[min(32rem,calc(100vw-2rem))] rounded-xl px-3 py-3 text-left">
                  <div className="space-y-2">
                    <p className="text-[11px] font-medium">Copy this prompt for AI ROI conversion:</p>
                    <pre className="smooth-scrollbar max-h-72 overflow-y-auto whitespace-pre-wrap rounded-lg bg-background/15 p-3 font-mono text-[10px] leading-relaxed text-background/95 select-all">
                      {ROI_PROMPT_TEMPLATE}
                    </pre>
                  </div>
                </TooltipContent>
              </Tooltip>
            </div>
            <Textarea
              placeholder='{"referenceSize":[1920,1080],"includePolygonsNorm":[[[0.24,0.98],[0.99,0.99],[0.22,0.09]]]}'
              value={roiCoordinatesText}
              onChange={(e) => setRoiCoordinatesText(e.target.value)}
              className="min-h-32 bg-secondary font-mono text-xs border-border text-foreground placeholder:text-muted-foreground"
            />
            <p className="text-xs text-muted-foreground">Only pedestrian foot-points inside these normalized polygons will count toward PTSI.</p>
            {roiError && <p className="text-xs text-destructive">{roiError}</p>}
          </div>
        </div>

        <DialogFooter className="shrink-0 px-6 pb-6 pt-2">
          <Button variant="outline" onClick={handleClose} disabled={isSubmitting} className="border-border text-foreground">
            Cancel
          </Button>
          <Button 
            onClick={() => void handleSubmit()}
            disabled={!name || !latitude || !longitude || isSubmitting}
            className="bg-primary text-primary-foreground hover:bg-primary/90"
          >
            {isSubmitting ? "Saving..." : isEditing ? "Save Changes" : "Add Location"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
