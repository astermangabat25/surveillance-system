"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { FileVideo, Upload, Video, X, Zap } from "lucide-react"
import { getCountingConfigChoices, uploadInferenceRequirement } from "@/lib/api"

const LAST_COUNTING_CONFIG_STORAGE_KEY = "alive-last-counting-config"

interface AddVideoModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  locations: Array<{ id: string; name: string }>
  initialLocationId?: string
  onAddVideo?: (data: {
    file: File
    locationId: string
    date: string
    startTime: string
    endTime?: string
    manualDurationHours?: number
    manualDurationMinutes?: number
    countingConfig?: string
    fastMode: boolean
  }) => void | Promise<void>
}

function formatHumanDuration(totalSeconds: number) {
  if (totalSeconds < 60) {
    return `${totalSeconds} sec`
  }

  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  const parts = [
    hours > 0 ? `${hours} hr${hours === 1 ? "" : "s"}` : null,
    minutes > 0 ? `${minutes} min` : null,
    seconds > 0 ? `${seconds} sec` : null,
  ].filter(Boolean)

  return parts.join(" ")
}

function computeSchedule(startTime: string, durationSeconds: number | null) {
  const [hoursPart, minutesPart, secondsPart = "0"] = startTime.split(":")
  const startHours = Number(hoursPart)
  const startMinutes = Number(minutesPart)
  const startSeconds = Number(secondsPart)

  if (!startTime || !Number.isFinite(startHours) || !Number.isFinite(startMinutes) || !Number.isFinite(startSeconds) || !Number.isFinite(durationSeconds) || durationSeconds === null || durationSeconds <= 0) {
    return null
  }

  const resolvedDurationSeconds = Math.max(1, Math.round(durationSeconds))
  const totalSeconds = startHours * 3600 + startMinutes * 60 + startSeconds + resolvedDurationSeconds
  const dayOffset = Math.floor(totalSeconds / (24 * 3600))
  const endSeconds = ((totalSeconds % (24 * 3600)) + (24 * 3600)) % (24 * 3600)
  const endHours = Math.floor(endSeconds / 3600)
  const endMinuteValue = Math.floor((endSeconds % 3600) / 60)
  const endSecondValue = endSeconds % 60
  const includeSeconds = endSecondValue > 0 || startTime.split(":").length === 3 || resolvedDurationSeconds % 60 !== 0

  return {
    endTime: includeSeconds
      ? `${endHours.toString().padStart(2, "0")}:${endMinuteValue.toString().padStart(2, "0")}:${endSecondValue.toString().padStart(2, "0")}`
      : `${endHours.toString().padStart(2, "0")}:${endMinuteValue.toString().padStart(2, "0")}`,
    durationLabel: formatHumanDuration(resolvedDurationSeconds),
    dayOffset,
  }
}

const AUTO_DURATION_GUIDANCE_MESSAGE = "Couldn't auto-read video duration. Enter duration manually."

const ACCEPTED_VIDEO_MIME_TYPES = new Set(["video/mp4", "video/x-msvideo", "video/avi", "video/msvideo"])

function isAcceptedVideoFile(file: File) {
  const normalizedName = file.name.toLowerCase()
  if (normalizedName.endsWith(".mp4") || normalizedName.endsWith(".avi")) {
    return true
  }

  return ACCEPTED_VIDEO_MIME_TYPES.has(file.type.toLowerCase())
}

function readVideoDuration(file: File) {
  return new Promise<number>((resolve, reject) => {
    const video = document.createElement("video")
    const objectUrl = URL.createObjectURL(file)
    const timeoutMs = 7000
    const seekProbeTime = 10_000_000
    let hasSettled = false
    let usedSeekWorkaround = false
    let timeoutId: ReturnType<typeof setTimeout> | null = null

    const removeListeners = () => {
      video.removeEventListener("loadedmetadata", handleDurationCandidates)
      video.removeEventListener("durationchange", handleDurationCandidates)
      video.removeEventListener("timeupdate", handleSeekResolution)
      video.removeEventListener("seeked", handleSeekResolution)
      video.removeEventListener("error", handleError)
    }

    const cleanup = () => {
      if (timeoutId) {
        clearTimeout(timeoutId)
        timeoutId = null
      }
      removeListeners()
      URL.revokeObjectURL(objectUrl)
      video.removeAttribute("src")
      video.load()
    }

    const settle = (callback: () => void) => {
      if (hasSettled) {
        return
      }
      hasSettled = true
      callback()
      cleanup()
    }

    const resolveIfFiniteDuration = () => {
      const duration = Number(video.duration)
      if (!Number.isFinite(duration) || duration <= 0) {
        return false
      }

      if (usedSeekWorkaround) {
        try {
          video.currentTime = 0
        } catch {
          // Ignore reset errors; duration has already been resolved.
        }
      }

      settle(() => resolve(duration))
      return true
    }

    const attemptSeekWorkaround = () => {
      if (usedSeekWorkaround) {
        return
      }

      usedSeekWorkaround = true
      try {
        video.currentTime = seekProbeTime
      } catch {
        settle(() => reject(new Error("Failed to seek video for duration probing.")))
      }
    }

    function handleDurationCandidates() {
      if (resolveIfFiniteDuration()) {
        return
      }

      const duration = Number(video.duration)
      if (!Number.isFinite(duration) || Number.isNaN(duration) || duration === Infinity) {
        attemptSeekWorkaround()
      }
    }

    function handleSeekResolution() {
      resolveIfFiniteDuration()
    }

    function handleError() {
      settle(() => reject(new Error("Could not read the selected video's duration.")))
    }

    video.preload = "metadata"
    video.addEventListener("loadedmetadata", handleDurationCandidates)
    video.addEventListener("durationchange", handleDurationCandidates)
    video.addEventListener("timeupdate", handleSeekResolution)
    video.addEventListener("seeked", handleSeekResolution)
    video.addEventListener("error", handleError)

    timeoutId = setTimeout(() => {
      settle(() => reject(new Error("Timed out while reading video duration.")))
    }, timeoutMs)

    video.src = objectUrl
  })
}

export function AddVideoModal({ open, onOpenChange, locations, initialLocationId, onAddVideo }: AddVideoModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [locationId, setLocationId] = useState("")
  const [date, setDate] = useState("")
  const [startTime, setStartTime] = useState("")
  const [manualDurationHours, setManualDurationHours] = useState("0")
  const [manualDurationMinutes, setManualDurationMinutes] = useState("0")
  const [detectedDurationSeconds, setDetectedDurationSeconds] = useState<number | null>(null)
  const [durationError, setDurationError] = useState<string | null>(null)
  const [countingOptions, setCountingOptions] = useState<string[]>([])
  const [selectedCountingConfig, setSelectedCountingConfig] = useState("")
  const [countingConfigError, setCountingConfigError] = useState<string | null>(null)
  const [isLoadingCountingOptions, setIsLoadingCountingOptions] = useState(false)
  const [isUploadingCountingConfig, setIsUploadingCountingConfig] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [isDetectingDuration, setIsDetectingDuration] = useState(false)
  const [fastMode, setFastMode] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const countingConfigInputRef = useRef<HTMLInputElement>(null)

  const selectedManualDurationSeconds = useMemo(() => {
    const parsedHours = Number(manualDurationHours)
    const parsedMinutes = Number(manualDurationMinutes)
    if (!Number.isFinite(parsedHours) || !Number.isFinite(parsedMinutes)) {
      return null
    }

    const normalizedHours = Math.max(0, Math.floor(parsedHours))
    const normalizedMinutes = Math.max(0, Math.floor(parsedMinutes))
    const totalSeconds = (normalizedHours * 3600) + (normalizedMinutes * 60)
    return totalSeconds > 0 ? totalSeconds : null
  }, [manualDurationHours, manualDurationMinutes])

  const effectiveDurationSeconds = detectedDurationSeconds ?? selectedManualDurationSeconds

  const refreshCountingOptions = useCallback(async () => {
    setIsLoadingCountingOptions(true)
    setCountingConfigError(null)

    try {
      const response = await getCountingConfigChoices()
      const nextOptions = response.options ?? []
      const savedSelection = typeof window !== "undefined" ? window.localStorage.getItem(LAST_COUNTING_CONFIG_STORAGE_KEY) : null
      const fallbackSelection = response.defaultConfig && nextOptions.includes(response.defaultConfig) ? response.defaultConfig : nextOptions[0] ?? ""
      const resolvedSelection = savedSelection && nextOptions.includes(savedSelection) ? savedSelection : fallbackSelection

      setCountingOptions(nextOptions)
      setSelectedCountingConfig((current) => {
        if (current && nextOptions.includes(current)) {
          return current
        }
        return resolvedSelection
      })
    } catch (error) {
      setCountingConfigError(error instanceof Error ? error.message : "Failed to load counting configs.")
      setCountingOptions([])
      setSelectedCountingConfig("")
    } finally {
      setIsLoadingCountingOptions(false)
    }
  }, [])

  useEffect(() => {
    if (open) {
      setLocationId(initialLocationId ?? "")
      setSubmitError(null)
      void refreshCountingOptions()
    }
  }, [initialLocationId, open, refreshCountingOptions])

  useEffect(() => {
    if (!selectedFile) {
      setDetectedDurationSeconds(null)
      setDurationError(null)
      setIsDetectingDuration(false)
      return
    }

    let isCancelled = false
    setIsDetectingDuration(true)
    setDurationError(null)

    void readVideoDuration(selectedFile)
      .then((durationSeconds) => {
        if (isCancelled) return
        const roundedSeconds = Math.max(1, Math.round(durationSeconds))
        setDetectedDurationSeconds(roundedSeconds)
      })
      .catch((error) => {
        if (isCancelled) return
        setDetectedDurationSeconds(null)
        setDurationError(error instanceof Error ? AUTO_DURATION_GUIDANCE_MESSAGE : AUTO_DURATION_GUIDANCE_MESSAGE)
      })
      .finally(() => {
        if (!isCancelled) {
          setIsDetectingDuration(false)
        }
      })

    return () => {
      isCancelled = true
    }
  }, [selectedFile])

  useEffect(() => {
    if (selectedCountingConfig && typeof window !== "undefined") {
      window.localStorage.setItem(LAST_COUNTING_CONFIG_STORAGE_KEY, selectedCountingConfig)
    }
  }, [selectedCountingConfig])

  const computedSchedule = useMemo(() => computeSchedule(startTime, effectiveDurationSeconds), [effectiveDurationSeconds, startTime])

  const submitDisabledReason = useMemo(() => {
    if (isSubmitting) {
      return "Adding video to the queue..."
    }

    if (isDetectingDuration) {
      return "Reading the selected video before enabling upload..."
    }

    if (!selectedFile) {
      return "Select a video file to continue."
    }

    if (!locationId) {
      return "Choose a location to continue."
    }

    if (!date) {
      return "Choose a start date to continue."
    }

    if (!startTime) {
      return "Choose a start time to continue."
    }

    if (!selectedCountingConfig) {
      return "Choose a counting config to continue."
    }

    if (detectedDurationSeconds === null && selectedManualDurationSeconds === null) {
      return "Auto duration failed. Enter manual duration in hours and minutes to continue."
    }

    if (!computedSchedule) {
      return "Enter a valid start time and duration to continue."
    }

    return null
  }, [computedSchedule, date, detectedDurationSeconds, isDetectingDuration, isSubmitting, locationId, selectedCountingConfig, selectedManualDurationSeconds, selectedFile, startTime])

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    setSubmitError(null)

    if (e.dataTransfer.files?.[0]) {
      const file = e.dataTransfer.files[0]
      if (isAcceptedVideoFile(file)) {
        setSelectedFile(file)
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setSubmitError(null)
      setSelectedFile(e.target.files[0])
      e.target.value = ""
    }
  }

  const handleCountingConfigUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) {
      return
    }

    if (!file.name.toLowerCase().endsWith(".json")) {
      setCountingConfigError("Counting config upload must be a .json file.")
      return
    }

    setIsUploadingCountingConfig(true)
    setCountingConfigError(null)

    try {
      const result = await uploadInferenceRequirement({
        file,
        requirementType: "counting-config",
      })
      await refreshCountingOptions()
      setSelectedCountingConfig(result.filename)
    } catch (error) {
      setCountingConfigError(error instanceof Error ? error.message : "Failed to upload counting config.")
    } finally {
      setIsUploadingCountingConfig(false)
    }
  }

  const handleSubmit = async () => {
    if (!selectedFile || !locationId || !date || !startTime || !computedSchedule || !selectedCountingConfig || isSubmitting) return

    const manualHours = detectedDurationSeconds === null ? Math.max(0, Math.floor(Number(manualDurationHours) || 0)) : undefined
    const manualMinutes = detectedDurationSeconds === null ? Math.max(0, Math.floor(Number(manualDurationMinutes) || 0)) : undefined

    setSubmitError(null)
    setIsSubmitting(true)

    try {
      await onAddVideo?.({
        file: selectedFile,
        locationId,
        date,
        startTime,
        endTime: computedSchedule.endTime,
        manualDurationHours: manualHours,
        manualDurationMinutes: manualMinutes,
        countingConfig: selectedCountingConfig,
        fastMode,
      })
      handleClose()
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Failed to upload video.")
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    setSelectedFile(null)
    setLocationId("")
    setDate("")
    setStartTime("")
    setManualDurationHours("0")
    setManualDurationMinutes("0")
    setDetectedDurationSeconds(null)
    setDurationError(null)
    setCountingConfigError(null)
    setSubmitError(null)
    setIsDetectingDuration(false)
    setIsLoadingCountingOptions(false)
    setIsUploadingCountingConfig(false)
    setFastMode(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    if (countingConfigInputRef.current) {
      countingConfigInputRef.current.value = ""
    }
    onOpenChange(false)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          if (isSubmitting) {
            return
          }

          handleClose()
        } else {
          onOpenChange(nextOpen)
        }
      }}
    >
      <DialogContent
        showCloseButton={!isSubmitting}
        className="bg-card border-border sm:max-w-md"
        onEscapeKeyDown={(event) => {
          if (isSubmitting) {
            event.preventDefault()
          }
        }}
        onInteractOutside={(event) => {
          if (isSubmitting) {
            event.preventDefault()
          }
        }}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-foreground">
            <Video className="w-5 h-5" />
            Add New Video
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Upload a video file, choose the start time, and let the system calculate the end time from the duration.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          {/* File Upload Area */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Video File</label>
            <div
              className={`relative border-2 border-dashed rounded-lg p-6 transition-colors ${
                dragActive 
                  ? "border-primary bg-primary/5" 
                    : selectedFile
                    ? "border-primary/50 bg-muted/50" 
                    : "border-border hover:border-primary/50"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.avi,video/mp4,video/x-msvideo"
                onChange={handleFileChange}
                className="hidden"
              />

              {selectedFile ? (
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <FileVideo className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">{selectedFile.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                    {isDetectingDuration ? (
                      <p className="text-xs text-primary">Reading duration…</p>
                    ) : detectedDurationSeconds !== null ? (
                      <p className="text-xs text-primary">Detected duration: {formatHumanDuration(detectedDurationSeconds)}</p>
                    ) : durationError ? (
                      <p className="text-xs text-amber-400">{durationError}</p>
                    ) : null}
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setSelectedFile(null)}
                    disabled={isSubmitting}
                    className="h-8 w-8"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              ) : (
                <div 
                  className="flex flex-col items-center gap-2 cursor-pointer"
                  onClick={() => !isSubmitting && fileInputRef.current?.click()}
                >
                  <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
                    <Upload className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-foreground">Drop video here or click to upload</p>
                    <p className="text-xs text-muted-foreground mt-1">Supports MP4 and AVI formats</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Location Dropdown */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Location</label>
            <Select value={locationId} onValueChange={(value) => {
              setSubmitError(null)
              setLocationId(value)
            }}>
              <SelectTrigger className="bg-secondary border-border text-foreground">
                <SelectValue placeholder="Select location" />
              </SelectTrigger>
              <SelectContent className="bg-card border-border">
                {locations.map((loc) => (
                  <SelectItem key={loc.id} value={loc.id} className="text-foreground">
                    {loc.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <label className="text-sm font-medium text-foreground">Counting Config</label>
              <input
                ref={countingConfigInputRef}
                type="file"
                accept=".json,application/json"
                onChange={(event) => {
                  void handleCountingConfigUpload(event)
                }}
                className="hidden"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="border-border"
                disabled={isSubmitting || isUploadingCountingConfig}
                onClick={() => countingConfigInputRef.current?.click()}
              >
                {isUploadingCountingConfig ? "Uploading..." : "Upload New"}
              </Button>
            </div>
            <Select
              value={selectedCountingConfig}
              onValueChange={(value) => {
                setSubmitError(null)
                setCountingConfigError(null)
                setSelectedCountingConfig(value)
              }}
              disabled={isLoadingCountingOptions || isSubmitting}
            >
              <SelectTrigger className="bg-secondary border-border text-foreground">
                <SelectValue placeholder={isLoadingCountingOptions ? "Loading counting configs..." : "Select counting config"} />
              </SelectTrigger>
              <SelectContent className="bg-card border-border">
                {countingOptions.map((configName) => (
                  <SelectItem key={configName} value={configName} className="text-foreground">
                    {configName}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Choose an existing counting-line config or upload a new .json file.
            </p>
            {countingConfigError ? <p className="text-xs text-destructive">{countingConfigError}</p> : null}
          </div>

          {/* Date */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Start Date</label>
            <Input 
              type="date" 
              value={date}
              onChange={(e) => {
                setSubmitError(null)
                setDate(e.target.value)
              }}
              className="bg-secondary border-border text-foreground"
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Start Time</label>
              <Input 
                type="time" 
                step="1"
                value={startTime}
                onChange={(e) => {
                  setSubmitError(null)
                  setStartTime(e.target.value)
                }}
                className="bg-secondary border-border text-foreground"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Detected Duration</label>
              <div className="rounded-xl border border-border bg-secondary px-3 py-2 text-sm text-foreground">
                {isDetectingDuration
                  ? "Reading duration..."
                  : detectedDurationSeconds !== null
                    ? formatHumanDuration(detectedDurationSeconds)
                    : "Unavailable"}
              </div>
              <p className="text-xs text-muted-foreground">
                {isDetectingDuration
                  ? "Reading the uploaded file to auto-detect duration..."
                  : detectedDurationSeconds !== null
                    ? "Duration is auto-detected from video metadata."
                    : durationError ?? "Upload a file to auto-detect duration."}
              </p>
            </div>
          </div>

          {detectedDurationSeconds === null && !isDetectingDuration ? (
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Manual Duration (Hours)</label>
                <Input
                  type="number"
                  min="0"
                  step="1"
                  inputMode="numeric"
                  value={manualDurationHours}
                  onChange={(event) => {
                    setSubmitError(null)
                    setManualDurationHours(event.target.value)
                  }}
                  className="bg-secondary border-border text-foreground"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Manual Duration (Minutes)</label>
                <Input
                  type="number"
                  min="0"
                  step="1"
                  inputMode="numeric"
                  value={manualDurationMinutes}
                  onChange={(event) => {
                    setSubmitError(null)
                    setManualDurationMinutes(event.target.value)
                  }}
                  className="bg-secondary border-border text-foreground"
                />
              </div>
            </div>
          ) : null}

          <div className="rounded-xl border border-border bg-secondary/40 p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-foreground">Scheduled coverage</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {computedSchedule
                    ? `Starts at ${startTime} and ends at ${computedSchedule.endTime}${computedSchedule.dayOffset > 0 ? ` (+${computedSchedule.dayOffset} day)` : ""}. Total duration: ${computedSchedule.durationLabel}.`
                    : "Enter a valid start time and duration to preview the coverage window."}
                </p>
              </div>
              {computedSchedule && <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary">Auto end time</span>}
            </div>
          </div>

          <div className="rounded-xl border border-border bg-secondary/40 p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <Zap className="h-4 w-4 text-primary" />
                  Fast Mode
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Speeds up local testing by skipping some frames and using a smaller inference size.
                </p>
              </div>
              <Switch checked={fastMode} onCheckedChange={setFastMode} disabled={isSubmitting} className="data-[state=checked]:bg-primary" />
            </div>
          </div>
        </div>

        {submitError && (
          <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {submitError}
          </div>
        )}

        {!submitError && submitDisabledReason && (
          <p className="text-xs text-muted-foreground">
            {submitDisabledReason}
          </p>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={handleClose} disabled={isSubmitting} className="border-border text-foreground">
            Cancel
          </Button>
          <Button 
            type="button"
            onClick={() => void handleSubmit()}
            disabled={
              !selectedFile ||
              !locationId ||
              !date ||
              !startTime ||
              !computedSchedule ||
              !selectedCountingConfig ||
              isSubmitting ||
              isDetectingDuration
            }
            className="bg-primary text-primary-foreground hover:bg-primary/90"
          >
            {isSubmitting ? "Adding video..." : isDetectingDuration ? "Reading file..." : "Add Video"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
