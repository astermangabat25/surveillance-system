"use client"

import { useCallback, useEffect, useState } from "react"
import { KPICards } from "@/components/dashboard/kpi-cards"
import { PedestrianChart } from "@/components/dashboard/pedestrian-chart"
import { OcclusionTrendsChart } from "../../components/dashboard/occlusion-trends-chart"
import { OcclusionMap } from "@/components/dashboard/occlusion-map"
import { AISynthesis } from "@/components/dashboard/ai-synthesis"
import { useUploadQueue } from "@/components/uploads/upload-queue-provider"
import { Button } from "@/components/ui/button"
import { FootageDatePicker } from "@/components/ui/footage-date-picker"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { AlertCircle, Clock, Download, FileCode, Loader2, RefreshCw, Settings2, Upload } from "lucide-react"
import {
  downloadDashboardReport,
  getAISynthesis,
  getCurrentModel,
  getDashboardLOS,
  getInferenceStatus,
  getDashboardOcclusion,
  getDashboardOcclusionTrends,
  getDashboardSummary,
  getDashboardTraffic,
  getLocations,
  uploadInferenceRequirement,
  uploadModel,
  type AISynthesisResponse,
  type DashboardSummary,
  type LocationRecord,
  type InferenceStatus,
  type InferenceRequirementType,
  type ModelInfo,
  type PTSIMapResponse,
  type PTSITrendResponse,
  type TrafficResponse,
} from "@/lib/api"

const TIME_RANGE_OPTIONS = [
  { value: "12h", label: "12 hours" },
  { value: "6h", label: "6 hours" },
  { value: "4h", label: "4 hours" },
  { value: "3h", label: "3 hours" },
  { value: "2h", label: "2 hours" },
  { value: "1h", label: "1 hour" },
  { value: "30m", label: "30 minutes" },
] as const

const START_TIME_OPTIONS = Array.from({ length: 48 }, (_, index) => {
  const totalMinutes = index * 30
  const hours = String(Math.floor(totalMinutes / 60)).padStart(2, "0")
  const minutes = String(totalMinutes % 60).padStart(2, "0")
  const value = `${hours}:${minutes}`
  return { value, label: value }
})

const getCurrentLocalDate = () => {
  const now = new Date()
  const timezoneOffsetMilliseconds = now.getTimezoneOffset() * 60 * 1000
  return new Date(now.getTime() - timezoneOffsetMilliseconds).toISOString().slice(0, 10)
}

export default function DashboardPage() {
  const { settledUploadsVersion } = useUploadQueue()
  const [selectedDate, setSelectedDate] = useState("")
  const [timeRange, setTimeRange] = useState("12h")
  const [startTime, setStartTime] = useState("00:00")
  const [hourFilter, setHourFilter] = useState("all")
  const [focusTime, setFocusTime] = useState<string | undefined>(undefined)
  const [zoomLevel, setZoomLevel] = useState(0)
  const [vehicleChartType, setVehicleChartType] = useState<"line" | "bar">("line")
  const [inOutChartType, setInOutChartType] = useState<"line" | "bar">("line")
  const [modelDialogOpen, setModelDialogOpen] = useState(false)
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [requirementFile, setRequirementFile] = useState<File | null>(null)
  const [requirementType, setRequirementType] = useState<InferenceRequirementType>("infer-config")
  const [summary, setSummary] = useState<DashboardSummary | null>(null)
  const [traffic, setTraffic] = useState<TrafficResponse | null>(null)
  const [losTraffic, setLosTraffic] = useState<TrafficResponse | null>(null)
  const [occlusionTrends, setOcclusionTrends] = useState<PTSITrendResponse | null>(null)
  const [occlusion, setOcclusion] = useState<PTSIMapResponse | null>(null)
  const [synthesis, setSynthesis] = useState<AISynthesisResponse | null>(null)
  const [locations, setLocations] = useState<LocationRecord[]>([])
  const [selectedLocationId, setSelectedLocationId] = useState<string>("")
  const [footageDates, setFootageDates] = useState<string[]>([])
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<InferenceStatus | null>(null)
  const [dashboardLoading, setDashboardLoading] = useState(true)
  const [modelLoading, setModelLoading] = useState(true)
  const [modelUploading, setModelUploading] = useState(false)
  const [requirementUploading, setRequirementUploading] = useState(false)
  const [reportExporting, setReportExporting] = useState(false)
  const [dashboardError, setDashboardError] = useState<string | null>(null)
  const [modelError, setModelError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [requirementUploadMessage, setRequirementUploadMessage] = useState<string | null>(null)

  useEffect(() => {
    setSelectedDate(getCurrentLocalDate())
  }, [])

  const loadDashboard = useCallback(async () => {
    if (!selectedDate) {
      return
    }

    setDashboardLoading(true)
    try {
      const [summaryResponse, trafficResponse, losTrafficResponse, occlusionTrendsResponse, occlusionResponse, synthesisResponse] = await Promise.all([
        getDashboardSummary(selectedDate),
        getDashboardTraffic(selectedDate, timeRange, focusTime, zoomLevel, startTime),
        getDashboardLOS(selectedDate, timeRange, focusTime, zoomLevel, selectedLocationId || undefined, startTime),
        getDashboardOcclusionTrends(selectedDate, timeRange, focusTime, zoomLevel, startTime),
        getDashboardOcclusion(selectedDate, timeRange, startTime),
        getAISynthesis(selectedDate, timeRange, startTime),
      ])

      setSummary(summaryResponse)
      setTraffic(trafficResponse)
      setLosTraffic(losTrafficResponse)
      setOcclusionTrends(occlusionTrendsResponse)
      setOcclusion(occlusionResponse)
      setSynthesis(synthesisResponse)
      setDashboardError(null)
    } catch (error) {
      setSummary(null)
      setTraffic(null)
      setLosTraffic(null)
      setOcclusionTrends(null)
      setOcclusion(null)
      setSynthesis(null)
      setDashboardError(error instanceof Error ? error.message : "Failed to load dashboard data.")
    } finally {
      setDashboardLoading(false)
    }
  }, [focusTime, selectedDate, selectedLocationId, startTime, timeRange, zoomLevel])

  const loadModelInfo = useCallback(async () => {
    setModelLoading(true)
    try {
      const [modelResponse, inferenceResponse] = await Promise.all([getCurrentModel(), getInferenceStatus()])
      setModelInfo(modelResponse)
      setInferenceStatus(inferenceResponse)
      setModelError(null)
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Failed to load model information.")
    } finally {
      setModelLoading(false)
    }
  }, [])

  const loadFootageDates = useCallback(async () => {
    try {
      const response = await getLocations()
      setLocations(response)
      if (!selectedLocationId && response.length > 0) {
        setSelectedLocationId(response[0].id)
      }
      if (selectedLocationId && !response.some((location) => location.id === selectedLocationId)) {
        setSelectedLocationId(response[0]?.id ?? "")
      }
      setFootageDates(Array.from(new Set(response.flatMap((location) => location.videos.map((video) => video.date)))).sort())
    } catch {
      // Leave existing date highlights untouched if this auxiliary request fails.
    }
  }, [selectedLocationId])

  useEffect(() => {
    void loadDashboard()
  }, [loadDashboard])

  useEffect(() => {
    void loadModelInfo()
  }, [loadModelInfo])

  useEffect(() => {
    void loadFootageDates()
  }, [loadFootageDates])

  useEffect(() => {
    if (settledUploadsVersion === 0) {
      return
    }

    void loadDashboard()
    void loadFootageDates()
  }, [loadDashboard, loadFootageDates, settledUploadsVersion])

  useEffect(() => {
    if (hourFilter !== "all" && !occlusion?.availableHours.includes(hourFilter)) {
      setHourFilter("all")
    }
  }, [hourFilter, occlusion])

  const handleModelUpload = async () => {
    if (!modelFile) {
      return
    }

    setModelUploading(true)
    try {
      await uploadModel(modelFile)
      await loadModelInfo()
      setModelError(null)
      setModelDialogOpen(false)
      setModelFile(null)
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Failed to upload model.")
    } finally {
      setModelUploading(false)
    }
  }

  const handleRequirementUpload = async () => {
    if (!requirementFile) {
      return
    }

    setRequirementUploading(true)
    try {
      const response = await uploadInferenceRequirement({
        file: requirementFile,
        requirementType,
      })
      setRequirementUploadMessage(`${response.message} Saved to ${response.savedPath}`)
      setActionError(null)
      setRequirementFile(null)
      await loadModelInfo()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to upload requirement file.")
    } finally {
      setRequirementUploading(false)
    }
  }

  const handleExportReport = async () => {
    setReportExporting(true)
    setActionError(null)

    try {
      const { blob, filename } = await downloadDashboardReport(selectedDate, timeRange, startTime)
      const downloadUrl = window.URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = downloadUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(downloadUrl)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to export report.")
    } finally {
      setReportExporting(false)
    }
  }

  const currentModelLabel = modelInfo?.currentModel ?? "No model uploaded yet"
  const currentModelTimestamp = modelInfo?.uploadedAt ? new Date(modelInfo.uploadedAt).toLocaleString("en-US") : null
  const inferenceReady = Boolean(inferenceStatus?.ready)
  const missingRequiredPath = inferenceStatus?.missingFixedPath
  const bannerError = dashboardError ?? modelError ?? actionError

  const handleDateChange = (value: string) => {
    setSelectedDate(value)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleTimeRangeChange = (value: string) => {
    setTimeRange(value)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleStartTimeChange = (value: string) => {
    setStartTime(value)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleAnalyticsZoom = (time: string) => {
    const canZoomInAnyChart = Boolean(traffic?.canZoomIn) || Boolean(losTraffic?.canZoomIn) || Boolean(occlusionTrends?.canZoomIn)
    if (!canZoomInAnyChart) {
      return
    }

    setFocusTime(time)
    setZoomLevel((current) => current + 1)
  }

  const handleResetZoom = () => {
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-border bg-card/50 px-6 py-4 backdrop-blur-sm">
        <div>
          <h1 className="text-xl font-semibold text-foreground">System Analytics Dashboard</h1>
          <p className="text-sm text-muted-foreground">Real-time pedestrian tracking metrics</p>
        </div>

        <div className="flex items-center gap-3">
          <FootageDatePicker
            value={selectedDate}
            onChange={handleDateChange}
            highlightedDates={footageDates}
            placeholder="Select date"
          />

          <Select value={timeRange} onValueChange={handleTimeRangeChange}>
            <SelectTrigger className="w-44 rounded-2xl border-border bg-secondary text-foreground">
              <Clock className="mr-2 h-4 w-4 text-muted-foreground" />
              <SelectValue placeholder="Select time range" />
            </SelectTrigger>
            <SelectContent className="rounded-xl border-border bg-popover">
              {TIME_RANGE_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value} className="rounded-lg text-foreground">
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={startTime} onValueChange={handleStartTimeChange}>
            <SelectTrigger className="w-36 rounded-2xl border-border bg-secondary text-foreground">
              <SelectValue placeholder="Start time" />
            </SelectTrigger>
            <SelectContent className="max-h-80 rounded-xl border-border bg-popover">
              {START_TIME_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value} className="rounded-lg text-foreground">
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={selectedLocationId} onValueChange={setSelectedLocationId}>
            <SelectTrigger className="w-44 rounded-2xl border-border bg-secondary text-foreground">
              <SelectValue placeholder="Select location" />
            </SelectTrigger>
            <SelectContent className="rounded-xl border-border bg-popover">
              {locations.map((location) => (
                <SelectItem key={location.id} value={location.id} className="rounded-lg text-foreground">
                  {location.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Dialog open={modelDialogOpen} onOpenChange={setModelDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" className="rounded-2xl border-border px-4 text-foreground hover:bg-secondary">
                <Settings2 className="mr-2 h-4 w-4" />
                Edit Model
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md rounded-3xl border-border bg-card">
              <DialogHeader>
                <DialogTitle className="text-foreground">Detection Model Settings</DialogTitle>
                <DialogDescription className="text-muted-foreground">
                  Upload a PyTorch model file (.pt or .pth) for pedestrian detection
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4 pt-4">
                <div className="rounded-2xl border border-border bg-secondary p-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/20">
                      <FileCode className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-foreground">Current Model</p>
                      <p className="text-xs text-muted-foreground">{modelLoading ? "Loading model..." : currentModelLabel}</p>
                      {currentModelTimestamp && (
                        <p className="mt-1 text-[11px] text-muted-foreground">Uploaded {currentModelTimestamp}</p>
                      )}
                      {!modelLoading && (
                        <p className={`mt-2 text-[11px] font-medium ${inferenceReady ? "text-emerald-600" : "text-amber-600"}`}>
                          {inferenceReady ? "Inference ready for video processing" : "Inference not ready yet"}
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-border bg-secondary p-4">
                  <p className="text-sm font-medium text-foreground">Inference Requirements</p>
                  <p className="mt-1 text-xs text-muted-foreground">The backend must satisfy all checks below before video upload processing can start.</p>

                  <ul className="mt-3 space-y-1 text-xs text-foreground">
                    <li>Model uploaded (.pt/.pth): {inferenceStatus?.modelExists ? "OK" : "Missing"}</li>
                    <li>RT-DETR pipeline files found: {inferenceStatus?.installed ? "OK" : "Missing"}</li>
                    <li>Ready to process videos: {inferenceReady ? "Yes" : "No"}</li>
                  </ul>

                  <div className="mt-3 rounded-xl border border-border/70 bg-background/60 p-3 text-[11px] text-muted-foreground">
                    <p className="font-medium text-foreground">Required files checklist</p>
                    <p className="mt-1">1. tools/infer.py in the RT-DETR repo</p>
                    <p>2. Config in backend/storage/inference_requirements/configs/rtdetr/</p>
                    <p>3. Counting config in backend/storage/inference_requirements/counting/</p>
                    <p>4. Active model weights (.pt or .pth)</p>
                  </div>

                  {missingRequiredPath && (
                    <p className="mt-3 text-[11px] text-amber-700">
                      Missing required path: {missingRequiredPath}
                    </p>
                  )}

                  <div className="mt-4 space-y-3 rounded-xl border border-border/70 bg-background/60 p-3">
                    <p className="text-xs font-medium text-foreground">Upload Missing Requirement File</p>

                    <Select value={requirementType} onValueChange={(value) => setRequirementType(value as InferenceRequirementType)}>
                      <SelectTrigger className="h-9 rounded-xl border-border bg-background text-xs text-foreground">
                        <SelectValue placeholder="Select requirement type" />
                      </SelectTrigger>
                      <SelectContent className="rounded-xl border-border bg-popover">
                        <SelectItem value="infer-config" className="rounded-lg text-foreground">Infer Config (.yml/.yaml)</SelectItem>
                        <SelectItem value="annotations" className="rounded-lg text-foreground">Annotations (.json)</SelectItem>
                        <SelectItem value="counting-config" className="rounded-lg text-foreground">Counting Config (.json)</SelectItem>
                      </SelectContent>
                    </Select>

                    <input
                      type="file"
                      accept={requirementType === "infer-config" ? ".yml,.yaml" : ".json"}
                      onChange={(e) => setRequirementFile(e.target.files?.[0] || null)}
                      className="block w-full text-xs text-muted-foreground file:mr-3 file:rounded-lg file:border-0 file:bg-secondary file:px-3 file:py-2 file:text-xs file:font-medium file:text-foreground hover:file:bg-secondary/80"
                    />

                    <Button
                      onClick={handleRequirementUpload}
                      disabled={!requirementFile || requirementUploading}
                      variant="outline"
                      className="h-9 w-full rounded-xl border-border text-xs text-foreground hover:bg-secondary"
                    >
                      {requirementUploading ? (
                        <>
                          <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                          Uploading Requirement...
                        </>
                      ) : (
                        "Upload Requirement File"
                      )}
                    </Button>

                    {requirementUploadMessage && <p className="text-[11px] text-emerald-700">{requirementUploadMessage}</p>}
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Upload New Model</label>
                  <div
                    className={`rounded-2xl border-2 border-dashed p-6 text-center transition-colors ${
                      modelFile ? "border-accent bg-accent/10" : "border-border hover:border-muted-foreground"
                    }`}
                  >
                    <input
                      type="file"
                      accept=".pt,.pth"
                      onChange={(e) => setModelFile(e.target.files?.[0] || null)}
                      className="hidden"
                      id="model-upload"
                    />
                    <label htmlFor="model-upload" className="cursor-pointer">
                      <Upload className={`mx-auto mb-2 h-8 w-8 ${modelFile ? "text-accent" : "text-muted-foreground"}`} />
                      {modelFile ? (
                        <p className="text-sm font-medium text-accent">{modelFile.name}</p>
                      ) : (
                        <>
                          <p className="text-sm text-foreground">Click to upload .pt or .pth file</p>
                          <p className="mt-1 text-xs text-muted-foreground">PyTorch model weights</p>
                        </>
                      )}
                    </label>
                  </div>
                </div>

                <Button
                  onClick={handleModelUpload}
                  disabled={!modelFile || modelUploading}
                  className="w-full rounded-2xl bg-primary text-primary-foreground hover:bg-primary/90"
                >
                  {modelUploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading Model...
                    </>
                  ) : (
                    "Update Model"
                  )}
                </Button>
              </div>
            </DialogContent>
          </Dialog>

          <Button
            variant="outline"
            className="rounded-2xl border-border px-4 text-foreground hover:bg-secondary"
            onClick={() => {
              void loadDashboard()
              void loadModelInfo()
              void loadFootageDates()
            }}
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${dashboardLoading || modelLoading ? "animate-spin" : ""}`} />
            Refresh
          </Button>

          <Button
            className="rounded-2xl bg-accent px-4 text-accent-foreground shadow-elevated-sm hover:bg-accent/90"
            onClick={() => {
              void handleExportReport()
            }}
            disabled={reportExporting}
          >
            {reportExporting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Download className="mr-2 h-4 w-4" />}
            {reportExporting ? "Exporting..." : "Export Report"}
          </Button>
        </div>
      </header>

      <div className="flex-1 space-y-6 overflow-auto p-6">
        {bannerError && (
          <div className="flex items-start gap-3 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{bannerError}</span>
          </div>
        )}

        <KPICards summary={summary} loading={dashboardLoading} />

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[65%_35%]">
          <div className="space-y-6">
            <PedestrianChart
              title="Vehicle Count"
              description="Estimated cumulative vehicle count for each location over the selected timeline."
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={traffic?.series ?? []}
              metricKey="cumulativeUniquePedestrians"
              metricLabel="Vehicle Count"
              seriesColor="#22C55E"
              locationTotals={traffic?.locationTotals ?? []}
              bucketMinutes={traffic?.bucketMinutes ?? 60}
              zoomLevel={traffic?.zoomLevel ?? 0}
              canZoomIn={traffic?.canZoomIn ?? false}
              focusTime={traffic?.focusTime}
              windowStart={traffic?.windowStart}
              windowEnd={traffic?.windowEnd}
              loading={dashboardLoading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
              chartType={vehicleChartType}
              onChartTypeChange={setVehicleChartType}
            />
            <PedestrianChart
              title="LOS"
              description="Level of Service trend for the selected location across the chosen time window."
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={losTraffic?.series ?? []}
              metricKey="los"
              metricLabel="LOS"
              seriesColor="#06B6D4"
              bucketMinutes={losTraffic?.bucketMinutes ?? 60}
              zoomLevel={losTraffic?.zoomLevel ?? 0}
              canZoomIn={losTraffic?.canZoomIn ?? false}
              focusTime={losTraffic?.focusTime}
              windowStart={losTraffic?.windowStart}
              windowEnd={losTraffic?.windowEnd}
              loading={dashboardLoading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
            />
            <OcclusionTrendsChart
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={occlusionTrends?.series ?? []}
              bucketMinutes={occlusionTrends?.bucketMinutes ?? 60}
              zoomLevel={occlusionTrends?.zoomLevel ?? 0}
              canZoomIn={occlusionTrends?.canZoomIn ?? false}
              focusTime={occlusionTrends?.focusTime}
              windowStart={occlusionTrends?.windowStart}
              windowEnd={occlusionTrends?.windowEnd}
              loading={dashboardLoading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
              chartType={inOutChartType}
              onChartTypeChange={setInOutChartType}
            />
          </div>
          <OcclusionMap hourFilter={hourFilter} onHourFilterChange={setHourFilter} data={occlusion} loading={dashboardLoading} />
        </div>

        <AISynthesis selectedDate={selectedDate} timeRange={timeRange} data={synthesis} loading={dashboardLoading} />
      </div>
    </div>
  )
}
