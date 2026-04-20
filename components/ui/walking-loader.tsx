"use client"

import { X } from "lucide-react"
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react"

interface WalkingLoaderProps {
  isVisible: boolean
  label: string
  progress?: number | null
  onClose?: () => void
}

export function WalkingLoader({ isVisible, label, progress = null, onClose }: WalkingLoaderProps) {
  const [frame, setFrame] = useState(0)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  
  useEffect(() => {
    if (!isVisible) return
    
    const interval = setInterval(() => {
      setFrame((prev) => (prev + 1) % 4)
    }, 200)
    
    return () => clearInterval(interval)
  }, [isVisible])

  useEffect(() => {
    if (!isVisible) {
      setElapsedSeconds(0)
      return
    }

    const startedAt = Date.now()
    setElapsedSeconds(0)

    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000))
    }, 1000)

    return () => clearInterval(interval)
  }, [isVisible])

  if (!isVisible) return null

  const showElapsedTime = elapsedSeconds > 5

  const vehicleBounceY = frame % 2 === 0 ? 0 : -0.7
  const wheelSpokeRotation = frame * 45

  return (
    <div className="fixed inset-0 z-[80] flex items-start justify-center pt-32 pointer-events-none">
      <div className="pointer-events-auto relative flex w-[min(92vw,32rem)] flex-col items-center gap-4 rounded-3xl border border-border bg-card p-6 shadow-elevated">
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            className="absolute right-4 top-4 inline-flex h-8 w-8 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            aria-label="Hide loader"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null}
        {/* Vehicle Animation Container */}
        <div className="relative w-20 h-24 flex items-center justify-center">
          {/* Road line with moving lane marks */}
          <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-border overflow-hidden">
            <div 
              className="absolute top-0 left-0 w-full h-full"
              style={{
                background: 'repeating-linear-gradient(90deg, transparent, transparent 8px, var(--primary) 8px, var(--primary) 12px)',
                animation: 'slideRight 0.4s linear infinite',
              }}
            />
          </div>
          
          {/* Moving Vehicle */}
          <svg 
            viewBox="0 0 60 72" 
            className="w-16 h-20"
            style={{
              filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))',
              transform: `translateY(${vehicleBounceY}px)`,
              transition: 'transform 120ms linear',
            }}
          >
            {/* Vehicle Body */}
            <path
              d="M10 42 L14 34 Q18 26 28 26 L39 26 Q45 26 48 31 L52 36 L52 42 Z"
              fill="url(#gradient)"
              stroke="#0F172A"
              strokeWidth="1.2"
            />

            {/* Cabin */}
            <path
              d="M22 27 L29 27 L35 33 L19 33 Z"
              fill="#E2E8F0"
              stroke="#0F172A"
              strokeWidth="0.9"
            />

            {/* Headlights */}
            <circle cx="51" cy="38" r="1.3" fill="#FDE047" />
            <rect x="10" y="37" width="1.6" height="3.2" rx="0.8" fill="#FCA5A5" />

            {/* Wheels */}
            <g>
              <circle cx="20" cy="42" r="4.7" fill="#111827" />
              <circle cx="20" cy="42" r="2.2" fill="#94A3B8" />
              <path
                d="M20 39.8 L20 44.2 M17.8 42 L22.2 42"
                stroke="#0F172A"
                strokeWidth="0.8"
                strokeLinecap="round"
                transform={`rotate(${wheelSpokeRotation} 20 42)`}
              />
            </g>
            <g>
              <circle cx="41" cy="42" r="4.7" fill="#111827" />
              <circle cx="41" cy="42" r="2.2" fill="#94A3B8" />
              <path
                d="M41 39.8 L41 44.2 M38.8 42 L43.2 42"
                stroke="#0F172A"
                strokeWidth="0.8"
                strokeLinecap="round"
                transform={`rotate(${wheelSpokeRotation} 41 42)`}
              />
            </g>

            {/* Motion streak */}
            <path
              d="M6 30 H11 M4 34 H10 M5 38 H9"
              stroke="#64748B"
              strokeWidth="1.2"
              strokeLinecap="round"
              opacity="0.85"
            />
            
            {/* Gradient Definition */}
            <defs>
              <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#A855F7" />
                <stop offset="100%" stopColor="#22D3EE" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        
        {/* Label */}
        <div className="w-full max-w-[26rem] text-center">
          <p className="break-words text-sm font-medium leading-snug text-foreground">{label}</p>
          {typeof progress === "number" ? (
            <p className="mt-2 text-xs font-medium uppercase tracking-[0.18em] text-primary">{progress}% complete</p>
          ) : null}
          <div className="mt-1 flex items-center justify-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          {showElapsedTime ? (
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
              Elapsed time: <span className="font-medium text-foreground">{elapsedSeconds} {elapsedSeconds === 1 ? "second" : "seconds"}</span>
            </p>
          ) : null}
        </div>
      </div>
      
      {/* CSS Animation */}
      <style jsx>{`
        @keyframes slideRight {
          from {
            transform: translateX(0);
          }
          to {
            transform: translateX(20px);
          }
        }
      `}</style>
    </div>
  )
}

interface LoadingContextType {
  showLoader: (label: string, progress?: number | null) => void
  updateLoader: (payload: { label?: string; progress?: number | null }) => void
  hideLoader: () => void
  isLoading: boolean
  loadingLabel: string
  loadingProgress: number | null
}

const LoadingContext = createContext<LoadingContextType | null>(null)

export function LoadingProvider({ children }: { children: ReactNode }) {
  const [isLoading, setIsLoading] = useState(false)
  const [loadingLabel, setLoadingLabel] = useState("")
  const [loadingProgress, setLoadingProgress] = useState<number | null>(null)

  const showLoader = useCallback((label: string, progress: number | null = null) => {
    setLoadingLabel(label)
    setLoadingProgress(progress)
    setIsLoading(true)
  }, [])

  const updateLoader = useCallback(({ label, progress }: { label?: string; progress?: number | null }) => {
    if (typeof label === "string") {
      setLoadingLabel(label)
    }
    if (progress !== undefined) {
      setLoadingProgress(progress)
    }
  }, [])

  const hideLoader = useCallback(() => {
    setIsLoading(false)
    setLoadingLabel("")
    setLoadingProgress(null)
  }, [])

  const value = useMemo(
    () => ({ showLoader, updateLoader, hideLoader, isLoading, loadingLabel, loadingProgress }),
    [hideLoader, isLoading, loadingLabel, loadingProgress, showLoader, updateLoader],
  )

  return (
    <LoadingContext.Provider value={value}>
      {children}
      <WalkingLoader isVisible={isLoading} label={loadingLabel} progress={loadingProgress} />
    </LoadingContext.Provider>
  )
}

export function useLoading() {
  const context = useContext(LoadingContext)
  if (!context) {
    throw new Error("useLoading must be used within a LoadingProvider")
  }
  return context
}
