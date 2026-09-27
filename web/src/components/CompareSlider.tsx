"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronsLeftRight } from "lucide-react";

interface Props {
  before: string;
  after: string;
  beforeLabel: string;
  afterLabel: string;
  /** gently sweep the divider back and forth until the user interacts */
  autoplay?: boolean;
  className?: string;
}

/** Drag-to-compare view of two aligned images. */
export function CompareSlider({ before, after, beforeLabel, afterLabel, autoplay = false, className = "" }: Props) {
  const [pos, setPos] = useState(50);
  const [touched, setTouched] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!autoplay || touched) return;
    let frame = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      setPos(50 + 32 * Math.sin((t - t0) / 1400));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [autoplay, touched]);

  const move = useCallback((clientX: number) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    setPos(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  }, []);

  return (
    <div
      ref={ref}
      className={`relative select-none overflow-hidden rounded-[16px] border border-line bg-surface-2 ${className}`}
      onPointerDown={(e) => {
        setTouched(true);
        (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
        move(e.clientX);
      }}
      onPointerMove={(e) => e.buttons === 1 && move(e.clientX)}
      onKeyDown={(e) => {
        const step = e.key === "ArrowLeft" ? -5 : e.key === "ArrowRight" ? 5 : 0;
        if (step) {
          setTouched(true);
          setPos((p) => Math.min(100, Math.max(0, p + step)));
        }
      }}
      role="slider"
      aria-label={`Compare ${beforeLabel} and ${afterLabel}`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pos)}
      tabIndex={0}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={after} alt={afterLabel} className="block aspect-square w-full object-cover" draggable={false} />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={before} alt={beforeLabel} className="block aspect-square w-full object-cover" draggable={false} />
      </div>
      <div className="pointer-events-none absolute inset-y-0" style={{ left: `${pos}%` }}>
        <div className="absolute inset-y-0 -left-px w-0.5 bg-neon shadow-[0_0_16px_2px_rgb(34_211_238/0.7)]" />
        <div className="absolute top-1/2 grid size-10 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-neon/60 bg-ink/70 text-neon backdrop-blur">
          <ChevronsLeftRight className="size-5" />
        </div>
      </div>
      <span className="glass pointer-events-none absolute left-3 top-3 rounded-full px-3 py-1 text-xs font-medium">
        {beforeLabel}
      </span>
      <span className="glass pointer-events-none absolute right-3 top-3 rounded-full px-3 py-1 text-xs font-medium text-neon">
        {afterLabel}
      </span>
    </div>
  );
}
