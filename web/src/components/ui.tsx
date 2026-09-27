"use client";

import { animate, useInView } from "motion/react";
import { useEffect, useRef, useState, type ReactNode } from "react";

/** Section wrapper with an Apple-style eyebrow, title and lead text. */
export function Section({
  id,
  eyebrow,
  title,
  lead,
  children,
  className = "",
}: {
  id?: string;
  eyebrow?: string;
  title: ReactNode;
  lead?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={`mx-auto w-full max-w-6xl px-5 py-10 sm:py-12 ${className}`}>
      <div className="mx-auto mb-8 max-w-3xl text-center">
        {eyebrow && <p className="mb-2 text-[13px] font-semibold uppercase tracking-[0.18em] text-neon">{eyebrow}</p>}
        <h2 className="text-gradient text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h2>
        {lead && <p className="mt-3 text-base leading-relaxed text-subtle sm:text-lg">{lead}</p>}
      </div>
      {children}
    </section>
  );
}

/** Number that counts up when it scrolls into view (or when its value changes). */
export function CountUp({ value, decimals = 0, suffix = "" }: { value: number; decimals?: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (!inView) return;
    const controls = animate(0, value, { duration: 1.1, ease: [0.16, 1, 0.3, 1], onUpdate: setShown });
    return () => controls.stop();
  }, [inView, value]);
  return (
    <span ref={ref} className="tabular-nums">
      {shown.toFixed(decimals)}
      {suffix}
    </span>
  );
}

/** Apple-style switch. */
export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled = false,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
  disabled?: boolean;
}) {
  return (
    <label className={`flex items-start justify-between gap-4 ${disabled ? "opacity-50" : "cursor-pointer"}`}>
      <span>
        <span className="block text-[15px] font-medium">{label}</span>
        {hint && <span className="mt-0.5 block text-[13px] leading-snug text-muted">{hint}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-[30px] w-[51px] shrink-0 rounded-full transition-colors duration-200 ${
          checked ? "bg-neon" : "bg-surface-2 ring-1 ring-line-strong"
        }`}
      >
        <span
          className={`absolute top-[2px] size-[26px] rounded-full bg-white shadow-md transition-transform duration-200 ${
            checked ? "translate-x-[23px]" : "translate-x-[2px]"
          }`}
        />
      </button>
    </label>
  );
}

/** Small rounded status pill. */
export function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "neon" | "amber" }) {
  const tones = {
    neutral: "border-line-strong text-subtle",
    neon: "border-neon/40 bg-neon-soft text-neon",
    amber: "border-amber/40 bg-amber/10 text-amber",
  };
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}
