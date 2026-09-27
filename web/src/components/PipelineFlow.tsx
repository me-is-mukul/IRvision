"use client";

import { motion } from "motion/react";
import { Check } from "lucide-react";
import { STAGE_DEFS, type NodeState } from "@/lib/stages";

interface Props {
  states: Record<string, NodeState>;
  times?: Record<string, number>;
  compact?: boolean;
}

const fmt = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);

/** The ten pipeline stages as a connected, animated flow. */
export function PipelineFlow({ states, times = {}, compact = false }: Props) {
  return (
    <div className="-mx-4 overflow-x-auto px-4 pb-2 [scrollbar-width:none]">
      <ol className="flex min-w-max items-start">
        {STAGE_DEFS.map((stage, i) => {
          const state = states[stage.key] ?? "idle";
          const next = STAGE_DEFS[i + 1];
          const nextState = next ? states[next.key] : undefined;
          const Icon = stage.icon;
          return (
            <li key={stage.key} className="flex items-start">
              <div className={`flex flex-col items-center text-center ${compact ? "w-[86px]" : "w-[100px]"}`}>
                <motion.div
                  layout
                  animate={{ scale: state === "active" ? 1.08 : 1, opacity: state === "off" ? 0.35 : 1 }}
                  transition={{ type: "spring", stiffness: 300, damping: 22 }}
                  className={[
                    "relative grid place-items-center rounded-2xl border transition-colors duration-300",
                    compact ? "size-12" : "size-14",
                    state === "active" && "border-neon/70 bg-neon-soft text-neon animate-pulse-neon",
                    state === "done" && "border-neon/40 bg-neon/10 text-neon",
                    state === "idle" && "border-line-strong bg-surface-2 text-subtle",
                    state === "off" && "border-dashed border-line-strong bg-transparent text-muted",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  <Icon className={compact ? "size-5" : "size-6"} strokeWidth={1.75} />
                  {state === "done" && (
                    <motion.span
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="absolute -right-1.5 -top-1.5 grid size-5 place-items-center rounded-full bg-neon text-ink"
                    >
                      <Check className="size-3" strokeWidth={3} />
                    </motion.span>
                  )}
                </motion.div>
                <span
                  className={`mt-2 text-[13px] font-medium leading-tight ${
                    state === "active" ? "text-neon" : state === "off" ? "text-muted" : "text-text"
                  }`}
                >
                  {compact ? stage.short : stage.label}
                </span>
                <span className="mt-1 h-4 font-mono text-[11px] text-muted">
                  {state === "done" && times[stage.key] !== undefined
                    ? fmt(times[stage.key])
                    : state === "off"
                      ? "off"
                      : state === "active"
                        ? "running"
                        : ""}
                </span>
              </div>
              {next && (
                <div className={`relative mx-1 h-px ${compact ? "mt-4 w-6" : "mt-7 w-8"} overflow-hidden bg-line-strong`}>
                  <motion.div
                    className="absolute inset-y-0 left-0 bg-neon"
                    initial={false}
                    animate={{ width: state === "done" && nextState !== "idle" ? "100%" : "0%" }}
                    transition={{ duration: 0.35, ease: "easeOut" }}
                  />
                  {state === "done" && nextState === "active" && (
                    <motion.div
                      className="absolute top-1/2 size-2 -translate-y-1/2 rounded-full bg-neon shadow-[0_0_12px_2px_rgb(34_211_238/0.8)]"
                      initial={{ left: "-10%" }}
                      animate={{ left: "110%" }}
                      transition={{ duration: 0.7, repeat: Infinity, ease: "easeInOut" }}
                    />
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
