"use client";

import { motion } from "motion/react";
import { Info } from "lucide-react";
import { useState } from "react";
import type { Payload } from "@/lib/api";
import { CompareSlider } from "./CompareSlider";
import { CountUp } from "./ui";

type View = "thermal" | "truth";

/** Everything one pipeline run produced: images, scores, land cover, detections. */
export function Results({ payload }: { payload: Payload }) {
  const { images, metrics } = payload;
  const hasTruth = Boolean(images.reference);
  const [view, setView] = useState<View>("thermal");
  const colour = images.detections ?? images.colorized ?? "";

  const cards = [
    metrics.psnr !== undefined && { label: "PSNR", value: metrics.psnr, decimals: 2, suffix: " dB", hint: "Colour accuracy vs. true colour. Higher is better." },
    metrics.ssim !== undefined && { label: "SSIM", value: metrics.ssim, decimals: 3, suffix: "", hint: "Structural similarity, 0 to 1. Higher is better." },
    metrics.semantic && { label: "Land-cover agreement", value: metrics.semantic.agreement * 100, decimals: 1, suffix: " %", hint: "Pixels classified the same as in the real image." },
    { label: "Processing time", value: metrics.total_ms, decimals: 0, suffix: " ms", hint: "Whole pipeline, this image." },
  ].filter(Boolean) as { label: string; value: number; decimals: number; suffix: string; hint: string }[];

  return (
    <div className="mt-4 space-y-4">
      {/* scores */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className={`card grid ${cards.length === 1 ? "grid-cols-1" : "grid-cols-2"} divide-line overflow-hidden ${cards.length > 3 ? "lg:grid-cols-4" : "lg:grid-cols-3"} lg:divide-x`}
      >
        {cards.map((c) => (
          <div key={c.label} className="border-line p-5 [&:nth-child(-n+2)]:border-b lg:[&:nth-child(-n+2)]:border-b-0">
            <p className="text-[12px] font-medium uppercase tracking-wider text-muted">{c.label}</p>
            <p className="mt-1 text-[28px] font-semibold leading-tight tracking-tight tabular-nums">
              <CountUp value={c.value} decimals={c.decimals} suffix={c.suffix} />
            </p>
            <p className="mt-1 text-[12px] leading-snug text-muted">{c.hint}</p>
          </div>
        ))}
      </motion.div>

      {/* compare + gallery */}
      <div className="grid gap-4 lg:grid-cols-[1.15fr_1fr]">
        <div className="card p-3 sm:p-4">
          <div className="mb-4 flex items-center justify-between gap-3 px-1">
            <p className="text-[15px] font-semibold tracking-tight">Compare</p>
            {hasTruth && (
              <div className="flex rounded-full border border-line bg-surface-2 p-1 text-[13px]">
                {(["thermal", "truth"] as View[]).map((v) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setView(v)}
                    className={`rounded-full px-3.5 py-1 transition ${view === v ? "bg-text text-ink" : "text-subtle hover:text-text"}`}
                  >
                    {v === "thermal" ? "Thermal ↔ Colour" : "Colour ↔ True"}
                  </button>
                ))}
              </div>
            )}
          </div>
          {view === "thermal" || !hasTruth ? (
            <CompareSlider before={images.enhanced ?? ""} after={colour} beforeLabel="Enhanced thermal" afterLabel="IRVision colour" />
          ) : (
            <CompareSlider before={colour} after={images.reference ?? ""} beforeLabel="IRVision colour" afterLabel="True colour" />
          )}
          <p className="mt-3 px-1 text-[12px] leading-snug text-muted">
            All colour images use the same fixed natural-colour display rendering; scores are computed on the unmodified data.
          </p>
        </div>

        <div className="card p-3 sm:p-4">
          <p className="mb-4 px-1 text-[15px] font-semibold tracking-tight">Outputs</p>
          <div className="grid grid-cols-2 gap-3">
            {[
              ["Thermal input", images.input],
              ["Enhanced (CLAHE)", images.enhanced],
              [images.detections ? "Colour + detections" : "IRVision colour", colour],
              ["True colour", images.reference],
            ]
              .filter(([, src]) => src)
              .map(([label, src]) => (
                <figure key={label} className="min-w-0">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={src} alt={label} className="aspect-square w-full rounded-xl border border-line object-cover" />
                  <figcaption className="truncate pt-2 text-center text-[12.5px] text-subtle">{label}</figcaption>
                </figure>
              ))}
          </div>
        </div>
      </div>

      {/* land cover */}
      {images.semantic && (
        <div className="card p-5 sm:p-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="text-[13px] font-medium uppercase tracking-wider text-muted">Semantic validation</p>
              <h3 className="mt-1 text-2xl font-semibold tracking-tight">Does the colour image mean the same thing?</h3>
            </div>
            {metrics.semantic && (
              <p className="text-[15px] text-subtle">
                mIoU <span className="font-semibold text-text">{metrics.semantic.miou.toFixed(3)}</span> · Dice{" "}
                <span className="font-semibold text-text">{metrics.semantic.mean_dice.toFixed(3)}</span>
              </p>
            )}
          </div>
          <div className="mt-5 grid gap-4 sm:grid-cols-2">
            {[
              ["Land cover in IRVision colour", images.semantic],
              ["Land cover in the real image", images.reference_semantic],
            ]
              .filter(([, src]) => src)
              .map(([label, src]) => (
                <figure key={label} className="min-w-0">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={src} alt={label} className="aspect-square w-full rounded-xl border border-line object-cover [image-rendering:pixelated]" />
                  <figcaption className="pt-2 text-center text-[12.5px] text-subtle">{label}</figcaption>
                </figure>
              ))}
          </div>
          <ul className="mt-5 flex flex-wrap gap-2 text-[13px]">
            {payload.legend.map((c) => (
              <li key={c.name} className="flex items-center gap-2 rounded-full border border-line bg-surface-2 py-1 pl-2 pr-3">
                <span className="size-3 rounded-full ring-1 ring-white/15" style={{ background: c.color }} />
                <span className="capitalize text-subtle">{c.name}</span>
                {metrics.semantic?.iou[c.name] != null && (
                  <span className="font-mono text-[12px] text-text">{(metrics.semantic.iou[c.name] as number).toFixed(2)}</span>
                )}
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[12px] text-muted">Number next to each class: IoU between the two maps.</p>
        </div>
      )}

      {/* detections */}
      {payload.detections && (
        <div className="card flex flex-wrap items-center gap-x-8 gap-y-3 p-5">
          <p className="text-[15px] font-semibold tracking-tight">Object detection</p>
          <p className="text-[14px] text-subtle">
            In colour image: <span className="font-semibold text-text">{payload.detections.length}</span>
          </p>
          {payload.reference_detections && (
            <p className="text-[14px] text-subtle">
              In real image: <span className="font-semibold text-text">{payload.reference_detections.length}</span>
            </p>
          )}
          {metrics.detection && (
            <p className="text-[14px] text-subtle">
              Matched: <span className="font-semibold text-text">{metrics.detection.tp}</span>
            </p>
          )}
          <p className="basis-full text-[13px] text-muted">
            At 30 m per pixel, ships, planes and vehicles are only a few pixels wide. Treat detections as unverified.
          </p>
        </div>
      )}

      {payload.warnings.length > 0 && (
        <div className="flex items-start gap-3 rounded-2xl border border-amber/30 bg-amber/5 p-4 text-[14px] text-amber">
          <Info className="mt-0.5 size-4 shrink-0" />
          <ul className="space-y-1">
            {payload.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
