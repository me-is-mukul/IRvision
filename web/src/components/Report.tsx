"use client";

import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { fetchReport, type Report as ReportData } from "@/lib/api";
import { useMode } from "./ModeProvider";
import { Section } from "./ui";

/** Horizontal bar with a label and value; the highlighted bar glows neon. */
function Bar({ label, value, max, display, highlight }: { label: string; value: number; max: number; display: string; highlight?: boolean }) {
  return (
    <div>
      <div className="mb-1.5 flex justify-between text-[14px]">
        <span className={highlight ? "font-medium text-text" : "text-subtle"}>{label}</span>
        <span className={`font-mono ${highlight ? "text-neon" : "text-muted"}`}>{display}</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-surface-2">
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${Math.max(2, (value / max) * 100)}%` }}
          viewport={{ once: true }}
          transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
          className={`h-full rounded-full ${highlight ? "bg-neon shadow-[0_0_16px_rgb(34_211_238/0.6)]" : "bg-white/25"}`}
        />
      </div>
    </div>
  );
}

export function Report() {
  const { mode } = useMode();
  const [data, setData] = useState<ReportData | null>(null);

  useEffect(() => {
    fetchReport(mode).then(setData).catch(() => setData({}));
  }, [mode]);

  const col = data?.colorization;
  const sem = data?.semantic;
  const adv = data?.advanced;
  const isOurs = (m: string) => m.startsWith("IRVision");

  return (
    <Section
      id="results"
      eyebrow="Results"
      title="Measured, not claimed."
      lead={
        <>
          Tested on {col?.patches ?? 355} image patches, including a whole city, Hyderabad, that was never used for
          training. Every number below comes from an evaluation script.
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card p-5">
          <h3 className="text-xl font-semibold tracking-tight">Structure preserved (SSIM)</h3>
          <p className="mt-1 text-[14px] text-muted">Unseen city · higher is better</p>
          <div className="mt-4 space-y-4">
            {col?.rows.map((r) => (
              <Bar key={r.method} label={r.method} value={r.holdout?.[1] ?? r.all[1]} max={0.6}
                   display={(r.holdout?.[1] ?? r.all[1]).toFixed(3)} highlight={isOurs(r.method)} />
            ))}
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-xl font-semibold tracking-tight">Meaning preserved (land-cover mIoU)</h3>
          <p className="mt-1 text-[14px] text-muted">
            Colorized vs. real image · all test patches · model ceiling {sem?.ceiling.toFixed(3) ?? "—"}
          </p>
          <div className="mt-4 space-y-4">
            {sem?.rows.map((r) => (
              <Bar key={r.method} label={r.method} value={r.miou} max={sem.ceiling}
                   display={r.miou.toFixed(3)} highlight={isOurs(r.method)} />
            ))}
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-xl font-semibold tracking-tight">Colour accuracy (PSNR)</h3>
          <p className="mt-1 text-[14px] text-muted">All test patches · higher is better</p>
          <div className="mt-4 space-y-4">
            {col?.rows.map((r) => (
              <Bar key={r.method} label={r.method} value={r.all[0] - 15} max={9}
                   display={`${r.all[0].toFixed(2)} dB`} highlight={isOurs(r.method)} />
            ))}
          </div>
          <p className="mt-3 text-[13px] leading-relaxed text-muted">
            On the unseen city, colour accuracy matches the lookup table
            ({col?.rows.find((r) => r.method === "Lookup table")?.holdout?.[0].toFixed(2)} vs.{" "}
            {col?.rows.find((r) => isOurs(r.method))?.holdout?.[0].toFixed(2)} dB); the model’s advantage there is structure and meaning.
          </p>
        </div>

        <div className="card p-5">
          <h3 className="text-xl font-semibold tracking-tight">Land cover, class by class</h3>
          <p className="mt-1 text-[14px] text-muted">IoU of IRVision colour vs. real image</p>
          <div className="mt-4 space-y-4">
            {sem &&
              Object.entries(sem.per_class).map(([name, v]) => (
                <Bar key={name} label={name[0].toUpperCase() + name.slice(1)} value={v ?? 0} max={1}
                     display={(v ?? 0).toFixed(2)} highlight={(v ?? 0) > 0.2} />
              ))}
          </div>
          <p className="mt-3 text-[13px] leading-relaxed text-muted">
            Water and vegetation survive colorization; dense city does not yet. Colour metrics alone would not show this.
          </p>
        </div>
      </div>

      {adv && (
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          {[
            {
              title: "Super-resolution",
              value: `${adv.super_resolution[1].psnr.toFixed(2)} dB`,
              text: `EDSR ×2 before colorization lowered PSNR from ${adv.super_resolution[0].psnr.toFixed(2)} dB. Optional, off by default.`,
            },
            {
              title: "Object detection",
              value: `${adv.detection.matched} matched`,
              text: `YOLOv8 found ${Object.values(adv.detection.true_rgb).reduce((a, b) => a + b, 0)} objects in real images; none matched on colorized images at 30 m.`,
            },
            {
              title: "Perceptual loss",
              value: "no gain",
              text: "Adding a VGG19 perceptual loss did not improve validation scores (24.11 vs. 24.12 dB), so the simpler model stays.",
            },
          ].map((c) => (
            <div key={c.title} className="card p-5">
              <p className="text-[13px] font-medium uppercase tracking-wider text-muted">{c.title}</p>
              <p className="mt-2 text-2xl font-semibold tracking-tight text-amber">{c.value}</p>
              <p className="mt-2 text-[14px] leading-relaxed text-muted">{c.text}</p>
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}
