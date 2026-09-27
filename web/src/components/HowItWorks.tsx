"use client";

import { motion, useInView } from "motion/react";
import { useEffect, useRef, useState } from "react";
import type { Payload } from "@/lib/api";
import { STAGE_DEFS, type NodeState } from "@/lib/stages";
import { PipelineFlow } from "./PipelineFlow";
import { Section } from "./ui";

const LOOP_SOURCE = "/demo/hyderabad_farmland/sr1_det1/payload.json";   // every stage enabled
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Explainer: the full pipeline replays in a loop with real measured stage times. */
export function HowItWorks() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { margin: "-20% 0px" });
  const [times, setTimes] = useState<Record<string, number>>({});
  const [states, setStates] = useState<Record<string, NodeState>>(
    Object.fromEntries(STAGE_DEFS.map((s) => [s.key, "idle"])),
  );
  const [active, setActive] = useState<string>(STAGE_DEFS[0].key);

  useEffect(() => {
    fetch(LOOP_SOURCE)
      .then((r) => r.json())
      .then((p: Payload) => setTimes(Object.fromEntries(p.stages.map((s) => [s.key, s.ms]))))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!inView) return;
    let cancelled = false;
    (async () => {
      while (!cancelled) {
        const st: Record<string, NodeState> = Object.fromEntries(STAGE_DEFS.map((s) => [s.key, "idle"]));
        setStates({ ...st });
        await sleep(600);
        for (const s of STAGE_DEFS) {
          if (cancelled) return;
          st[s.key] = "active";
          setActive(s.key);
          setStates({ ...st });
          await sleep(900);
          st[s.key] = "done";
          setStates({ ...st });
        }
        await sleep(2400);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [inView]);

  const current = STAGE_DEFS.find((s) => s.key === active) ?? STAGE_DEFS[0];

  return (
    <Section
      id="flow"
      eyebrow="How it works"
      title="One image. Ten steps. Every one measured."
      lead="From raw satellite temperature to a colour image and its own quality report, in about a second on a laptop GPU."
    >
      <div ref={ref} className="card p-5 sm:p-6">
        <div className="flex justify-center">
          <PipelineFlow states={states} times={times} />
        </div>
        <motion.div
          key={current.key}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-auto mt-5 flex max-w-xl items-center justify-center gap-3 text-center"
        >
          <current.icon className="size-5 shrink-0 text-neon" />
          <p className="text-[15px] text-subtle">
            <span className="font-medium text-text">{current.label}:</span> {current.description}
            {current.optional && <span className="text-muted"> (optional)</span>}
          </p>
        </motion.div>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Enhance", "Clouds and missing data are masked; local contrast is boosted so faint thermal structure becomes visible."],
          ["Colorize", "A U-Net trained on 1,297 Landsat patches from nine cities predicts the true-colour image from heat alone."],
          ["Verify", "A land-cover model compares water, vegetation and built-up areas in the colorized and the real image."],
          ["Report", "PSNR, SSIM, land-cover agreement and per-stage timing, computed on every run."],
        ].map(([t, d], i) => (
          <motion.div
            key={t}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.08, duration: 0.6 }}
            className="card p-5"
          >
            <p className="font-mono text-xs text-neon">0{i + 1}</p>
            <h3 className="mt-2 text-lg font-semibold tracking-tight">{t}</h3>
            <p className="mt-2 text-[14px] leading-relaxed text-muted">{d}</p>
          </motion.div>
        ))}
      </div>
    </Section>
  );
}
