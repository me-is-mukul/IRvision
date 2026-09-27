"use client";

import { motion } from "motion/react";
import { ArrowRight, Play } from "lucide-react";
import { CompareSlider } from "./CompareSlider";

const HERO = "/demo/hyderabad_farmland/sr0_det0";

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden pt-20 sm:pt-24">
      <div className="grid-bg pointer-events-none absolute inset-0" />
      <div className="pointer-events-none absolute left-1/2 top-24 h-[480px] w-[820px] -translate-x-1/2 rounded-full bg-neon/10 blur-[140px]" />

      <div className="relative mx-auto grid max-w-6xl items-center gap-10 px-5 pb-12 lg:grid-cols-[1.1fr_1fr]">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        >
          <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-neon/30 bg-neon-soft px-3 py-1 text-xs font-medium text-neon">
            Landsat 8/9 · Thermal band 10 · Deep learning
          </p>
          <h1 className="text-4xl font-semibold leading-[1.02] tracking-[-0.035em] sm:text-6xl">
            <span className="text-gradient">See heat.</span>
            <br />
            <span className="text-neon-gradient">In colour.</span>
          </h1>
          <p className="mt-4 max-w-xl text-base leading-relaxed text-subtle sm:text-lg">
            IRVision turns single-band thermal satellite images into readable colour images, then checks
            that the result still shows the same water, fields and cities as reality.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <a
              href="#playground"
              className="group inline-flex items-center gap-2 rounded-full bg-neon px-6 py-3 text-[15px] font-semibold text-ink transition hover:bg-neon-strong hover:shadow-[0_0_32px_rgb(34_211_238/0.5)]"
            >
              <Play className="size-4 fill-ink" />
              Run the pipeline
            </a>
            <a
              href="#flow"
              className="group inline-flex items-center gap-1.5 rounded-full px-5 py-3 text-[15px] font-medium text-neon transition hover:bg-neon-soft"
            >
              How it works
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
            </a>
          </div>
          <dl className="mt-8 grid max-w-lg grid-cols-3 gap-4 border-t border-line pt-4">
            {[
              ["10", "Landsat scenes"],
              ["1.6 s", "per 60 km scene"],
              ["8×", "more land cover kept"],
            ].map(([v, l]) => (
              <div key={l}>
                <dt className="text-2xl font-semibold tracking-tight">{v}</dt>
                <dd className="mt-1 text-[13px] leading-snug text-muted">{l}</dd>
              </div>
            ))}
          </dl>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="relative"
        >
          <div className="absolute -inset-4 rounded-[28px] bg-gradient-to-b from-neon/20 to-transparent blur-2xl" />
          <div className="glass relative rounded-[22px] p-3">
            <CompareSlider
              before={`${HERO}/input.webp`}
              after={`${HERO}/colorized.webp`}
              beforeLabel="Thermal input"
              afterLabel="IRVision colour"
              autoplay
            />
            <p className="px-2 pb-1 pt-3 text-center text-[13px] text-muted">
              Hyderabad farmland: a city the model never saw during training. Drag to compare.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
