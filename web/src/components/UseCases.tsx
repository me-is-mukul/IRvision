"use client";

import { motion } from "motion/react";
import { Building2, Droplets, GraduationCap, Siren, Sprout, Thermometer } from "lucide-react";
import { Section } from "./ui";

const CASES = [
  {
    icon: Siren,
    title: "Disaster response",
    text: "Thermal sensors keep imaging at night. A colour view makes flood-affected areas readable for responders who are not remote-sensing experts, and water is the class IRVision preserves best.",
  },
  {
    icon: Sprout,
    title: "Agriculture",
    text: "Fields and vegetation are preserved well, and heat already signals irrigation and crop stress. Colorized views are easier to share with farmers and local officials.",
  },
  {
    icon: Droplets,
    title: "Water management",
    text: "Lakes, reservoirs and rivers stand out clearly in the colorized output, with 0.74 land-cover agreement for water.",
  },
  {
    icon: Thermometer,
    title: "Climate research",
    text: "Pair measured surface temperature with a readable visual of the same scene for reports and field teams.",
  },
  {
    icon: GraduationCap,
    title: "Education",
    text: "Make satellite thermal data understandable for students and the public, with an honest quality score on every image.",
  },
];

export function UseCases() {
  return (
    <Section
      id="impact"
      eyebrow="Who it helps"
      title="Readable heat for people who need it."
      lead="Thermal imagery is valuable but hard to interpret. IRVision makes it accessible, and says how far to trust it."
    >
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {CASES.map((c, i) => (
          <motion.div
            key={c.title}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.06, duration: 0.6 }}
            className="card group p-5 transition hover:border-neon/30"
          >
            <span className="grid size-11 place-items-center rounded-2xl bg-neon-soft text-neon transition group-hover:neon-glow">
              <c.icon className="size-5" strokeWidth={1.8} />
            </span>
            <h3 className="mt-3 text-lg font-semibold tracking-tight">{c.title}</h3>
            <p className="mt-2 text-[14.5px] leading-relaxed text-muted">{c.text}</p>
          </motion.div>
        ))}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="card border-amber/25 p-5"
        >
          <span className="grid size-11 place-items-center rounded-2xl bg-amber/10 text-amber">
            <Building2 className="size-5" strokeWidth={1.8} />
          </span>
          <h3 className="mt-3 text-lg font-semibold tracking-tight">Not yet: urban mapping</h3>
          <p className="mt-2 text-[14.5px] leading-relaxed text-muted">
            Our own validation shows dense built-up areas come out looking like vegetation. Don’t use the output to judge
            city extent. Fixing this is the next step.
          </p>
        </motion.div>
      </div>
    </Section>
  );
}
