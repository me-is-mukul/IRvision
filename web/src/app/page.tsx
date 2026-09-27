import { Footer } from "@/components/Footer";
import { Hero } from "@/components/Hero";
import { HowItWorks } from "@/components/HowItWorks";
import { ModeProvider } from "@/components/ModeProvider";
import { Nav } from "@/components/Nav";
import { Playground } from "@/components/Playground";
import { Report } from "@/components/Report";
import { UseCases } from "@/components/UseCases";

export default function Home() {
  return (
    <ModeProvider>
      <Nav />
      <main>
        <Hero />
        <HowItWorks />
        <Playground />
        <Report />
        <UseCases />
      </main>
      <Footer />
    </ModeProvider>
  );
}
