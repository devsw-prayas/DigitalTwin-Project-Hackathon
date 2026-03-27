"use client";

import { motion } from "framer-motion";
import { orbitron } from "@/app/system/font/fonts";

export default function Hero() {
    return (
        <section className="w-full py-40 px-6">
            <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">

                {/* LEFT — Text */}
                <div className="flex flex-col gap-8">
                    <h2 className={`${orbitron.className} text-3xl md:text-4xl font-semibold text-white`}>
                        High-Performance Computing | Rendering | Research
                    </h2>

                    <p className="text-lg text-white/60 max-w-xl leading-relaxed">
                        An independent studio building graphics systems,
                        high-performance computation,
                        and low-level engine infrastructure.
                    </p>

                    <div className="flex gap-6 pt-4">
                        <a
                            href="/projects"
                            className="px-8 py-3 rounded-xl bg-white/10 backdrop-blur-md
                     border border-white/20 text-white font-medium
                     hover:bg-white/20 transition-all"
                        >
                            Explore Projects
                        </a>

                        <a
                            href="/docs"
                            className="px-8 py-3 rounded-xl bg-white text-black font-semibold
                     hover:bg-neutral-200 transition-all"
                        >
                            Visit Docs
                        </a>
                    </div>
                </div>

                {/* RIGHT — Visual Anchor */}
                <div className="relative">
                    <div className="rounded-4xl overflow-hidden shadow-2xl">
                        <img
                            src="/hro.png"
                            alt=""
                            className="w-full h-full object-cover"
                        />
                    </div>
                </div>

            </div>
        </section>
    );
}
