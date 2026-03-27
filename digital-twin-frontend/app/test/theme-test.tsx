"use client";

import {useApplyTheme} from "@/app/system/theme/apply-theme";
import {useTheme} from "@/app/system/theme/theme-context";
import {useSubsystem} from "@/app/system/theme/subsystem-context";
import {ThemeRegistry} from "@/app/system/theme/theme-palette";

export default function ThemeTestPage() {
    useApplyTheme();
    const { mode, toggleMode } = useTheme();
    const { subsystem, setSubsystem } = useSubsystem();

    const systems = Object.keys(ThemeRegistry) as (keyof typeof ThemeRegistry)[];

    return (
        <main className="min-h-screen flex items-center justify-center text-[var(--textPrimary)] transition-all duration-700">
            <div className="w-full max-w-5xl px-6 py-16 text-center space-y-10">
                {/* Header */}
                <div>
                    <h1 className="text-5xl font- tracking-tight mb-3">
                        Spectra Theme Showcase
                    </h1>
                    <p className="text-[var(--textMuted)] text-lg">
                        Visual test for the dynamic subsystem + theme switcher
                    </p>
                </div>

                {/* Subsystem Buttons */}
                <div className="flex flex-wrap justify-center gap-4">
                    {systems.map((sys) => (
                        <button
                            key={sys}
                            onClick={() => setSubsystem(sys)}
                            className={`px-5 py-2.5 rounded-lg border-2 font-semibold transition-all duration-500 ${
                                subsystem === sys
                                    ? "scale-105 shadow-lg"
                                    : "opacity-80 hover:opacity-100"
                            }`}
                            style={{
                                borderColor: "var(--accent)",
                                background:
                                    subsystem === sys ? "var(--accent)" : "var(--surface)",
                                color:
                                    subsystem === sys
                                        ? mode === "dark"
                                            ? "#000"
                                            : "#fff"
                                        : "var(--textPrimary)",
                            }}
                        >
                            {sys}
                        </button>
                    ))}
                </div>

                {/* Info Section */}
                <div className="space-y-1">
                    <p className="text-[var(--textMuted)] text-lg">
                        Active Subsystem:{" "}
                        <span className="font-bold text-[var(--accent)]">{subsystem}</span>
                    </p>
                    <p className="text-[var(--textMuted)] text-lg">
                        Current Mode:{" "}
                        <span className="font-bold text-[var(--accent)] capitalize">
              {mode}
            </span>
                    </p>
                </div>

                {/* Theme Toggle */}
                <button
                    onClick={toggleMode}
                    className="px-6 py-2.5 rounded-full font-semibold border-2 transition-all duration-500 shadow-md"
                    style={{
                        borderColor: "var(--accent)",
                        background: "var(--accent)",
                        color: mode === "dark" ? "#000" : "#fff",
                    }}
                >
                    Switch to {mode === "dark" ? "Light" : "Dark"} Mode
                </button>

                {/* Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-10">
                    {["Primary Features", "Data Processing", "System Integration"].map(
                        (title, i) => (
                            <div
                                key={i}
                                className="rounded-2xl p-6 text-left backdrop-blur-md transition-all duration-700"
                                style={{
                                    background: "var(--surface)",
                                    boxShadow: "0 6px 20px rgba(0,0,0,0.25)",
                                }}
                            >
                                <h3 className="text-2xl font-bold mb-3 flex items-center gap-2">
                  <span
                      className="inline-block w-3 h-3 rounded-full shadow-md"
                      style={{
                          background: "var(--accent)",
                          boxShadow: `0 0 12px var(--accent)`,
                      }}
                  />
                                    {title}
                                </h3>
                                <p className="text-[var(--textMuted)] leading-relaxed">
                                    Lorem ipsum dolor sit amet, consectetur adipiscing elit. In
                                    Spectra, this module handles the core logic behind{" "}
                                    {title.toLowerCase()}.
                                </p>
                            </div>
                        )
                    )}
                </div>

            </div>
        </main>
    );
}