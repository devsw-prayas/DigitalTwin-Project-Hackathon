"use client";

import React, { useEffect, useRef } from "react";
import "@/app/globals.css";
import { useTheme } from "@/app/system/theme/theme-context";
import { useSubsystem } from "./subsystem-context";
import { ThemeRegistry } from "@/app/system/theme/theme-palette";
import { orbitron } from "@/app/system/font/fonts";

export const Background: React.FC = () => {
    const { mode } = useTheme();
    const { subsystem } = useSubsystem();
    const palette = ThemeRegistry[subsystem][mode];
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const mouseRef = useRef<{ x: number; y: number }>({ x: -9999, y: -9999 });
    const animRef = useRef<number>(0);
    const paletteRef = useRef(palette);
    const modeRef = useRef(mode);

    useEffect(() => {
        paletteRef.current = palette;
        modeRef.current = mode;
    }, [palette, mode]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const DOT_SPACING = 15;
        const DOT_RADIUS = 1.2;
        const GLOW_RADIUS = 50;

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener("resize", resize);

        const onMouseMove = (e: MouseEvent) => {
            mouseRef.current = { x: e.clientX, y: e.clientY };
        };
        const onMouseLeave = () => {
            mouseRef.current = { x: -9999, y: -9999 };
        };
        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseleave", onMouseLeave);

        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const currentPalette = paletteRef.current;
            const currentMode = modeRef.current;
            const { x: mx, y: my } = mouseRef.current;

            // Parse accent hex to rgb
            const hex = currentPalette.accent.replace("#", "");
            const ar = parseInt(hex.substring(0, 2), 16);
            const ag = parseInt(hex.substring(2, 4), 16);
            const ab = parseInt(hex.substring(4, 6), 16);

            const cols = Math.ceil(canvas.width / DOT_SPACING) + 1;
            const rows = Math.ceil(canvas.height / DOT_SPACING) + 1;

            for (let i = 0; i < cols; i++) {
                for (let j = 0; j < rows; j++) {
                    const dx = i * DOT_SPACING;
                    const dy = j * DOT_SPACING;
                    const dist = Math.sqrt((dx - mx) ** 2 + (dy - my) ** 2);
                    const proximity = Math.max(0, 1 - dist / GLOW_RADIUS);

                    if (currentMode === "dark") {
                        const alpha = 0.10 + proximity * 0.80;
                        const radius = DOT_RADIUS + proximity * 1.8;
                        ctx.beginPath();
                        ctx.arc(dx, dy, radius, 0, Math.PI * 2);
                        ctx.fillStyle = `rgba(${ar}, ${ag}, ${ab}, ${alpha})`;
                        ctx.fill();
                    } else {
                        const alpha = 0.15 + proximity * 0.50;
                        const radius = DOT_RADIUS + proximity * 1.2;
                        ctx.beginPath();
                        ctx.arc(dx, dy, radius, 0, Math.PI * 2);
                        ctx.fillStyle = `rgba(0, 0, 0, ${alpha})`;
                        ctx.fill();
                    }
                }
            }

            animRef.current = requestAnimationFrame(draw);
        };

        animRef.current = requestAnimationFrame(draw);

        return () => {
            cancelAnimationFrame(animRef.current);
            window.removeEventListener("resize", resize);
            window.removeEventListener("mousemove", onMouseMove);
            window.removeEventListener("mouseleave", onMouseLeave);
        };
    }, []);

    return (
        <div
            className="fixed inset-0 transition-all duration-700 ease-[cubic-bezier(0.4,0,0.2,1)]"
            style={{
                background: palette.background,
                zIndex: -10,
                isolation: "isolate",
            }}
        >
            {/* Dot grid — below watermark */}
            <canvas
                ref={canvasRef}
                style={{
                    position: "absolute",
                    inset: 0,
                    zIndex: -1000,
                    pointerEvents: "none",
                }}
            />

            {/* Watermark — above grid */}
            <div
                className={`${orbitron.className} absolute bottom-[2vh] left-[2vw] select-none pointer-events-none`}
                style={{
                    fontSize: "14vw",
                    fontWeight: 900,
                    lineHeight: 1,
                    letterSpacing: "-0.05em",
                    color: "var(--accent)",
                    opacity: 0.35,
                    mixBlendMode: "overlay",
                    userSelect: "none",
                    whiteSpace: "nowrap",
                    zIndex: 1000,
                }}
            >
                {subsystem}
            </div>
        </div>
    );
};