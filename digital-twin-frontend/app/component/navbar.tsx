'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import '@/app/globals.css';
import { useApplyTheme } from "@/app/system/theme/apply-theme";
import { useTheme } from "@/app/system/theme/theme-context";
import { useSubsystem } from "@/app/system/theme/subsystem-context";
import { ThemeRegistry } from "@/app/system/theme/theme-palette";
import { Sun, Moon, Menu, X } from "lucide-react";
import { orbitron } from "@/app/system/font/fonts";

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
    } : null;
}

const navItems = ["Home", "Twin", "Insights", "Predictions", "Profile"];

export default function NavBar() {
    useApplyTheme();
    const pathname = usePathname();
    const { mode, toggleMode } = useTheme();
    const { subsystem } = useSubsystem();
    const [hovered, setHovered] = useState<string | null>(null);
    const [mobileOpen, setMobileOpen] = useState(false);

    const palette = ThemeRegistry[subsystem][mode];
    const accentColor = palette.accent;
    const isDark = mode === "dark";

    // Matte dark — 28% of accent brightness, gives a visible hue not near-black
    const rgb = hexToRgb(accentColor);
    const matteBg = rgb
        ? `rgb(${Math.round(rgb.r * 0.28)}, ${Math.round(rgb.g * 0.28)}, ${Math.round(rgb.b * 0.28)})`
        : "#1a1a22";

    // Light mode bg
    const lightBg = "rgba(255, 255, 255, 0.85)";
    const navBg = isDark ? matteBg : lightBg;

    // Text colors based on mode
    const textColor = isDark ? "#f0f0f0" : "#1a1a1a";
    const mutedTextColor = isDark ? "rgba(255,255,255,0.75)" : "rgba(0,0,0,0.7)";
    const iconMutedColor = isDark ? "rgba(255,255,255,0.70)" : "rgba(0,0,0,0.65)";

    // Check if item is active
    const isActive = (item: string) => {
        const itemPath = item.toLowerCase() === "home" ? "/" : `/${item.toLowerCase()}`;
        return pathname === itemPath;
    };

    return (
        <div style={{
            position: "fixed",
            top: "16px",
            left: 0,
            right: 0,
            zIndex: 50,
            display: "flex",
            justifyContent: "center",
            pointerEvents: "none",
            padding: "0 16px",
        }}>
            <motion.nav
                initial={{ y: -60, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
                style={{
                    pointerEvents: "auto",
                    background: navBg,
                    backdropFilter: "blur(20px) saturate(150%)",
                    WebkitBackdropFilter: "blur(20px) saturate(150%)",
                    border: `1px solid ${isDark
                        ? `color-mix(in srgb, ${accentColor} 35%, transparent)`
                        : "rgba(0,0,0,0.1)"}`,
                    borderRadius: "16px",
                    boxShadow: isDark
                        ? `0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px color-mix(in srgb, ${accentColor} 10%, transparent)`
                        : "0 8px 32px rgba(0,0,0,0.1)",
                    transition: "background 0.5s ease, border-color 0.5s ease",
                    width: "100%",
                    maxWidth: "1100px",
                }}
            >
                <div className="navbar-container" style={{ padding: "0.6rem 1.75rem" }}>
                    <div className="navbar-left">
                        <Link
                            href="/"
                            className={`${orbitron.className} navbar-logo`}
                            style={{
                                color: hovered === "logo" ? accentColor : textColor,
                                transition: "color 0.2s ease",
                            }}
                            onMouseEnter={() => setHovered("logo")}
                            onMouseLeave={() => setHovered(null)}
                        >
                            Digi-Twin
                        </Link>
                    </div>

                    {/* Desktop Menu */}
                    <ul className="navbar-menu" style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
                        {navItems.map((item) => {
                            const active = isActive(item);
                            return (
                                <li key={item} style={{ position: "relative" }}>
                                    <Link
                                        href={item.toLowerCase() === "home" ? "/" : `/${item.toLowerCase()}`}
                                        className="navbar-link"
                                        style={{
                                            color: active
                                                ? accentColor
                                                : hovered === item
                                                    ? accentColor
                                                    : mutedTextColor,
                                            transition: "color 0.2s ease",
                                            fontWeight: active ? 600 : 400,
                                        }}
                                        onMouseEnter={() => setHovered(item)}
                                        onMouseLeave={() => setHovered(null)}
                                    >
                                        {item}
                                    </Link>
                                    {/* Active indicator dot */}
                                    {active && (
                                        <motion.div
                                            layoutId="activeIndicator"
                                            style={{
                                                position: "absolute",
                                                bottom: -4,
                                                left: "50%",
                                                transform: "translateX(-50%)",
                                                width: 4,
                                                height: 4,
                                                borderRadius: "50%",
                                                background: accentColor,
                                            }}
                                            transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                        />
                                    )}
                                </li>
                            );
                        })}
                        <button
                            onClick={toggleMode}
                            onMouseEnter={() => setHovered("toggle")}
                            onMouseLeave={() => setHovered(null)}
                            aria-label="Toggle theme"
                            style={{
                                color: hovered === "toggle" ? accentColor : iconMutedColor,
                                transition: "color 0.2s ease",
                                background: "none",
                                border: "none",
                                cursor: "pointer",
                                padding: 0,
                                display: "flex",
                                alignItems: "center",
                            }}
                        >
                            {isDark ? <Sun size={18} /> : <Moon size={18} />}
                        </button>
                    </ul>

                    {/* Mobile Menu Button */}
                    <button
                        className="mobile-menu-btn"
                        onClick={() => setMobileOpen(!mobileOpen)}
                        aria-label="Toggle menu"
                        style={{
                            display: "none",
                            background: "none",
                            border: "none",
                            cursor: "pointer",
                            padding: "0.25rem",
                            color: textColor,
                        }}
                    >
                        {mobileOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>
                </div>

                {/* Mobile Dropdown */}
                <AnimatePresence>
                    {mobileOpen && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
                            style={{
                                overflow: "hidden",
                                borderTop: `1px solid ${isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            }}
                        >
                            <ul style={{
                                display: "flex",
                                flexDirection: "column",
                                padding: "1rem 1.75rem",
                                margin: 0,
                                listStyle: "none",
                                gap: "0.75rem",
                            }}>
                                {navItems.map((item) => {
                                    const active = isActive(item);
                                    return (
                                        <li key={`mobile-${item}`}>
                                            <Link
                                                href={item.toLowerCase() === "home" ? "/" : `/${item.toLowerCase()}`}
                                                style={{
                                                    color: active ? accentColor : mutedTextColor,
                                                    fontWeight: active ? 600 : 400,
                                                    textDecoration: "none",
                                                    fontSize: "1rem",
                                                }}
                                                onClick={() => setMobileOpen(false)}
                                            >
                                                {item}
                                            </Link>
                                        </li>
                                    );
                                })}
                                <li>
                                    <button
                                        onClick={() => {
                                            toggleMode();
                                        }}
                                        style={{
                                            color: iconMutedColor,
                                            background: "none",
                                            border: "none",
                                            cursor: "pointer",
                                            padding: 0,
                                            display: "flex",
                                            alignItems: "center",
                                            gap: "0.5rem",
                                            fontSize: "1rem",
                                        }}
                                    >
                                        {isDark ? <Sun size={18} /> : <Moon size={18} />}
                                        <span>{isDark ? "Light Mode" : "Dark Mode"}</span>
                                    </button>
                                </li>
                            </ul>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.nav>

            <style jsx global>{`
                @media (max-width: 768px) {
                    .navbar-menu {
                        display: none !important;
                    }
                    .mobile-menu-btn {
                        display: flex !important;
                    }
                    .navbar-container {
                        display: flex !important;
                        justify-content: space-between !important;
                        align-items: center !important;
                    }
                    .navbar-left {
                        margin-right: 0 !important;
                    }
                }
            `}</style>
        </div>
    );
}