"use client";

import {useEffect, useState} from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useTheme } from "@/app/system/theme/theme-context";
import Link from "next/link";
import {useRouter} from "next/navigation";
import {useSubsystem} from "@/app/system/theme/subsystem-context";

export default function ProfilePage() {
    const { mode } = useTheme();
    const [isLogin, setIsLogin] = useState(true);
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [name, setName] = useState("");

    const {setSubsystem} = useSubsystem();
    const isDark = mode === "dark";

    // Fake redirect - always false for now
    const isAuthenticated = true;

    const router = useRouter();

    useEffect(() => {
        if (isAuthenticated) {
            router.push("/setup");
        }
        setSubsystem("Profile");
    }, [isAuthenticated, router, setSubsystem]);

    const inputStyle = {
        width: "100%",
        padding: "0.7rem 0.875rem",
        background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.03)",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
        borderRadius: "10px",
        color: isDark ? "#fff" : "#1a1a1a",
        fontSize: "0.9rem",
        outline: "none",
        transition: "border-color 0.2s ease, background 0.2s ease",
    };

    const buttonPrimary = {
        width: "100%",
        padding: "0.7rem",
        background: "linear-gradient(135deg, #38bdf8 0%, #818cf8 100%)",
        border: "none",
        borderRadius: "10px",
        color: "#fff",
        fontSize: "0.9rem",
        fontWeight: 600,
        cursor: "pointer",
        transition: "opacity 0.2s ease, transform 0.2s ease",
    };

    const buttonSecondary = {
        width: "100%",
        padding: "0.7rem",
        background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.03)",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
        borderRadius: "10px",
        color: isDark ? "#fff" : "#1a1a1a",
        fontSize: "0.9rem",
        fontWeight: 500,
        cursor: "pointer",
        transition: "background 0.2s ease",
    };

    return (
        <div style={{
            minHeight: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "2rem",
        }}>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                style={{
                    width: "100%",
                    maxWidth: "360px",
                }}
            >
                {/* Card - Glassmorphism */}
                <div style={{
                    background: isDark
                        ? "rgba(255,255,255,0.05)"
                        : "rgba(255,255,255,0.6)",
                    backdropFilter: "blur(24px) saturate(150%)",
                    WebkitBackdropFilter: "blur(24px) saturate(150%)",
                    border: `1px solid ${isDark
                        ? "rgba(255,255,255,0.1)"
                        : "rgba(255,255,255,0.5)"}`,
                    borderRadius: "20px",
                    padding: "1.75rem",
                    boxShadow: isDark
                        ? "0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1)"
                        : "0 8px 32px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.5)",
                }}>
                    {/* Header */}
                    <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
                        <h1 style={{
                            fontSize: "1.35rem",
                            fontWeight: 700,
                            color: isDark ? "#fff" : "#1a1a1a",
                            marginBottom: "0.5rem",
                            fontFamily: "var(--font-oxanium)",
                        }}>
                            {isLogin ? "Welcome Back" : "Create Account"}
                        </h1>
                        <p style={{
                            color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
                            fontSize: "0.85rem",
                        }}>
                            {isLogin
                                ? "Sign in to access your digital twin"
                                : "Start your health journey today"
                            }
                        </p>
                    </div>

                    {/* Tab Switcher */}
                    <div style={{
                        display: "flex",
                        background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)",
                        borderRadius: "10px",
                        padding: "3px",
                        marginBottom: "1.25rem",
                    }}>
                        <button
                            onClick={() => setIsLogin(true)}
                            style={{
                                flex: 1,
                                padding: "0.625rem",
                                background: isLogin
                                    ? (isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.06)")
                                    : "transparent",
                                border: "none",
                                borderRadius: "8px",
                                color: isLogin
                                    ? (isDark ? "#fff" : "#1a1a1a")
                                    : (isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)"),
                                fontSize: "0.9rem",
                                fontWeight: isLogin ? 600 : 500,
                                cursor: "pointer",
                                transition: "all 0.2s ease",
                            }}
                        >
                            Login
                        </button>
                        <button
                            onClick={() => setIsLogin(false)}
                            style={{
                                flex: 1,
                                padding: "0.625rem",
                                background: !isLogin
                                    ? (isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.06)")
                                    : "transparent",
                                border: "none",
                                borderRadius: "8px",
                                color: !isLogin
                                    ? (isDark ? "#fff" : "#1a1a1a")
                                    : (isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)"),
                                fontSize: "0.9rem",
                                fontWeight: !isLogin ? 600 : 500,
                                cursor: "pointer",
                                transition: "all 0.2s ease",
                            }}
                        >
                            Sign Up
                        </button>
                    </div>

                    {/* Form */}
                    <form onSubmit={(e) => e.preventDefault()}>
                        <AnimatePresence mode="wait">
                            {!isLogin && (
                                <motion.div
                                    key="name"
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: "auto" }}
                                    exit={{ opacity: 0, height: 0 }}
                                    transition={{ duration: 0.2 }}
                                    style={{ marginBottom: "1rem", overflow: "hidden" }}
                                >
                                    <label style={{
                                        display: "block",
                                        color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)",
                                        fontSize: "0.85rem",
                                        marginBottom: "0.5rem",
                                        fontWeight: 500,
                                    }}>
                                        Name
                                    </label>
                                    <input
                                        type="text"
                                        value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        placeholder="Your name"
                                        style={inputStyle}
                                    />
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Email */}
                        <div style={{ marginBottom: "1rem" }}>
                            <label style={{
                                display: "block",
                                color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)",
                                fontSize: "0.85rem",
                                marginBottom: "0.5rem",
                                fontWeight: 500,
                            }}>
                                Email
                            </label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="you@example.com"
                                style={inputStyle}
                            />
                        </div>

                        {/* Password */}
                        <div style={{ marginBottom: "1.25rem" }}>
                            <label style={{
                                display: "block",
                                color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)",
                                fontSize: "0.85rem",
                                marginBottom: "0.5rem",
                                fontWeight: 500,
                            }}>
                                Password
                            </label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="••••••••"
                                style={inputStyle}
                            />
                        </div>

                        {/* Submit */}
                        <motion.button
                            type="submit"
                            style={buttonPrimary}
                            whileHover={{ opacity: 0.9 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            {isLogin ? "Sign In" : "Create Account"}
                        </motion.button>

                        {/* Divider */}
                        <div style={{
                            display: "flex",
                            alignItems: "center",
                            margin: "1rem 0",
                            gap: "0.75rem",
                        }}>
                            <div style={{
                                flex: 1,
                                height: "1px",
                                background: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
                            }} />
                            <span style={{
                                color: isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)",
                                fontSize: "0.85rem",
                            }}>
                                or continue with
                            </span>
                            <div style={{
                                flex: 1,
                                height: "1px",
                                background: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
                            }} />
                        </div>

                        {/* Social buttons */}
                        <motion.button
                            type="button"
                            style={buttonSecondary}
                            whileHover={{ background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)" }}
                            whileTap={{ scale: 0.98 }}
                        >
                            <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem" }}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                                </svg>
                                Google
                            </span>
                        </motion.button>
                    </form>

                    {/* Back link */}
                    <div style={{ textAlign: "center", marginTop: "1rem" }}>
                        <Link
                            href="/"
                            style={{
                                color: isDark ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)",
                                fontSize: "0.85rem",
                                textDecoration: "none",
                            }}
                        >
                            ← Back to home
                        </Link>
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
