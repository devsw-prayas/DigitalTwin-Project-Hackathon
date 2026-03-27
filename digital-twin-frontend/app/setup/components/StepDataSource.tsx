"use client";

import { motion } from "framer-motion";
import { SetupData, DATA_SOURCES } from "./types";

interface StepDataSourceProps {
    data: SetupData;
    setData: (data: SetupData) => void;
    isDark: boolean;
}

export default function StepDataSource({ data, setData, isDark }: StepDataSourceProps) {
    return (
        <div>
            <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📱</div>
                <h2 style={{
                    fontSize: "1.35rem",
                    fontWeight: 700,
                    color: isDark ? "#fff" : "#1a1a1a",
                    marginBottom: "0.25rem",
                }}>
                    Connect Your Data
                </h2>
                <p style={{
                    color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
                    fontSize: "0.9rem",
                }}>
                    How would you like to track your health?
                </p>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginBottom: "1rem" }}>
                {DATA_SOURCES.map((source) => (
                    <div
                        key={source.id}
                        onClick={() => setData({ ...data, dataSource: source.id })}
                        style={{
                            padding: "1rem 1.25rem",
                            background: data.dataSource === source.id 
                                ? `linear-gradient(135deg, ${source.color}15 0%, ${source.color}10 100%)`
                                : (isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)"),
                            border: `1px solid ${data.dataSource === source.id 
                                ? `${source.color}60` 
                                : (isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)")}`,
                            borderRadius: "12px",
                            cursor: "pointer",
                            transition: "all 0.2s ease",
                            display: "flex",
                            alignItems: "center",
                            gap: "1rem",
                        }}
                    >
                        <span style={{ fontSize: "1.75rem" }}>{source.logo}</span>
                        <div style={{ flex: 1 }}>
                            <div style={{
                                fontWeight: 600,
                                color: isDark ? "#fff" : "#1a1a1a",
                                marginBottom: "0.125rem",
                            }}>
                                {source.label}
                            </div>
                            <div style={{
                                fontSize: "0.8rem",
                                color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
                            }}>
                                {source.description}
                            </div>
                        </div>
                        <div style={{
                            fontSize: "0.75rem",
                            padding: "0.25rem 0.5rem",
                            background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)",
                            borderRadius: "4px",
                            color: isDark ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.5)",
                        }}>
                            {source.platform}
                        </div>
                    </div>
                ))}
            </div>

            {/* Apple Health connection */}
            {data.dataSource === 'apple' && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    style={{ overflow: "hidden" }}
                >
                    <ConnectionCard
                        logo="🍎"
                        title="Connect Apple Health"
                        subtitle="We'll read: steps, heart rate, HRV, sleep, SpO2"
                        buttonLabel="Sign in with Apple"
                        buttonBg="#000"
                        buttonColor="#fff"
                        isDark={isDark}
                    />
                </motion.div>
            )}

            {/* Google Health Connect */}
            {data.dataSource === 'google' && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    style={{ overflow: "hidden" }}
                >
                    <ConnectionCard
                        logo="🏥"
                        title="Connect Health Connect"
                        subtitle="Syncs with Fitbit, Samsung Health, Whoop & more"
                        buttonLabel="Sign in with Google"
                        buttonBg="#fff"
                        buttonColor="#1a1a1a"
                        isGoogle
                        isDark={isDark}
                    />
                </motion.div>
            )}

            {/* Manual Entry info */}
            {data.dataSource === 'manual' && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    style={{ overflow: "hidden" }}
                >
                    <div style={{
                        padding: "1rem",
                        background: isDark ? "rgba(56,189,248,0.1)" : "rgba(56,189,248,0.08)",
                        borderRadius: "12px",
                        border: "1px solid rgba(56,189,248,0.3)",
                    }}>
                        <div style={{
                            fontWeight: 600,
                            marginBottom: "0.5rem",
                            color: isDark ? "#fff" : "#1a1a1a",
                        }}>
                            What you&apos;ll track:
                        </div>
                        <div style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(2, 1fr)",
                            gap: "0.5rem",
                            fontSize: "0.85rem",
                            color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)",
                        }}>
                            <div>😴 Sleep duration & quality</div>
                            <div>💓 Resting heart rate</div>
                            <div>🚶 Daily steps</div>
                            <div>⚡ Energy level</div>
                            <div>😰 Stress level</div>
                            <div>🏃 Activity type</div>
                        </div>
                        <div style={{
                            marginTop: "0.75rem",
                            padding: "0.5rem 0.75rem",
                            background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)",
                            borderRadius: "6px",
                            fontSize: "0.8rem",
                            color: isDark ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)",
                        }}>
                            💡 Takes ~30 seconds daily. We&apos;ll send gentle reminders.
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
}

// Sub-component for connection cards
function ConnectionCard({
    logo,
    title,
    subtitle,
    buttonLabel,
    buttonBg,
    buttonColor,
    isGoogle,
    isDark,
}: {
    logo: string;
    title: string;
    subtitle: string;
    buttonLabel: string;
    buttonBg: string;
    buttonColor: string;
    isGoogle?: boolean;
    isDark: boolean;
}) {
    return (
        <div style={{
            padding: "1rem",
            background: isDark ? "rgba(0,0,0,0.2)" : "rgba(0,0,0,0.03)",
            borderRadius: "12px",
            border: "1px solid " + (isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"),
        }}>
            <div style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                marginBottom: "0.75rem",
            }}>
                <div style={{
                    width: "32px",
                    height: "32px",
                    background: isGoogle ? "#34A853" : buttonBg,
                    borderRadius: "8px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "1rem",
                }}>
                    {logo}
                </div>
                <div>
                    <div style={{ fontWeight: 600, color: isDark ? "#fff" : "#1a1a1a" }}>
                        {title}
                    </div>
                    <div style={{ fontSize: "0.8rem", color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}>
                        {subtitle}
                    </div>
                </div>
            </div>
            <button style={{
                width: "100%",
                padding: "0.6rem",
                background: buttonBg,
                border: isGoogle ? "1px solid #dadce0" : "none",
                borderRadius: "8px",
                color: buttonColor,
                fontWeight: 500,
                cursor: "pointer",
                fontSize: "0.9rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "0.5rem",
            }}>
                {isGoogle && (
                    <svg width="18" height="18" viewBox="0 0 24 24">
                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                    </svg>
                )}
                {buttonLabel}
            </button>
        </div>
    );
}
