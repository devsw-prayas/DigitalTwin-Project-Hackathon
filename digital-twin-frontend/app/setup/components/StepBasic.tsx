"use client";

import { User } from "lucide-react";
import { SetupData } from "./types";

interface StepBasicProps {
    data: SetupData;
    setData: (data: SetupData) => void;
    isDark: boolean;
}

export default function StepBasic({ data, setData, isDark }: StepBasicProps) {
    const inputStyle = {
        width: "100%",
        padding: "0.75rem 1rem",
        background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.03)",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
        borderRadius: "10px",
        color: isDark ? "#fff" : "#1a1a1a",
        fontSize: "1rem",
        outline: "none",
    };

    const chipStyle = (selected: boolean) => ({
        padding: "0.6rem 1rem",
        background: selected 
            ? "linear-gradient(135deg, rgba(56,189,248,0.2) 0%, rgba(129,140,248,0.2) 100%)"
            : (isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)"),
        border: `1px solid ${selected 
            ? "rgba(56,189,248,0.5)" 
            : (isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)")}`,
        borderRadius: "8px",
        color: isDark ? "#fff" : "#1a1a1a",
        cursor: "pointer",
        transition: "all 0.2s ease",
        fontSize: "0.9rem",
    });

    return (
        <div>
            <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
                <User size={32} style={{ color: "#38bdf8", marginBottom: "0.5rem" }} />
                <h2 style={{
                    fontSize: "1.35rem",
                    fontWeight: 700,
                    color: isDark ? "#fff" : "#1a1a1a",
                    marginBottom: "0.25rem",
                }}>
                    Basic Information
                </h2>
                <p style={{
                    color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
                    fontSize: "0.9rem",
                }}>
                    This helps calibrate your baseline
                </p>
            </div>

            <div style={{ marginBottom: "1.25rem" }}>
                <label style={{
                    display: "block",
                    color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)",
                    fontSize: "0.85rem",
                    marginBottom: "0.5rem",
                }}>
                    Age
                </label>
                <input
                    type="number"
                    placeholder="Your age"
                    value={data.age}
                    onChange={(e) => setData({ ...data, age: e.target.value })}
                    style={inputStyle}
                />
            </div>

            <div>
                <label style={{
                    display: "block",
                    color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)",
                    fontSize: "0.85rem",
                    marginBottom: "0.5rem",
                }}>
                    Sex
                </label>
                <div style={{ display: "flex", gap: "0.75rem" }}>
                    {['Male', 'Female', 'Other'].map((s) => (
                        <button
                            key={s}
                            onClick={() => setData({ ...data, sex: s.toLowerCase() })}
                            style={chipStyle(data.sex === s.toLowerCase())}
                        >
                            {s}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}
