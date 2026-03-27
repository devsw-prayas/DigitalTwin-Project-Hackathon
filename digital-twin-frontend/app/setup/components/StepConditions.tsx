"use client";

import { Heart } from "lucide-react";
import { SetupData, CONDITIONS } from "./types";

interface StepConditionsProps {
    data: SetupData;
    setData: (data: SetupData) => void;
    isDark: boolean;
}

export default function StepConditions({ data, setData, isDark }: StepConditionsProps) {
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

    const toggleCondition = (id: string) => {
        const conditions = data.conditions.includes(id)
            ? data.conditions.filter(c => c !== id)
            : [...data.conditions, id];
        setData({ ...data, conditions });
    };

    return (
        <div>
            <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
                <Heart size={32} style={{ color: "#ef4444", marginBottom: "0.5rem" }} />
                <h2 style={{
                    fontSize: "1.35rem",
                    fontWeight: 700,
                    color: isDark ? "#fff" : "#1a1a1a",
                    marginBottom: "0.25rem",
                }}>
                    Health Conditions
                </h2>
                <p style={{
                    color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
                    fontSize: "0.9rem",
                }}>
                    Select any that apply (optional)
                </p>
            </div>

            <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: "0.5rem",
            }}>
                {CONDITIONS.map((c) => (
                    <button
                        key={c.id}
                        onClick={() => toggleCondition(c.id)}
                        style={chipStyle(data.conditions.includes(c.id))}
                    >
                        {c.label}
                    </button>
                ))}
            </div>
        </div>
    );
}
