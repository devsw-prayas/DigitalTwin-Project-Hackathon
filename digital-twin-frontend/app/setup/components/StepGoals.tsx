"use client";

import { Target } from "lucide-react";
import { SetupData, GOALS } from "./types";

interface StepGoalsProps {
    data: SetupData;
    setData: (data: SetupData) => void;
    isDark: boolean;
}

export default function StepGoals({ data, setData, isDark }: StepGoalsProps) {
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

    const toggleGoal = (id: string) => {
        const goals = data.goals.includes(id)
            ? data.goals.filter(g => g !== id)
            : [...data.goals, id];
        setData({ ...data, goals });
    };

    return (
        <div>
            <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
                <Target size={32} style={{ color: "#22c55e", marginBottom: "0.5rem" }} />
                <h2 style={{
                    fontSize: "1.35rem",
                    fontWeight: 700,
                    color: isDark ? "#fff" : "#1a1a1a",
                    marginBottom: "0.25rem",
                }}>
                    Your Goals
                </h2>
                <p style={{
                    color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
                    fontSize: "0.9rem",
                }}>
                    What do you want to improve?
                </p>
            </div>

            <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: "0.5rem",
            }}>
                {GOALS.map((g) => (
                    <button
                        key={g.id}
                        onClick={() => toggleGoal(g.id)}
                        style={chipStyle(data.goals.includes(g.id))}
                    >
                        <span style={{ marginRight: "0.5rem" }}>{g.icon}</span>
                        {g.label}
                    </button>
                ))}
            </div>
        </div>
    );
}
