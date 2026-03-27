"use client";

interface ProgressBarProps {
    step: number;
    totalSteps: number;
    isDark: boolean;
}

export default function ProgressBar({ step, totalSteps, isDark }: ProgressBarProps) {
    const progress = (step / totalSteps) * 100;

    return (
        <div style={{ marginBottom: "1.5rem" }}>
            <div style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "0.5rem",
                fontSize: "0.85rem",
                color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
            }}>
                <span>Step {step} of {totalSteps}</span>
                <span>{Math.round(progress)}%</span>
            </div>
            <div style={{
                height: "4px",
                background: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
                borderRadius: "2px",
                overflow: "hidden",
            }}>
                <div style={{
                    height: "100%",
                    width: `${progress}%`,
                    background: "linear-gradient(90deg, #38bdf8 0%, #818cf8 100%)",
                    transition: "width 0.3s ease",
                }} />
            </div>
        </div>
    );
}
