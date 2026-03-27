"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useTheme } from "@/app/system/theme/theme-context";
import { ChevronRight, ChevronLeft } from "lucide-react";

import { SetupData } from "./components/types";
import ProgressBar from "./components/ProgressBar";
import StepBasic from "./components/StepBasic";
import StepConditions from "./components/StepConditions";
import StepGoals from "./components/StepGoals";
import StepDataSource from "./components/StepDataSource";

export default function SetupPage() {
    const { mode } = useTheme();
    const isDark = mode === "dark";

    const [step, setStep] = useState(1);
    const [data, setData] = useState<SetupData>({
        age: '',
        sex: '',
        conditions: [],
        goals: [],
        dataSource: null,
    });

    const totalSteps = 4;

    const isStepValid = () => {
        switch (step) {
            case 1: return data.age && data.sex;
            case 2: return true; // Conditions optional
            case 3: return data.goals.length > 0;
            case 4: return data.dataSource !== null;
            default: return false;
        }
    };

    const cardStyle = {
        background: isDark 
            ? "rgba(255,255,255,0.05)" 
            : "rgba(255,255,255,0.7)",
        backdropFilter: "blur(24px) saturate(150%)",
        WebkitBackdropFilter: "blur(24px) saturate(150%)",
        border: `1px solid ${isDark 
            ? "rgba(255,255,255,0.1)" 
            : "rgba(255,255,255,0.5)"}`,
        borderRadius: "20px",
        padding: "2rem",
        boxShadow: isDark
            ? "0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1)"
            : "0 8px 32px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.5)",
    };

    const buttonPrimary = {
        padding: "0.75rem 1.5rem",
        background: "linear-gradient(135deg, #38bdf8 0%, #818cf8 100%)",
        border: "none",
        borderRadius: "10px",
        color: "#fff",
        fontSize: "0.95rem",
        fontWeight: 600,
        cursor: isStepValid() ? "pointer" : "not-allowed",
        opacity: isStepValid() ? 1 : 0.5,
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        transition: "opacity 0.2s ease",
    };

    const buttonSecondary = {
        padding: "0.75rem 1.5rem",
        background: "transparent",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)"}`,
        borderRadius: "10px",
        color: isDark ? "#fff" : "#1a1a1a",
        fontSize: "0.95rem",
        fontWeight: 500,
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
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
                style={{
                    width: "100%",
                    maxWidth: "480px",
                }}
            >
                <ProgressBar step={step} totalSteps={totalSteps} isDark={isDark} />

                <div style={cardStyle}>
                    <AnimatePresence mode="wait">
                        {step === 1 && (
                            <motion.div
                                key="step1"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                transition={{ duration: 0.2 }}
                            >
                                <StepBasic data={data} setData={setData} isDark={isDark} />
                            </motion.div>
                        )}

                        {step === 2 && (
                            <motion.div
                                key="step2"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                transition={{ duration: 0.2 }}
                            >
                                <StepConditions data={data} setData={setData} isDark={isDark} />
                            </motion.div>
                        )}

                        {step === 3 && (
                            <motion.div
                                key="step3"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                transition={{ duration: 0.2 }}
                            >
                                <StepGoals data={data} setData={setData} isDark={isDark} />
                            </motion.div>
                        )}

                        {step === 4 && (
                            <motion.div
                                key="step4"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                transition={{ duration: 0.2 }}
                            >
                                <StepDataSource data={data} setData={setData} isDark={isDark} />
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Navigation */}
                    <div style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginTop: "1.75rem",
                        paddingTop: "1rem",
                        borderTop: `1px solid ${isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                    }}>
                        <button
                            onClick={() => setStep(step - 1)}
                            style={{
                                ...buttonSecondary,
                                visibility: step === 1 ? "hidden" : "visible",
                            }}
                        >
                            <ChevronLeft size={18} />
                            Back
                        </button>

                        {step < totalSteps ? (
                            <button
                                onClick={() => setStep(step + 1)}
                                disabled={!isStepValid()}
                                style={buttonPrimary}
                            >
                                Next
                                <ChevronRight size={18} />
                            </button>
                        ) : (
                            <button
                                onClick={() => {
                                    // Would redirect to /twin or manual input form
                                    console.log("Setup complete:", data);
                                }}
                                disabled={!isStepValid()}
                                style={buttonPrimary}
                            >
                                Complete Setup
                                <ChevronRight size={18} />
                            </button>
                        )}
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
