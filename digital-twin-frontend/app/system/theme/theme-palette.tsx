export const ThemeRegistry = {

    Home: {
        dark: {
            background: "linear-gradient(160deg, #0c1222 0%, #151d35 35%, #1a2744 70%, #0f1a2e 100%)",
            accent: "#60a5fa",
            textPrimary: "#f1f5f9",
            textMuted: "#64748b",
            surface: "rgba(96, 165, 250, 0.07)",
        },
        light: {
            background: "linear-gradient(160deg, #ffffff 0%, #e2e8f0 40%, #94a3b8 100%)",
            accent: "#2563eb",
            textPrimary: "#0f172a",
            textMuted: "#475569",
            surface: "rgba(37, 99, 235, 0.08)",
        },
    },

    Twin: {
        dark: {
            background: "linear-gradient(165deg, #030712 0%, #0c1929 30%, #0f2847 60%, #0a1628 100%)",
            accent: "#22d3ee",
            textPrimary: "#ecfeff",
            textMuted: "#67e8f9",
            surface: "rgba(34, 211, 238, 0.08)",
        },
        light: {
            background: "linear-gradient(165deg, #ffffff 0%, #cffafe 40%, #22d3ee 100%)",
            accent: "#0891b2",
            textPrimary: "#0f172a",
            textMuted: "#0e7490",
            surface: "rgba(8, 145, 178, 0.08)",
        },
    },

    Insights: {
        dark: {
            background: "linear-gradient(160deg, #022c22 0%, #064e3b 30%, #065f46 60%, #014737 100%)",
            accent: "#10b981",
            textPrimary: "#ecfdf5",
            textMuted: "#6ee7b7",
            surface: "rgba(16, 185, 129, 0.08)",
        },
        light: {
            background: "linear-gradient(160deg, #ffffff 0%, #d1fae5 40%, #10b981 100%)",
            accent: "#059669",
            textPrimary: "#0f172a",
            textMuted: "#166534",
            surface: "rgba(5, 150, 105, 0.08)",
        },
    },

    Predictions: {
        dark: {
            background: "linear-gradient(160deg, #1c1917 0%, #292524 25%, #44403c 60%, #1c1917 100%)",
            accent: "#f59e0b",
            textPrimary: "#fefce8",
            textMuted: "#fcd34d",
            surface: "rgba(245, 158, 11, 0.10)",
        },
        light: {
            background: "linear-gradient(160deg, #ffffff 0%, #fef3c7 40%, #f59e0b 100%)",
            accent: "#d97706",
            textPrimary: "#0f172a",
            textMuted: "#92400e",
            surface: "rgba(217, 119, 6, 0.08)",
        },
    },

    Profile: {
        dark: {
            background: "linear-gradient(165deg, #18181b 0%, #27272a 30%, #3f3f46 60%, #1f1f23 100%)",
            accent: "#a78bfa",
            textPrimary: "#faf5ff",
            textMuted: "#c4b5fd",
            surface: "rgba(167, 139, 250, 0.08)",
        },
        light: {
            background: "linear-gradient(165deg, #ffffff 0%, #e9d5ff 40%, #a78bfa 100%)",
            accent: "#7c3aed",
            textPrimary: "#0f172a",
            textMuted: "#6b21a8",
            surface: "rgba(124, 58, 237, 0.08)",
        },
    },

} as const;