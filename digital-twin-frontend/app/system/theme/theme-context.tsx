"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

type ThemeMode = "dark" | "light";
interface ThemeContextType {
    mode: ThemeMode;
    toggleMode: () => void;
}

const ThemeContext = createContext<ThemeContextType>({
    mode: "dark",
    toggleMode: () => {},
});

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
    const [mode, setMode] = useState<ThemeMode>("dark"); // Default safe value

    // Load stored mode *after* mounting (browser only)
    useEffect(() => {
        const storedMode = typeof window !== "undefined"
            ? (localStorage.getItem("themeMode") as ThemeMode)
            : null;

        if (storedMode) {
            setMode(storedMode);
        }
    }, []);

    // Persist + apply mode whenever it changes
    useEffect(() => {
        if (typeof window !== "undefined") {
            document.documentElement.setAttribute("data-theme", mode);
            localStorage.setItem("themeMode", mode);
        }
    }, [mode]);

    const toggleMode = () => setMode((m) => (m === "dark" ? "light" : "dark"));

    return (
        <ThemeContext.Provider value={{ mode, toggleMode }}>
            {children}
        </ThemeContext.Provider>
    );
};

export const useTheme = () => useContext(ThemeContext);
