// src/system/theme/useApplyTheme.ts
"use client";

import { useEffect } from "react";
import { useTheme } from "./theme-context";
import {ThemeRegistry} from "./theme-palette";
import {useSubsystem} from "./subsystem-context";


export const useApplyTheme = () => {
    const { mode } = useTheme();
    const { subsystem } = useSubsystem();

    useEffect(() => {
        const palette = ThemeRegistry[subsystem][mode];
        Object.entries(palette).forEach(([key, value]) => {
            document.documentElement.style.setProperty(`--${key}`, value);
        });
    }, [mode, subsystem]);
};
