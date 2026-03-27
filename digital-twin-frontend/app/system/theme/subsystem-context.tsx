"use client";

import React, { createContext, useContext, useState } from "react";
import type { SubsystemKey } from "./theme-palette";

interface SubsystemContextType {
    subsystem: SubsystemKey;
    setSubsystem: (s: SubsystemKey) => void;
}

const SubsystemContext = createContext<SubsystemContextType>({
    subsystem: "Home",
    setSubsystem: () => {},
});

export const SubsystemProvider = ({ children }: { children: React.ReactNode }) => {
    const [subsystem, setSubsystem] = useState<SubsystemKey>("Home");

    return (
        <SubsystemContext.Provider value={{ subsystem, setSubsystem }}>
            {children}
        </SubsystemContext.Provider>
    );
};

export const useSubsystem = () => useContext(SubsystemContext);
