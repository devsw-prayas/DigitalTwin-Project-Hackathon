"use client";

import { createContext, useContext, useState, ReactNode } from 'react';
import { AgentMode, getActiveAgents, AgentConfig } from '@/app/agents/agents';

interface AgentModeContextType {
    mode: AgentMode;
    setMode: (mode: AgentMode) => void;
    agents: AgentConfig[];
    agentCount: number;
}

const AgentModeContext = createContext<AgentModeContextType | undefined>(undefined);

export function AgentModeProvider({ children }: { children: ReactNode }) {
    const [mode, setMode] = useState<AgentMode>('mvp');
    const agents = getActiveAgents(mode);
    const agentCount = agents.length;

    return (
        <AgentModeContext.Provider value={{ mode, setMode, agents, agentCount }}>
            {children}
        </AgentModeContext.Provider>
    );
}

export function useAgentMode() {
    const ctx = useContext(AgentModeContext);
    if (!ctx) throw new Error('useAgentMode must be inside AgentModeProvider');
    return ctx;
}
