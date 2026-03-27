// Agent configuration - extensible for MVP (2) or Full (8)

export type AgentId = 
    | 'cardio' 
    | 'mental' 
    | 'metabolic' 
    | 'recovery' 
    | 'immune' 
    | 'respiratory' 
    | 'hormonal' 
    | 'cog_fatigue';

export interface AgentConfig {
    id: AgentId;
    name: string;
    color: string;
    description: string;
    signals: string[]; // Key signals this agent focuses on
    alwaysOn: boolean; // Core vs sparse specialist
}

export const AGENT_DEFINITIONS: Record<AgentId, AgentConfig> = {
    cardio: {
        id: 'cardio',
        name: 'Cardiovascular',
        color: '#ef4444',
        description: 'Heart health and circulation patterns',
        signals: ['HRV', 'Resting HR', 'Recovery time'],
        alwaysOn: true,
    },
    mental: {
        id: 'mental',
        name: 'Mental Health',
        color: '#8b5cf6',
        description: 'Cognitive and emotional wellbeing',
        signals: ['Sleep quality', 'Screen time', 'HRV variance'],
        alwaysOn: true,
    },
    metabolic: {
        id: 'metabolic',
        name: 'Metabolic',
        color: '#f59e0b',
        description: 'Energy and metabolic function',
        signals: ['Activity', 'Meal timing', 'VO2max'],
        alwaysOn: true,
    },
    recovery: {
        id: 'recovery',
        name: 'Recovery',
        color: '#22c55e',
        description: 'Rest and restoration cycles',
        signals: ['Sleep duration', 'Rest days', 'HRV recovery'],
        alwaysOn: true,
    },
    immune: {
        id: 'immune',
        name: 'Immune',
        color: '#06b6d4',
        description: 'Immune system readiness',
        signals: ['HRV suppression', 'Illness events', 'AQI'],
        alwaysOn: false,
    },
    respiratory: {
        id: 'respiratory',
        name: 'Respiratory',
        color: '#0ea5e9',
        description: 'Lung and breathing health',
        signals: ['SpO2', 'AQI exposure', 'Respiratory rate'],
        alwaysOn: false,
    },
    hormonal: {
        id: 'hormonal',
        name: 'Hormonal',
        color: '#ec4899',
        description: 'Hormonal balance and cycles',
        signals: ['Cycle phase', 'Mood patterns', 'Energy'],
        alwaysOn: false,
    },
    cog_fatigue: {
        id: 'cog_fatigue',
        name: 'Cognitive Fatigue',
        color: '#64748b',
        description: 'Mental exhaustion and burnout',
        signals: ['Screen time', 'Sleep debt', 'Workload'],
        alwaysOn: false,
    },
};

// Mode configuration
export type AgentMode = 'mvp' | 'full';

export function getActiveAgents(mode: AgentMode): AgentConfig[] {
    if (mode === 'mvp') {
        // MVP: Cardio + Mental only
        return [AGENT_DEFINITIONS.cardio, AGENT_DEFINITIONS.mental];
    }
    // Full: All 8 agents
    return Object.values(AGENT_DEFINITIONS);
}

export function getCoreAgents(): AgentConfig[] {
    return Object.values(AGENT_DEFINITIONS).filter(a => a.alwaysOn);
}

export function getSparseAgents(): AgentConfig[] {
    return Object.values(AGENT_DEFINITIONS).filter(a => !a.alwaysOn);
}

// Grid layout helper - returns optimal grid columns for agent count
export function getAgentGridCols(count: number): number {
    if (count <= 2) return 2;
    if (count <= 4) return 2;
    return 4; // 8 agents = 4x2 grid
}
