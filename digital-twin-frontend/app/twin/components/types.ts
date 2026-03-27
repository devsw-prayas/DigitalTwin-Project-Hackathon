// Types for Twin dashboard

export interface AgentRiskData {
    id: string;
    name: string;
    color: string;
    currentRisk: number;        // 0-1 scale
    p10: number;                // 10th percentile forecast
    p50: number;                // 50th percentile (median)
    p90: number;                // 90th percentile
    velocity: number;           // Rate of change per week
    timeToThreshold?: number;   // Days until crossing threshold (if applicable)
    trend: 'improving' | 'stable' | 'declining';
    signals: {
        name: string;
        value: number;
        unit: string;
        impact: 'positive' | 'neutral' | 'negative';
    }[];
}

export interface TwinSummary {
    overallRisk: number;
    riskLevel: 'low' | 'moderate' | 'elevated' | 'high';
    activeAgents: number;
    lastSync: Date;
    dataQuality: number;        // 0-100%
}

export type TimeHorizon = 7 | 30 | 90 | 180;

export interface TrajectoryPoint {
    day: number;
    p10: number;
    p50: number;
    p90: number;
}
