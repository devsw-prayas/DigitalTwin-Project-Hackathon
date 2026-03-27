// Mock data for Twin dashboard (to be replaced with real API calls)
import { AgentRiskData, TwinSummary, TrajectoryPoint } from './types';

// Generate trajectory data
function generateTrajectory(currentRisk: number, velocity: number, days: number = 180): TrajectoryPoint[] {
    const points: TrajectoryPoint[] = [];
    for (let d = 0; d <= days; d += 7) {
        const base = currentRisk + velocity * d;
        const uncertainty = 0.05 + (d / 180) * 0.15; // Uncertainty grows with time
        points.push({
            day: d,
            p10: Math.max(0, Math.min(1, base - uncertainty)),
            p50: Math.max(0, Math.min(1, base)),
            p90: Math.max(0, Math.min(1, base + uncertainty)),
        });
    }
    return points;
}

// Mock agent data
export const mockAgentRisks: AgentRiskData[] = [
    {
        id: 'cardio',
        name: 'Cardiovascular',
        color: '#ef4444',
        currentRisk: 0.32,
        p10: 0.25,
        p50: 0.35,
        p90: 0.48,
        velocity: 0.002,  // +0.2% per day
        timeToThreshold: undefined,
        trend: 'stable',
        signals: [
            { name: 'HRV', value: 45, unit: 'ms', impact: 'neutral' },
            { name: 'Resting HR', value: 72, unit: 'bpm', impact: 'negative' },
            { name: 'Recovery', value: 85, unit: '%', impact: 'positive' },
        ],
    },
    {
        id: 'mental',
        name: 'Mental Health',
        color: '#8b5cf6',
        currentRisk: 0.28,
        p10: 0.18,
        p50: 0.30,
        p90: 0.42,
        velocity: 0.003,
        timeToThreshold: undefined,
        trend: 'declining',
        signals: [
            { name: 'Sleep Quality', value: 68, unit: '%', impact: 'negative' },
            { name: 'Screen Time', value: 8.5, unit: 'hrs', impact: 'negative' },
            { name: 'HRV Variance', value: 12, unit: '%', impact: 'neutral' },
        ],
    },
    {
        id: 'metabolic',
        name: 'Metabolic',
        color: '#f59e0b',
        currentRisk: 0.45,
        p10: 0.38,
        p50: 0.48,
        p90: 0.62,
        velocity: 0.005,
        timeToThreshold: 120,
        trend: 'declining',
        signals: [
            { name: 'Activity', value: 4200, unit: 'steps', impact: 'negative' },
            { name: 'Deep Sleep', value: 42, unit: 'min', impact: 'neutral' },
            { name: 'Resting HR', value: 78, unit: 'bpm', impact: 'negative' },
        ],
    },
    {
        id: 'recovery',
        name: 'Recovery',
        color: '#22c55e',
        currentRisk: 0.22,
        p10: 0.15,
        p50: 0.24,
        p90: 0.35,
        velocity: -0.001,
        timeToThreshold: undefined,
        trend: 'improving',
        signals: [
            { name: 'Sleep Duration', value: 7.2, unit: 'hrs', impact: 'positive' },
            { name: 'HRV Recovery', value: 92, unit: '%', impact: 'positive' },
            { name: 'Rest Days', value: 2, unit: '/wk', impact: 'neutral' },
        ],
    },
    {
        id: 'immune',
        name: 'Immune',
        color: '#06b6d4',
        currentRisk: 0.35,
        p10: 0.28,
        p50: 0.38,
        p90: 0.52,
        velocity: 0.004,
        timeToThreshold: undefined,
        trend: 'stable',
        signals: [
            { name: 'HRV Suppression', value: 8, unit: '%', impact: 'neutral' },
            { name: 'Recent Illness', value: 0, unit: 'events', impact: 'positive' },
            { name: 'AQI Exposure', value: 35, unit: 'avg', impact: 'neutral' },
        ],
    },
    {
        id: 'respiratory',
        name: 'Respiratory',
        color: '#0ea5e9',
        currentRisk: 0.18,
        p10: 0.12,
        p50: 0.20,
        p90: 0.28,
        velocity: 0.001,
        timeToThreshold: undefined,
        trend: 'stable',
        signals: [
            { name: 'SpO2', value: 98.2, unit: '%', impact: 'positive' },
            { name: 'AQI Exposure', value: 28, unit: 'avg', impact: 'positive' },
            { name: 'Resp Rate', value: 14, unit: '/min', impact: 'positive' },
        ],
    },
    {
        id: 'hormonal',
        name: 'Hormonal',
        color: '#ec4899',
        currentRisk: 0.25,
        p10: 0.18,
        p50: 0.28,
        p90: 0.38,
        velocity: 0.002,
        timeToThreshold: undefined,
        trend: 'stable',
        signals: [
            { name: 'Cycle Phase', value: 14, unit: 'day', impact: 'neutral' },
            { name: 'Mood Pattern', value: 72, unit: '%', impact: 'neutral' },
            { name: 'Energy Level', value: 65, unit: '%', impact: 'neutral' },
        ],
    },
    {
        id: 'cog_fatigue',
        name: 'Cognitive Fatigue',
        color: '#64748b',
        currentRisk: 0.42,
        p10: 0.32,
        p50: 0.45,
        p90: 0.58,
        velocity: 0.006,
        timeToThreshold: 85,
        trend: 'declining',
        signals: [
            { name: 'Screen Time', value: 9.2, unit: 'hrs', impact: 'negative' },
            { name: 'Sleep Debt', value: 4.5, unit: 'hrs', impact: 'negative' },
            { name: 'Workload', value: 85, unit: '%', impact: 'negative' },
        ],
    },
];

export const mockTwinSummary: TwinSummary = {
    overallRisk: 0.31,
    riskLevel: 'moderate',
    activeAgents: 8,
    lastSync: new Date(Date.now() - 1000 * 60 * 15), // 15 minutes ago
    dataQuality: 87,
};

export function getAgentTrajectory(agentId: string): TrajectoryPoint[] {
    const agent = mockAgentRisks.find(a => a.id === agentId);
    if (!agent) return [];
    return generateTrajectory(agent.currentRisk, agent.velocity);
}
