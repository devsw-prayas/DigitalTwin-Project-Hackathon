// Mock data for Predictions page
import { WhatIfScenario, ScenarioResult, PredictionTrajectory } from './types';

export const mockScenarios: WhatIfScenario[] = [
    {
        id: 'steps_increase',
        name: 'Daily Steps',
        description: 'Increase daily step count',
        icon: '🚶',
        category: 'activity',
        defaultValue: 4200,
        minValue: 2000,
        maxValue: 15000,
        unit: 'steps',
        step: 500,
    },
    {
        id: 'sleep_duration',
        name: 'Sleep Duration',
        description: 'Average sleep per night',
        icon: '😴',
        category: 'sleep',
        defaultValue: 7.2,
        minValue: 5,
        maxValue: 9,
        unit: 'hrs',
        step: 0.5,
    },
    {
        id: 'screen_time',
        name: 'Screen Time',
        description: 'Daily screen exposure',
        icon: '📱',
        category: 'behavior',
        defaultValue: 9.2,
        minValue: 2,
        maxValue: 12,
        unit: 'hrs',
        step: 0.5,
    },
    {
        id: 'active_minutes',
        name: 'Active Minutes',
        description: 'Moderate+ vigorous activity',
        icon: '🏃',
        category: 'activity',
        defaultValue: 28,
        minValue: 10,
        maxValue: 90,
        unit: 'min',
        step: 5,
    },
    {
        id: 'rest_days',
        name: 'Rest Days',
        description: 'Recovery days per week',
        icon: '🧘',
        category: 'recovery',
        defaultValue: 1,
        minValue: 0,
        maxValue: 3,
        unit: '/week',
        step: 1,
    },
];

// Simulate scenario impact on agents
export function simulateScenario(
    scenarioId: string,
    value: number,
    baseline: number
): ScenarioResult[] {
    const results: ScenarioResult[] = [];

    // Simplified impact model
    const impacts: Record<string, Record<string, { base: number; factor: number }>> = {
        steps_increase: {
            cardio: { base: -0.02, factor: 0.00003 },
            metabolic: { base: -0.03, factor: 0.00005 },
            mental: { base: -0.01, factor: 0.00001 },
            recovery: { base: -0.01, factor: 0.00002 },
        },
        sleep_duration: {
            mental: { base: -0.05, factor: 0.02 },
            recovery: { base: -0.04, factor: 0.015 },
            cog_fatigue: { base: -0.06, factor: 0.025 },
            cardio: { base: -0.02, factor: 0.008 },
        },
        screen_time: {
            mental: { base: 0.03, factor: -0.005 },
            cog_fatigue: { base: 0.04, factor: -0.008 },
            recovery: { base: 0.01, factor: -0.002 },
        },
        active_minutes: {
            cardio: { base: -0.03, factor: 0.001 },
            metabolic: { base: -0.04, factor: 0.0015 },
            recovery: { base: -0.02, factor: 0.0005 },
        },
        rest_days: {
            recovery: { base: -0.05, factor: 0.02 },
            cardio: { base: -0.02, factor: 0.008 },
            immune: { base: -0.01, factor: 0.005 },
        },
    };

    const agentColors: Record<string, string> = {
        cardio: '#ef4444',
        mental: '#8b5cf6',
        metabolic: '#f59e0b',
        recovery: '#22c55e',
        immune: '#06b6d4',
        respiratory: '#0ea5e9',
        hormonal: '#ec4899',
        cog_fatigue: '#64748b',
    };

    const agentBaselines: Record<string, number> = {
        cardio: 0.32,
        mental: 0.28,
        metabolic: 0.45,
        recovery: 0.22,
        immune: 0.35,
        respiratory: 0.18,
        hormonal: 0.25,
        cog_fatigue: 0.42,
    };

    const scenarioImpacts = impacts[scenarioId] || {};
    const scenario = mockScenarios.find(s => s.id === scenarioId);
    if (!scenario) return results;

    // Calculate deviation from default
    const deviation = value - scenario.defaultValue;
    const normalizedDeviation = deviation / (scenario.maxValue - scenario.minValue);

    Object.entries(agentBaselines).forEach(([agentId, agentBaseline]) => {
        const impact = scenarioImpacts[agentId];
        let projectedRisk = agentBaseline;

        if (impact) {
            const change = impact.base + impact.factor * deviation;
            projectedRisk = Math.max(0, Math.min(1, agentBaseline + change));
        }

        results.push({
            scenarioId,
            agentId,
            agentName: agentId.charAt(0).toUpperCase() + agentId.slice(1).replace('_', ' '),
            agentColor: agentColors[agentId] || '#888888',
            baselineRisk: agentBaseline,
            projectedRisk,
            absoluteChange: projectedRisk - agentBaseline,
            percentChange: ((projectedRisk - agentBaseline) / agentBaseline) * 100,
            confidence: 0.75 + Math.random() * 0.2,
            timeHorizon: 30,
        });
    });

    return results;
}

// Generate trajectory data
export function generateTrajectory(
    baseline: number,
    projected: number,
    days: number = 90
): PredictionTrajectory[] {
    const points: PredictionTrajectory[] = [];
    const uncertaintyGrowth = 0.001;

    for (let d = 0; d <= days; d += 7) {
        const progress = d / days;
        const currentBaseline = baseline + (projected - baseline) * progress;
        const uncertainty = uncertaintyGrowth * d;

        points.push({
            day: d,
            baseline,
            optimistic: Math.max(0, currentBaseline - uncertainty * 50),
            pessimistic: Math.min(1, currentBaseline + uncertainty * 50),
            scenario: currentBaseline,
        });
    }

    return points;
}
