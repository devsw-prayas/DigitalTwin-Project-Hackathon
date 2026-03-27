// Types for Predictions page

export interface WhatIfScenario {
    id: string;
    name: string;
    description: string;
    icon: string;
    category: 'activity' | 'sleep' | 'stress' | 'environment' | 'behavior' | 'recovery';
    defaultValue: number;
    minValue: number;
    maxValue: number;
    unit: string;
    step: number;
}

export interface ScenarioResult {
    scenarioId: string;
    agentId: string;
    agentName: string;
    agentColor: string;
    baselineRisk: number;      // Current risk
    projectedRisk: number;     // After scenario
    absoluteChange: number;    // Absolute change
    percentChange: number;     // Percent change
    confidence: number;        // 0-1
    timeHorizon: number;       // Days
}

export interface PredictionTrajectory {
    day: number;
    baseline: number;
    optimistic: number;
    pessimistic: number;
    scenario?: number;
}

export type TimeHorizon = 7 | 30 | 90 | 180;
