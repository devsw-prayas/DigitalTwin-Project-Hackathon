// Mock data for Insights page
import { HealthInsight, TrendData, CorrelationData, WeeklySummary } from './types';

export const mockInsights: HealthInsight[] = [
    {
        id: '1',
        type: 'warning',
        title: 'Metabolic risk trending upward',
        description: 'Your metabolic risk has increased 8% over the past 2 weeks. This is primarily driven by reduced activity levels and elevated resting heart rate.',
        agent: 'metabolic',
        agentColor: '#f59e0b',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
        priority: 'high',
        actionable: true,
        action: { label: 'View Recommendations', href: '/predictions' },
    },
    {
        id: '2',
        type: 'success',
        title: 'Recovery improving steadily',
        description: 'Your recovery score has improved 12% this week. Keep maintaining consistent sleep schedules.',
        agent: 'recovery',
        agentColor: '#22c55e',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 8), // 8 hours ago
        priority: 'medium',
    },
    {
        id: '3',
        type: 'warning',
        title: 'Cognitive fatigue threshold approaching',
        description: 'Based on current trends, your cognitive fatigue may cross the elevated threshold in approximately 85 days if patterns continue.',
        agent: 'cog_fatigue',
        agentColor: '#64748b',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24), // 1 day ago
        priority: 'high',
        actionable: true,
        action: { label: 'Run Simulation', href: '/predictions' },
    },
    {
        id: '4',
        type: 'info',
        title: 'HRV-Sleep correlation detected',
        description: 'We noticed a strong correlation between your HRV and sleep quality. Days with higher HRV tend to follow better sleep nights.',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 36), // 36 hours ago
        priority: 'low',
    },
    {
        id: '5',
        type: 'action',
        title: 'Consider more rest days',
        description: 'Your activity pattern shows 6 consecutive high-activity days. Adding a rest day could improve recovery metrics.',
        agent: 'recovery',
        agentColor: '#22c55e',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 4), // 4 hours ago
        priority: 'medium',
        actionable: true,
        action: { label: 'Adjust Schedule', href: '/predictions' },
    },
    {
        id: '6',
        type: 'success',
        title: 'Respiratory health stable',
        description: 'SpO2 levels and respiratory rate remain in optimal range despite recent AQI fluctuations.',
        agent: 'respiratory',
        agentColor: '#0ea5e9',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 12), // 12 hours ago
        priority: 'low',
    },
];

export const mockTrends: TrendData[] = [
    { signal: 'HRV', current: 45, previous: 42, change: 7.1, unit: 'ms', trend: 'up', period: '7 days' },
    { signal: 'Resting HR', current: 72, previous: 70, change: 2.9, unit: 'bpm', trend: 'up', period: '7 days' },
    { signal: 'Sleep Duration', current: 7.2, previous: 6.8, change: 5.9, unit: 'hrs', trend: 'up', period: '7 days' },
    { signal: 'Steps', current: 4200, previous: 5500, change: -23.6, unit: 'avg/day', trend: 'down', period: '7 days' },
    { signal: 'SpO2', current: 98.2, previous: 98.0, change: 0.2, unit: '%', trend: 'stable', period: '7 days' },
    { signal: 'Deep Sleep', current: 42, previous: 48, change: -12.5, unit: 'min', trend: 'down', period: '7 days' },
    { signal: 'Screen Time', current: 9.2, previous: 8.1, change: 13.6, unit: 'hrs/day', trend: 'up', period: '7 days' },
    { signal: 'Active Minutes', current: 28, previous: 35, change: -20.0, unit: 'min', trend: 'down', period: '7 days' },
];

export const mockCorrelations: CorrelationData[] = [
    { signal1: 'HRV', signal2: 'Sleep Quality', correlation: 0.72, strength: 'strong' },
    { signal1: 'Steps', signal2: 'Resting HR', correlation: -0.58, strength: 'moderate' },
    { signal1: 'Screen Time', signal2: 'Sleep Duration', correlation: -0.65, strength: 'moderate' },
    { signal1: 'Deep Sleep', signal2: 'Recovery', correlation: 0.81, strength: 'strong' },
    { signal1: 'Resting HR', signal2: 'Metabolic Risk', correlation: 0.45, strength: 'moderate' },
    { signal1: 'HRV', signal2: 'Cognitive Fatigue', correlation: -0.52, strength: 'moderate' },
];

export const mockWeeklySummary: WeeklySummary = {
    weekStart: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7),
    overallScore: 72,
    topImprovements: ['Recovery +12%', 'Sleep Duration +5.9%', 'HRV +7.1%'],
    topDeclines: ['Steps -23.6%', 'Deep Sleep -12.5%', 'Active Minutes -20%'],
    keyInsights: [
        'Activity levels dropped significantly this week',
        'Sleep quality metrics are improving',
        'Recovery patterns are stabilizing',
    ],
};
