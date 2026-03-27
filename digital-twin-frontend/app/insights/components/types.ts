// Types for Insights page

export interface HealthInsight {
    id: string;
    type: 'warning' | 'info' | 'success' | 'action';
    title: string;
    description: string;
    agent?: string;
    agentColor?: string;
    timestamp: Date;
    priority: 'high' | 'medium' | 'low';
    actionable?: boolean;
    action?: {
        label: string;
        href: string;
    };
}

export interface TrendData {
    signal: string;
    current: number;
    previous: number;
    change: number;
    unit: string;
    trend: 'up' | 'down' | 'stable';
    period: string;
}

export interface CorrelationData {
    signal1: string;
    signal2: string;
    correlation: number;  // -1 to 1
    strength: 'strong' | 'moderate' | 'weak' | 'none';
}

export interface WeeklySummary {
    weekStart: Date;
    overallScore: number;
    topImprovements: string[];
    topDeclines: string[];
    keyInsights: string[];
}
