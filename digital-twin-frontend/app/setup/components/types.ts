// Shared types for setup wizard

export interface SetupData {
    age: string;
    sex: string;
    conditions: string[];
    goals: string[];
    dataSource: 'apple' | 'google' | 'manual' | null;
}

export const CONDITIONS = [
    { id: 'diabetes', label: 'Diabetes' },
    { id: 'hypertension', label: 'Hypertension' },
    { id: 'heart_disease', label: 'Heart Disease' },
    { id: 'asthma', label: 'Asthma' },
    { id: 'anxiety', label: 'Anxiety/Depression' },
    { id: 'insomnia', label: 'Chronic Insomnia' },
    { id: 'obesity', label: 'Obesity' },
    { id: 'thyroid', label: 'Thyroid Disorder' },
];

export const GOALS = [
    { id: 'energy', label: 'More Energy', icon: '⚡' },
    { id: 'sleep', label: 'Better Sleep', icon: '😴' },
    { id: 'fitness', label: 'Improve Fitness', icon: '💪' },
    { id: 'stress', label: 'Reduce Stress', icon: '🧘' },
    { id: 'longevity', label: 'Longevity', icon: '🧬' },
    { id: 'weight', label: 'Weight Management', icon: '⚖️' },
];

export const DATA_SOURCES = [
    {
        id: 'apple' as const,
        label: 'Apple Health',
        logo: '🍎',
        description: 'Sync from iPhone & Apple Watch',
        platform: 'iOS',
        color: '#000000',
    },
    {
        id: 'google' as const,
        label: 'Google Health Connect',
        logo: '🏥',
        description: 'Sync from Android devices',
        platform: 'Android',
        color: '#34A853',
    },
    {
        id: 'manual' as const,
        label: 'Manual Entry',
        logo: '✏️',
        description: 'Log your data manually',
        platform: 'Any',
        color: '#38bdf8',
    },
];
