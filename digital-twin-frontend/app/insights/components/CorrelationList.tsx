'use client';

import { motion } from 'framer-motion';
import { CorrelationData } from './types';

interface CorrelationListProps {
    correlations: CorrelationData[];
}

function getStrengthColor(strength: CorrelationData['strength']): string {
    switch (strength) {
        case 'strong':
            return '#22c55e';
        case 'moderate':
            return '#eab308';
        case 'weak':
            return '#6b7280';
        default:
            return '#374151';
    }
}

function getCorrelationBarColor(correlation: number): string {
    if (correlation > 0.6) return '#22c55e';
    if (correlation > 0.3) return '#eab308';
    if (correlation > 0) return '#6b7280';
    if (correlation > -0.3) return '#6b7280';
    if (correlation > -0.6) return '#f97316';
    return '#ef4444';
}

export function CorrelationList({ correlations }: CorrelationListProps) {
    return (
        <div className="space-y-3">
            {correlations.map((corr, index) => (
                <motion.div
                    key={`${corr.signal1}-${corr.signal2}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="p-3 rounded-xl bg-white/5 backdrop-blur-md border border-white/10"
                >
                    <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                            <span className="text-sm text-white/70">{corr.signal1}</span>
                            <span className="text-xs text-white/30">↔</span>
                            <span className="text-sm text-white/70">{corr.signal2}</span>
                        </div>
                        <span
                            className="text-xs font-medium capitalize px-2 py-0.5 rounded-full"
                            style={{
                                background: `${getStrengthColor(corr.strength)}20`,
                                color: getStrengthColor(corr.strength),
                            }}
                        >
                            {corr.strength}
                        </span>
                    </div>
                    {/* Correlation bar */}
                    <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div
                            className="absolute h-full rounded-full transition-all"
                            style={{
                                width: `${Math.abs(corr.correlation) * 100}%`,
                                background: getCorrelationBarColor(corr.correlation),
                            }}
                        />
                    </div>
                    <div className="flex justify-between mt-1">
                        <span className="text-xs text-white/40">r = {corr.correlation.toFixed(2)}</span>
                        <span className="text-xs text-white/30">
                            {corr.correlation > 0 ? 'positive' : 'negative'} correlation
                        </span>
                    </div>
                </motion.div>
            ))}
        </div>
    );
}

export default CorrelationList;
