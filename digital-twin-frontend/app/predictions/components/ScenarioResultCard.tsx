'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, ArrowRight } from 'lucide-react';
import { ScenarioResult } from './types';

interface ScenarioResultCardProps {
    result: ScenarioResult;
    index: number;
}

export function ScenarioResultCard({ result, index }: ScenarioResultCardProps) {
    const isImprovement = result.absoluteChange < 0;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.03 }}
            className="p-3 rounded-xl bg-white/5 backdrop-blur-md border border-white/10"
        >
            <div className="flex items-center gap-2 mb-2">
                <div
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ background: result.agentColor }}
                />
                <span className="text-sm font-medium text-white/80">{result.agentName}</span>
            </div>

            <div className="flex items-center gap-2 mb-2">
                {/* Baseline */}
                <div className="text-center">
                    <div className="text-xs text-white/40 mb-0.5">Current</div>
                    <div className="text-lg font-bold text-white/60">
                        {(result.baselineRisk * 100).toFixed(0)}%
                    </div>
                </div>

                <ArrowRight className="w-4 h-4 text-white/30" />

                {/* Projected */}
                <div className="text-center">
                    <div className="text-xs text-white/40 mb-0.5">Projected</div>
                    <div
                        className="text-lg font-bold"
                        style={{ color: isImprovement ? '#22c55e' : '#ef4444' }}
                    >
                        {(result.projectedRisk * 100).toFixed(0)}%
                    </div>
                </div>

                {/* Change indicator */}
                <div className="ml-auto flex items-center gap-1">
                    {isImprovement ? (
                        <TrendingDown className="w-4 h-4 text-green-400" />
                    ) : (
                        <TrendingUp className="w-4 h-4 text-red-400" />
                    )}
                    <span
                        className="text-sm font-medium"
                        style={{ color: isImprovement ? '#22c55e' : '#ef4444' }}
                    >
                        {isImprovement ? '' : '+'}{result.percentChange.toFixed(1)}%
                    </span>
                </div>
            </div>

            {/* Progress bar */}
            <div className="relative h-1.5 bg-gray-800 rounded-full overflow-hidden">
                {/* Baseline marker */}
                <div
                    className="absolute top-0 w-0.5 h-full bg-white/40"
                    style={{ left: `${result.baselineRisk * 100}%` }}
                />
                {/* Projected value */}
                <div
                    className="absolute h-full rounded-full transition-all"
                    style={{
                        width: `${result.projectedRisk * 100}%`,
                        background: isImprovement ? '#22c55e' : '#ef4444',
                    }}
                />
            </div>

            {/* Confidence */}
            <div className="text-xs text-white/30 mt-1">
                Confidence: {(result.confidence * 100).toFixed(0)}% • {result.timeHorizon}d horizon
            </div>
        </motion.div>
    );
}

export default ScenarioResultCard;
