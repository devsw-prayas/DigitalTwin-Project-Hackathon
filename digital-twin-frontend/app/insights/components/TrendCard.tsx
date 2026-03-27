'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { TrendData } from './types';

interface TrendCardProps {
    trend: TrendData;
    index: number;
}

function getTrendIcon(trend: TrendData['trend']) {
    switch (trend) {
        case 'up':
            return <TrendingUp className="w-3.5 h-3.5" />;
        case 'down':
            return <TrendingDown className="w-3.5 h-3.5" />;
        default:
            return <Minus className="w-3.5 h-3.5" />;
    }
}

function getTrendColor(trend: TrendData['trend'], isNegativeGood: boolean) {
    if (trend === 'stable') return 'text-gray-400';

    // Some signals are better when going down (like resting HR, screen time)
    const isGood = isNegativeGood ? trend === 'down' : trend === 'up';
    return isGood ? 'text-green-400' : 'text-red-400';
}

// Signals where lower is better
const NEGATIVE_GOOD_SIGNALS = ['Resting HR', 'Screen Time', 'Metabolic Risk'];

export function TrendCard({ trend, index }: TrendCardProps) {
    const isNegativeGood = NEGATIVE_GOOD_SIGNALS.some(s => trend.signal.includes(s));
    const trendColor = getTrendColor(trend.trend, isNegativeGood);
    const changeColor = trend.trend === 'stable' ? 'text-gray-400' : trendColor;

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.03 }}
            className="p-3 rounded-xl bg-white/5 backdrop-blur-md border border-white/10"
        >
            <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-white/70">{trend.signal}</span>
                <div className={`flex items-center gap-1 ${trendColor}`}>
                    {getTrendIcon(trend.trend)}
                </div>
            </div>
            <div className="flex items-end justify-between">
                <div>
                    <span className="text-xl font-bold text-white/90">
                        {trend.current}
                    </span>
                    <span className="text-xs text-white/40 ml-1">{trend.unit}</span>
                </div>
                <div className={`text-sm font-medium ${changeColor}`}>
                    {trend.change > 0 ? '+' : ''}{trend.change.toFixed(1)}%
                </div>
            </div>
            <div className="text-xs text-white/30 mt-1">
                vs {trend.period} ago
            </div>
        </motion.div>
    );
}

export default TrendCard;
