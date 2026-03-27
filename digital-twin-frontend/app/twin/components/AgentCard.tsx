'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, AlertTriangle, Clock } from 'lucide-react';
import { AgentRiskData, TrajectoryPoint } from './types';

interface AgentCardProps {
    agent: AgentRiskData;
    trajectory?: TrajectoryPoint[];
    isExpanded?: boolean;
    onClick?: () => void;
}

function getTrendIcon(trend: AgentRiskData['trend']) {
    switch (trend) {
        case 'improving':
            return <TrendingDown className="w-4 h-4" />;
        case 'declining':
            return <TrendingUp className="w-4 h-4" />;
        default:
            return <Minus className="w-4 h-4" />;
    }
}

function getTrendColor(trend: AgentRiskData['trend']) {
    switch (trend) {
        case 'improving':
            return 'text-green-400';
        case 'declining':
            return 'text-red-400';
        default:
            return 'text-gray-400';
    }
}

function getRiskLevel(risk: number): { label: string; color: string } {
    if (risk < 0.25) return { label: 'Low', color: 'text-green-400' };
    if (risk < 0.50) return { label: 'Moderate', color: 'text-yellow-400' };
    if (risk < 0.75) return { label: 'Elevated', color: 'text-orange-400' };
    return { label: 'High', color: 'text-red-400' };
}

// Mini sparkline chart
function MiniChart({ trajectory, color }: { trajectory: TrajectoryPoint[]; color: string }) {
    if (!trajectory.length) return null;

    const width = 80;
    const height = 32;
    const padding = 2;

    const xScale = (width - padding * 2) / (trajectory.length - 1 || 1);
    const yScale = (height - padding * 2);

    const points = trajectory.map((p, i) => ({
        x: padding + i * xScale,
        y: padding + (1 - p.p50) * yScale,
    }));

    const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

    // Uncertainty band
    const bandPath = trajectory.map((p, i) => {
        const x = padding + i * xScale;
        return { x, yHigh: padding + (1 - p.p90) * yScale, yLow: padding + (1 - p.p10) * yScale };
    });

    const bandD = `M ${bandPath[0].x} ${bandPath[0].yHigh} ` +
        bandPath.map(p => `L ${p.x} ${p.yHigh}`).join(' ') +
        ` L ${bandPath[bandPath.length - 1].x} ${bandPath[bandPath.length - 1].yLow} ` +
        bandPath.reverse().map(p => `L ${p.x} ${p.yLow}`).join(' ') + ' Z';

    return (
        <svg width={width} height={height} className="opacity-60">
            {/* Uncertainty band */}
            <path d={bandD} fill={color} opacity={0.2} />
            {/* Main line */}
            <path d={pathD} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" />
        </svg>
    );
}

export function AgentCard({ agent, trajectory, isExpanded, onClick }: AgentCardProps) {
    const riskLevel = getRiskLevel(agent.currentRisk);

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onClick}
            className={`
                relative p-4 rounded-2xl cursor-pointer
                bg-white/5 backdrop-blur-md border border-white/10
                hover:border-white/20 hover:bg-white/8
                transition-all duration-300
                ${isExpanded ? 'ring-2 ring-offset-2 ring-offset-transparent' : ''}
            `}
            style={{ borderColor: isExpanded ? agent.color : undefined }}
        >
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                    <div
                        className="w-3 h-3 rounded-full"
                        style={{ background: agent.color, boxShadow: `0 0 12px ${agent.color}60` }}
                    />
                    <h3 className="font-semibold text-white/90">{agent.name}</h3>
                </div>
                <div className={`flex items-center gap-1 ${getTrendColor(agent.trend)}`}>
                    {getTrendIcon(agent.trend)}
                    <span className="text-xs capitalize">{agent.trend}</span>
                </div>
            </div>

            {/* Main Risk Score */}
            <div className="flex items-end justify-between mb-3">
                <div>
                    <div className="text-3xl font-bold" style={{ color: agent.color }}>
                        {(agent.currentRisk * 100).toFixed(0)}%
                    </div>
                    <div className={`text-xs ${riskLevel.color}`}>{riskLevel.label} Risk</div>
                </div>
                {trajectory && <MiniChart trajectory={trajectory} color={agent.color} />}
            </div>

            {/* Forecast Range */}
            <div className="flex items-center gap-2 text-xs text-white/40 mb-3">
                <span>30d forecast:</span>
                <span className="text-white/60">
                    {(agent.p10 * 100).toFixed(0)}% - {(agent.p90 * 100).toFixed(0)}%
                </span>
            </div>

            {/* Time to Threshold Warning */}
            {agent.timeToThreshold && (
                <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-orange-500/10 border border-orange-500/20 mb-3">
                    <AlertTriangle className="w-3.5 h-3.5 text-orange-400" />
                    <span className="text-xs text-orange-300">
                        Threshold in ~{agent.timeToThreshold} days
                    </span>
                </div>
            )}

            {/* Signals (expanded view) */}
            {isExpanded && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-3 pt-3 border-t border-white/10"
                >
                    <h4 className="text-xs uppercase tracking-wider text-white/40 mb-2">Key Signals</h4>
                    <div className="space-y-1.5">
                        {agent.signals.map((signal) => (
                            <div key={signal.name} className="flex items-center justify-between text-sm">
                                <span className="text-white/60">{signal.name}</span>
                                <div className="flex items-center gap-2">
                                    <span className="text-white/80">
                                        {signal.value}{signal.unit ? ` ${signal.unit}` : ''}
                                    </span>
                                    <div className={`w-2 h-2 rounded-full ${
                                        signal.impact === 'positive' ? 'bg-green-400' :
                                        signal.impact === 'negative' ? 'bg-red-400' : 'bg-gray-400'
                                    }`} />
                                </div>
                            </div>
                        ))}
                    </div>
                </motion.div>
            )}
        </motion.div>
    );
}

export default AgentCard;
