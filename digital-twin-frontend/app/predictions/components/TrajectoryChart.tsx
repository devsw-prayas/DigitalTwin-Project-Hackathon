'use client';

import { motion } from 'framer-motion';
import { PredictionTrajectory } from './types';

interface TrajectoryChartProps {
    trajectories: PredictionTrajectory[];
    agentColor: string;
    agentName: string;
}

export function TrajectoryChart({ trajectories, agentColor, agentName }: TrajectoryChartProps) {
    const width = 280;
    const height = 120;
    const padding = 20;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    if (!trajectories.length) return null;

    const maxDay = Math.max(...trajectories.map(t => t.day));
    const xScale = chartWidth / maxDay;
    const yScale = chartHeight;

    // Generate path for scenario line
    const scenarioPath = trajectories
        .map((t, i) => {
            const x = padding + t.day * xScale;
            const y = padding + (1 - t.scenario!) * yScale;
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        })
        .join(' ');

    // Generate uncertainty band
    const bandTop = trajectories
        .map((t, i) => {
            const x = padding + t.day * xScale;
            const y = padding + (1 - t.optimistic) * yScale;
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        })
        .join(' ');

    const bandBottom = trajectories
        .reverse()
        .map((t, i) => {
            const x = padding + t.day * xScale;
            const y = padding + (1 - t.pessimistic) * yScale;
            return `L ${x} ${y}`;
        })
        .join(' ');

    const bandPath = `${bandTop} ${bandBottom} Z`;

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="p-4 rounded-xl bg-white/5 backdrop-blur-md border border-white/10"
        >
            <div className="flex items-center gap-2 mb-2">
                <div
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ background: agentColor }}
                />
                <span className="text-sm font-medium text-white/80">{agentName}</span>
            </div>

            <svg width={width} height={height} className="overflow-visible">
                {/* Uncertainty band */}
                <path
                    d={bandPath}
                    fill={agentColor}
                    opacity={0.15}
                />

                {/* Scenario line */}
                <path
                    d={scenarioPath}
                    fill="none"
                    stroke={agentColor}
                    strokeWidth={2}
                    strokeLinecap="round"
                />

                {/* Baseline reference line */}
                <line
                    x1={padding}
                    y1={padding + (1 - trajectories[0].baseline) * yScale}
                    x2={width - padding}
                    y2={padding + (1 - trajectories[0].baseline) * yScale}
                    stroke="white"
                    strokeOpacity={0.2}
                    strokeDasharray="4,4"
                />

                {/* X-axis labels */}
                <text x={padding} y={height - 4} className="text-[10px] fill-white/30">
                    0d
                </text>
                <text x={width - padding - 15} y={height - 4} className="text-[10px] fill-white/30">
                    {maxDay}d
                </text>
            </svg>

            {/* Legend */}
            <div className="flex items-center gap-4 text-xs text-white/40 mt-2">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-0.5 bg-white/30" style={{ background: agentColor }} />
                    <span>Projected</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-0.5 border-t border-dashed border-white/30" />
                    <span>Baseline</span>
                </div>
            </div>
        </motion.div>
    );
}

export default TrajectoryChart;
