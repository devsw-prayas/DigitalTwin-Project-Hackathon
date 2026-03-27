'use client';

import {useEffect, useMemo, useState} from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Activity, Shield, Zap, Target, AlertTriangle,
    TrendingUp, TrendingDown, Calendar, ChevronDown,
} from 'lucide-react';
import { useAgentMode } from '@/app/agents/agent-mode-context';
import { mockAgentRisks, mockTwinSummary, getAgentTrajectory } from './components/mockData';
import {useSubsystem} from "@/app/system/theme/subsystem-context";

/* ─── tokens ─────────────────────────────────────────────────────────────── */
const textPri   = 'var(--textPrimary)';
const textMuted = 'var(--textMuted)';
const accent    = 'var(--accent)';
const border    = 'color-mix(in srgb, var(--textMuted) 15%, transparent)';

/* ─── glass styles ───────────────────────────────────────────────────────── */
const glass = {
    background: 'color-mix(in srgb, var(--surface) 80%, transparent)',
    backdropFilter: 'blur(16px) saturate(150%)',
    WebkitBackdropFilter: 'blur(16px) saturate(150%)',
    boxShadow: '0 4px 24px rgba(0,0,0,0.12), 0 1px 4px rgba(0,0,0,0.08)',
};

const glassSunken = {
    background: 'color-mix(in srgb, var(--surface) 50%, transparent)',
    backdropFilter: 'blur(8px)',
    WebkitBackdropFilter: 'blur(8px)',
    border: `1px solid ${border}`,
};


/* ─── MiniChart ──────────────────────────────────────────────────────────── */
function MiniChart({ trajectory, color }: { trajectory: any[]; color: string }) {
    if (!trajectory?.length) return null;
    const W = 100, H = 36, P = 3;
    const cw = W - P * 2, ch = H - P * 2;
    const xs = (i: number) => P + (i / (trajectory.length - 1 || 1)) * cw;
    const ys = (v: number) => P + (1 - v) * ch;
    const path = trajectory
        .map((t: any, i: number) => `${i ? 'L' : 'M'}${xs(i).toFixed(1)},${ys(t.p50).toFixed(1)}`)
        .join(' ');
    return (
        <svg width={W} height={H} style={{ display: 'block', opacity: 0.65 }}>
            <path d={path} fill="none" stroke={color} strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    );
}

/* ─── AgentCard ──────────────────────────────────────────────────────────── */
function AgentCard({ agent, trajectory, index }: { agent: any; trajectory: any[]; index: number }) {
    const [expanded, setExpanded] = useState(false);

    const riskLevel = agent.currentRisk < 0.25 ? 'Low' : agent.currentRisk < 0.5 ? 'Moderate' : agent.currentRisk < 0.75 ? 'Elevated' : 'High';
    const riskColor = agent.currentRisk < 0.25 ? '#16a34a' : agent.currentRisk < 0.5 ? '#ca8a04' : agent.currentRisk < 0.75 ? '#ea580c' : '#dc2626';

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05, duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        >
            <div
                onClick={() => setExpanded(!expanded)}
                className="relative p-4 rounded-2xl cursor-pointer transition-all duration-200 overflow-hidden"
                style={{
                    ...glass,
                    ...(expanded && {
                        borderColor: `color-mix(in srgb, var(--accent) 30%, transparent)`,
                        boxShadow: `0 4px 24px rgba(0,0,0,0.14), 0 0 0 1px color-mix(in srgb, var(--accent) 20%, transparent)`,
                    }),
                }}
            >
                {/* subtle colour wash */}
                <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                        background: `radial-gradient(ellipse at top left, ${agent.color}0f 0%, transparent 60%)`,
                    }}
                />

                {/* Header */}
                <div className="relative flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2.5">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ background: agent.color }} />
                        <div>
                            <div className="text-sm font-semibold leading-none" style={{ color: textPri }}>
                                {agent.name}
                            </div>
                            <div className="text-[11px] mt-0.5 capitalize" style={{ color: textMuted }}>
                                {agent.trend} trend
                            </div>
                        </div>
                    </div>

                    <div
                        className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium"
                        style={{
                            background: agent.trend === 'improving' ? 'rgba(22,163,74,0.10)' :
                                agent.trend === 'declining' ? 'rgba(220,38,38,0.10)' :
                                    'color-mix(in srgb, var(--textMuted) 10%, transparent)',
                            color: agent.trend === 'improving' ? '#16a34a' :
                                agent.trend === 'declining' ? '#dc2626' : textMuted,
                        }}
                    >
                        {agent.trend === 'improving' ? <TrendingDown className="w-3 h-3" /> :
                            agent.trend === 'declining' ? <TrendingUp className="w-3 h-3" /> :
                                <Activity className="w-3 h-3" />}
                    </div>
                </div>

                {/* Score + Chart */}
                <div className="relative flex items-end justify-between mb-3">
                    <div>
                        <div className="text-4xl font-black tracking-tight leading-none" style={{ color: agent.color }}>
                            {(agent.currentRisk * 100).toFixed(0)}<span className="text-lg font-semibold">%</span>
                        </div>
                        <div
                            className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium mt-1"
                            style={{ background: `color-mix(in srgb, ${riskColor} 12%, transparent)`, color: riskColor }}
                        >
                            {riskLevel} Risk
                        </div>
                    </div>
                    {trajectory && <MiniChart trajectory={trajectory} color={agent.color} />}
                </div>

                {/* Forecast Row */}
                <div
                    className="flex items-center justify-between px-3 py-2 rounded-xl mb-2"
                    style={glassSunken}
                >
                    <div className="flex items-center gap-1.5 text-[11px]" style={{ color: textMuted }}>
                        <Calendar className="w-3 h-3" />
                        30d forecast
                    </div>
                    <span className="text-xs font-semibold tabular-nums" style={{ color: textPri, opacity: 0.7 }}>
                        {(agent.p10 * 100).toFixed(0)}% – {(agent.p90 * 100).toFixed(0)}%
                    </span>
                </div>

                {/* Threshold Warning */}
                {agent.timeToThreshold && (
                    <div
                        className="flex items-center gap-2 px-3 py-2 rounded-xl border"
                        style={{ background: 'rgba(234,88,12,0.07)', borderColor: 'rgba(234,88,12,0.20)' }}
                    >
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0" style={{ color: '#ea580c' }} />
                        <span className="text-[11px] font-medium" style={{ color: '#ea580c' }}>
                            Threshold in ~{agent.timeToThreshold} days
                        </span>
                    </div>
                )}

                {/* Expandable Signals */}
                <AnimatePresence>
                    {expanded && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="mt-3 pt-3 border-t"
                            style={{ borderColor: border }}
                        >
                            <div className="text-[10px] font-bold uppercase tracking-widest mb-2" style={{ color: textMuted }}>
                                Key Signals
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                                {agent.signals.map((signal: any) => (
                                    <div key={signal.name} className="text-center p-2 rounded-xl" style={glassSunken}>
                                        <div className="text-[10px] truncate" style={{ color: textMuted, opacity: 0.7 }}>
                                            {signal.name}
                                        </div>
                                        <div className="text-sm font-semibold mt-0.5" style={{ color: textPri, opacity: 0.85 }}>
                                            {signal.value}
                                            <span className="text-[10px] font-normal ml-0.5" style={{ color: textMuted }}>
                                                {signal.unit}
                                            </span>
                                        </div>
                                        <div
                                            className="w-1.5 h-1.5 rounded-full mx-auto mt-1"
                                            style={{
                                                background: signal.impact === 'positive' ? '#16a34a' :
                                                    signal.impact === 'negative' ? '#dc2626' : textMuted,
                                            }}
                                        />
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Expand indicator */}
                <div className="flex justify-center mt-2">
                    <motion.div
                        animate={{ rotate: expanded ? 180 : 0 }}
                        transition={{ duration: 0.2 }}
                        style={{ color: textMuted, opacity: 0.35 }}
                    >
                        <ChevronDown className="w-4 h-4" />
                    </motion.div>
                </div>
            </div>
        </motion.div>
    );
}

/* ─── Main Page ──────────────────────────────────────────────────────────── */
export default function TwinPage() {
    const { mode } = useAgentMode();

    const trajectories = useMemo(() => {
        const result: Record<string, any> = {};
        mockAgentRisks.forEach((agent) => { result[agent.id] = getAgentTrajectory(agent.id); });
        return result;
    }, []);

    const { setSubsystem } = useSubsystem();
    useEffect(() => { setSubsystem("Twin"); }, [setSubsystem]);

    const visibleAgents = mode === 'mvp'
        ? mockAgentRisks.filter((a) => a.id === 'cardio' || a.id === 'mental')
        : mockAgentRisks;

    const gridCols = mode === 'mvp' ? 'md:grid-cols-2' : 'md:grid-cols-2 lg:grid-cols-4';

    return (
        <div
            className="h-full overflow-y-auto"
            style={{ scrollbarWidth: 'thin', scrollbarColor: `${accent} transparent` }}
        >
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                className="max-w-6xl mx-auto px-4 pt-6 pb-12"
            >
                {/* ── Header ── */}
                <div className="flex items-start justify-between mb-8">
                    <div>
                        <div className="flex items-center gap-2 mb-1.5">
                            <div
                                className="w-6 h-6 rounded-lg flex items-center justify-center"
                                style={glassSunken}
                            >
                                <Activity className="w-3.5 h-3.5" style={{ color: accent }} />
                            </div>
                            <span className="text-xs font-medium uppercase tracking-widest" style={{ color: textMuted }}>
                                Live Health Status
                            </span>
                        </div>
                        <h1
                            className="text-3xl font-black tracking-tight"
                            style={{ color: textPri, fontFamily: 'var(--font-orbitron, var(--font-oxanium))' }}
                        >
                            Your Digital Twin
                        </h1>
                        <p className="text-sm mt-1" style={{ color: textMuted }}>
                            Real-time trajectory analysis powered by {mode === 'mvp' ? '2 core' : '8 specialized'} agents
                        </p>
                    </div>
                </div>

                {/* ── Summary Cards ── */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8"
                >
                    {[
                        { icon: Shield,        label: 'Overall Risk',   value: '31%',               color: '#ca8a04' },
                        { icon: Zap,           label: 'Active Agents',  value: visibleAgents.length, color: accent   },
                        { icon: Target,        label: 'Data Quality',   value: '87%',               color: '#16a34a' },
                        { icon: AlertTriangle, label: 'Alerts',         value: '2',                 color: '#ea580c' },
                    ].map((item, i) => (
                        <motion.div
                            key={item.label}
                            initial={{ opacity: 0, scale: 0.94 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.12 + i * 0.04 }}
                            className="p-4 rounded-2xl"
                            style={glass}
                        >
                            <item.icon className="w-4 h-4 mb-2" style={{ color: item.color }} />
                            <div className="text-2xl font-bold" style={{ color: textPri }}>{item.value}</div>
                            <div className="text-[11px] mt-0.5" style={{ color: textMuted }}>{item.label}</div>
                        </motion.div>
                    ))}
                </motion.div>

                {/* ── Agent Grid ── */}
                <div className={`grid gap-3 ${gridCols}`}>
                    {visibleAgents.map((agent, index) => (
                        <AgentCard
                            key={agent.id}
                            agent={agent}
                            trajectory={trajectories[agent.id]}
                            index={index}
                        />
                    ))}
                </div>

                {/* ── Footer ── */}
                <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="text-center text-[11px] mt-10"
                    style={{ color: textMuted, opacity: 0.5 }}
                >
                    Predictions show 10th–90th percentile range • Updated 15 minutes ago
                </motion.p>
            </motion.div>
        </div>
    );
}