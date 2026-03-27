'use client';

/**
 * Predictions page — theme-aware (dark + light).
 *
 * Every colour token comes from the ThemeRegistry CSS variables that
 * useApplyTheme() writes onto :root:
 *
 *   --accent        e.g. #38bdf8  (Clinic dark) / #0284c7 (Clinic light)
 *   --textPrimary   e.g. #e6f6ff  / #0b1f2a
 *   --textMuted     e.g. #6b8ca3  / #4b6b7f
 *   --surface       e.g. rgba(56,189,248,0.08) / rgba(2,132,199,0.08)
 */

import {useState, useMemo, useEffect} from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Play, RotateCcw, Sparkles, Clock,
    TrendingUp, TrendingDown, ArrowRight, Activity, Zap,
} from 'lucide-react';
import { mockScenarios, simulateScenario, generateTrajectory } from './components/mockData';
import { WhatIfScenario, ScenarioResult, TimeHorizon } from './components/types';
import {useSubsystem} from "@/app/system/theme/subsystem-context";

/* ─── shared style tokens ────────────────────────────────────────────────── */
const surface      = 'var(--surface)';
const surface2     = 'color-mix(in srgb, var(--surface) 200%, transparent)';
const textPri      = 'var(--textPrimary)';
const textMuted    = 'var(--textMuted)';
const accent       = 'var(--accent)';
const border       = 'color-mix(in srgb, var(--textMuted) 18%, transparent)';
const borderAccent = 'color-mix(in srgb, var(--accent) 35%, transparent)';

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



/* ─── ScenarioSlider ─────────────────────────────────────────────────────── */
function ScenarioSlider({
                            scenario, value, onChange, index,
                        }: {
    scenario: WhatIfScenario;
    value: number;
    onChange: (v: number) => void;
    index: number;
}) {
    const progress  = ((value - scenario.minValue) / (scenario.maxValue - scenario.minValue)) * 100;
    const isDefault = value === scenario.defaultValue;
    const delta     = value - scenario.defaultValue;
    const deltaSign = delta > 0 ? '+' : '';

    const {setSubsystem} = useSubsystem();
    useEffect(() => {
        setSubsystem("Predictions");
    }, [setSubsystem]);

    return (
        <motion.div
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.06, ease: [0.22, 1, 0.36, 1] }}
        >
            <div
                className="relative p-4 rounded-2xl border transition-all duration-300"
                style={{
                    ...glass,
                    border: `1px solid ${isDefault ? border : borderAccent}`,
                    boxShadow: isDefault
                        ? glass.boxShadow
                        : ` 0 4px 24px rgba(0,0,0,0.14),  0 0 0 1px ${borderAccent}`,}}>
                {!isDefault && (
                    <div
                        className="absolute left-0 top-3 bottom-3 w-0.5 rounded-full"
                        style={{ background: accent }}
                    />
                )}

                <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2.5">
                        <div
                            className="w-8 h-8 rounded-xl flex items-center justify-center text-base"
                            style={{ background: surface2 }}
                        >
                            {scenario.icon}
                        </div>
                        <div>
                            <div className="text-sm font-semibold leading-none mb-0.5" style={{ color: textPri }}>
                                {scenario.name}
                            </div>
                            <div className="text-xs" style={{ color: textMuted }}>
                                {scenario.description}
                            </div>
                        </div>
                    </div>
                    <div className="text-right">
                        <div className="text-lg font-bold tabular-nums leading-none" style={{ color: accent }}>
                            {value}
                            <span className="text-xs font-normal ml-1" style={{ color: textMuted }}>
                                {scenario.unit}
                            </span>
                        </div>
                        {!isDefault && (
                            <div
                                className="text-xs font-medium mt-0.5"
                                style={{ color: delta > 0 ? '#16a34a' : '#dc2626' }}
                            >
                                {deltaSign}{delta.toFixed(scenario.step < 1 ? 1 : 0)}
                            </div>
                        )}
                    </div>
                </div>

                <div className="relative">
                    <div
                        className="relative h-1.5 rounded-full overflow-hidden"
                        style={{ background: border }}
                    >
                        <div
                            className="absolute top-0 bottom-0 w-px z-10"
                            style={{
                                background: textMuted,
                                opacity: 0.4,
                                left: `${((scenario.defaultValue - scenario.minValue) / (scenario.maxValue - scenario.minValue)) * 100}%`,
                            }}
                        />
                        <div
                            className="absolute left-0 top-0 bottom-0 rounded-full transition-all duration-100"
                            style={{
                                width: `${progress}%`,
                                background: `linear-gradient(90deg, color-mix(in srgb, ${accent} 55%, transparent), ${accent})`,
                            }}
                        />
                    </div>
                    <input
                        type="range"
                        min={scenario.minValue}
                        max={scenario.maxValue}
                        step={scenario.step}
                        value={value}
                        onChange={(e) => onChange(parseFloat(e.target.value))}
                        className="absolute inset-0 w-full opacity-0 cursor-pointer"
                        style={{ height: '100%' }}
                    />
                </div>

                <div className="flex justify-between mt-1.5">
                    <span className="text-[10px]" style={{ color: textMuted, opacity: 0.55 }}>
                        {scenario.minValue}
                    </span>
                    <span className="text-[10px]" style={{ color: textMuted, opacity: 0.55 }}>
                        {scenario.maxValue}
                    </span>
                </div>
            </div>
        </motion.div>
    );
}

/* ─── ScenarioResultCard ─────────────────────────────────────────────────── */
function ScenarioResultCard({ result, index }: { result: ScenarioResult; index: number }) {
    const isImprovement = result.absoluteChange < 0;

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.04, ease: [0.22, 1, 0.36, 1] }}
            className="relative p-3.5 rounded-2xl border overflow-hidden"
            style={{
                ...glass,
                border: `1px solid ${border}`,
            }}>
            <div
                className="absolute top-3 left-3 w-8 h-8 rounded-full pointer-events-none"
                style={{ background: result.agentColor, opacity: 0.1, filter: 'blur(12px)' }}
            />

            <div className="relative flex items-center gap-3">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                    <div
                        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                        style={{ background: result.agentColor, boxShadow: `0 0 6px ${result.agentColor}55` }}
                    />
                    <span className="text-sm font-medium truncate" style={{ color: textPri, opacity: 0.85 }}>
                        {result.agentName}
                    </span>
                </div>

                <div className="flex items-center gap-1.5 flex-shrink-0">
                    <span className="text-sm font-bold tabular-nums" style={{ color: textMuted }}>
                        {(result.baselineRisk * 100).toFixed(0)}%
                    </span>
                    <ArrowRight className="w-3 h-3" style={{ color: textMuted, opacity: 0.5 }} />
                    <span
                        className="text-sm font-bold tabular-nums"
                        style={{ color: isImprovement ? '#16a34a' : '#dc2626' }}
                    >
                        {(result.projectedRisk * 100).toFixed(0)}%
                    </span>
                </div>

                <div
                    className="flex items-center gap-1 px-2 py-0.5 rounded-lg flex-shrink-0"
                    style={{
                        background: isImprovement ? 'rgba(22,163,74,0.12)' : 'rgba(220,38,38,0.12)',
                        border: `1px solid ${isImprovement ? 'rgba(22,163,74,0.25)' : 'rgba(220,38,38,0.25)'}`,
                    }}
                >
                    {isImprovement
                        ? <TrendingDown className="w-3 h-3" style={{ color: '#16a34a' }} />
                        : <TrendingUp   className="w-3 h-3" style={{ color: '#dc2626' }} />
                    }
                    <span
                        className="text-xs font-semibold tabular-nums"
                        style={{ color: isImprovement ? '#16a34a' : '#dc2626' }}
                    >
                        {isImprovement ? '' : '+'}{result.percentChange.toFixed(1)}%
                    </span>
                </div>
            </div>

            <div
                className="relative h-1 rounded-full overflow-hidden mt-3"
                style={{ background: border }}
            >
                <motion.div
                    initial={{ width: `${result.baselineRisk * 100}%` }}
                    animate={{ width: `${result.projectedRisk * 100}%` }}
                    transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1], delay: index * 0.04 }}
                    className="absolute left-0 top-0 bottom-0 rounded-full"
                    style={{ background: isImprovement ? '#16a34a' : '#dc2626' }}
                />
                <div
                    className="absolute top-0 bottom-0 w-px"
                    style={{ background: textMuted, opacity: 0.35, left: `${result.baselineRisk * 100}%` }}
                />
            </div>

            <div className="flex items-center justify-between mt-1.5">
                <span className="text-[10px]" style={{ color: textMuted, opacity: 0.5 }}>
                    {(result.confidence * 100).toFixed(0)}% confidence
                </span>
                <span className="text-[10px]" style={{ color: textMuted, opacity: 0.5 }}>
                    {result.timeHorizon}d horizon
                </span>
            </div>
        </motion.div>
    );
}

/* ─── MiniTrajectoryChart ────────────────────────────────────────────────── */
function MiniTrajectoryChart({
                                 trajectories, agentColor, agentName, isImprovement,
                             }: {
    trajectories: any[];
    agentColor: string;
    agentName: string;
    isImprovement: boolean;
}) {
    const W = 200, H = 80, P = 8;
    const cw = W - P * 2, ch = H - P * 2;
    if (!trajectories.length) return null;

    const maxDay = Math.max(...trajectories.map((t) => t.day));
    const xs = (d: number) => P + (d / maxDay) * cw;
    const ys = (v: number) => P + (1 - v) * ch;

    const scenario = trajectories
        .map((t, i) => `${i ? 'L' : 'M'}${xs(t.day).toFixed(1)},${ys(t.scenario ?? t.baseline).toFixed(1)}`)
        .join(' ');
    const bandTop = trajectories
        .map((t, i) => `${i ? 'L' : 'M'}${xs(t.day).toFixed(1)},${ys(t.optimistic).toFixed(1)}`)
        .join(' ');
    const bandBot = [...trajectories]
        .reverse()
        .map((t) => `L${xs(t.day).toFixed(1)},${ys(t.pessimistic).toFixed(1)}`)
        .join(' ');

    const baseY = ys(trajectories[trajectories.length - 1]?.baseline ?? 0.5).toFixed(1);

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative p-4 rounded-2xl border"
            style={{
                ...glass,
                border: `1px solid ${border}`,
            }}>
            <div className="flex items-center gap-2 mb-3">
                <div
                    className="w-2 h-2 rounded-full"
                    style={{ background: agentColor, boxShadow: `0 0 4px ${agentColor}55` }}
                />
                <span className="text-xs font-medium" style={{ color: textPri, opacity: 0.75 }}>
                    {agentName}
                </span>
                <div
                    className="ml-auto text-xs px-1.5 py-0.5 rounded-md font-medium"
                    style={{
                        background: isImprovement ? 'rgba(22,163,74,0.12)' : 'rgba(220,38,38,0.12)',
                        color: isImprovement ? '#16a34a' : '#dc2626',
                    }}
                >
                    {isImprovement ? '↓' : '↑'} {isImprovement ? 'improving' : 'at risk'}
                </div>
            </div>

            <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
                <path d={`${bandTop} ${bandBot} Z`} fill={agentColor} opacity={0.1} />
                <path
                    d={scenario}
                    fill="none"
                    stroke={agentColor}
                    strokeWidth={1.5}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
                <line
                    x1={P} y1={baseY} x2={W - P} y2={baseY}
                    stroke="currentColor"
                    strokeOpacity={0.2}
                    strokeDasharray="3,3"
                    strokeWidth={1}
                    style={{ color: textMuted } as React.CSSProperties}
                />
            </svg>

            <div className="flex gap-4 mt-1">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-px" style={{ background: agentColor }} />
                    <span className="text-[10px]" style={{ color: textMuted, opacity: 0.6 }}>Projected</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3" style={{ borderTop: `1px dashed ${textMuted}`, opacity: 0.4 }} />
                    <span className="text-[10px]" style={{ color: textMuted, opacity: 0.6 }}>Baseline</span>
                </div>
            </div>
        </motion.div>
    );
}

/* ─── Main Page ──────────────────────────────────────────────────────────── */
export default function PredictionsPage() {
    const [scenarioValues, setScenarioValues] = useState<Record<string, number>>(() => {
        const init: Record<string, number> = {};
        mockScenarios.forEach((s) => { init[s.id] = s.defaultValue; });
        return init;
    });
    const [selectedHorizon, setSelectedHorizon] = useState<TimeHorizon>(30);
    const [results,   setResults]   = useState<ScenarioResult[]>([]);
    const [hasRun,    setHasRun]    = useState(false);
    const [isRunning, setIsRunning] = useState(false);

    const changedCount = useMemo(
        () => mockScenarios.filter((s) => Math.abs(scenarioValues[s.id] - s.defaultValue) > 0.01).length,
        [scenarioValues],
    );

    const runSimulation = async () => {
        setIsRunning(true);
        await new Promise((r) => setTimeout(r, 320));

        const allResults: ScenarioResult[] = [];
        mockScenarios.forEach((scenario) => {
            const value     = scenarioValues[scenario.id];
            const deviation = value - scenario.defaultValue;
            if (Math.abs(deviation) > 0.01)
                allResults.push(...simulateScenario(scenario.id, deviation, scenario.defaultValue));
        });

        const agentResults = new Map<string, ScenarioResult>();
        allResults.forEach((r) => {
            const ex = agentResults.get(r.agentId);
            if (!ex || Math.abs(r.absoluteChange) > Math.abs(ex.absoluteChange))
                agentResults.set(r.agentId, r);
        });

        setResults(Array.from(agentResults.values()));
        setHasRun(true);
        setIsRunning(false);
    };

    const resetScenarios = () => {
        const init: Record<string, number> = {};
        mockScenarios.forEach((s) => { init[s.id] = s.defaultValue; });
        setScenarioValues(init);
        setResults([]);
        setHasRun(false);
    };

    const topResults = useMemo(
        () => [...results]
            .sort((a, b) => Math.abs(b.absoluteChange) - Math.abs(a.absoluteChange))
            .slice(0, 4),
        [results],
    );

    const improving = results.filter((r) => r.absoluteChange < 0).length;
    const atRisk    = results.filter((r) => r.absoluteChange > 0).length;

    return (
        <div
            className="h-full overflow-y-auto"
            style={{ scrollbarWidth: 'thin', scrollbarColor: `${accent} transparent` }}
        >
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                className="max-w-6xl mx-auto px-4 pt-6 pb-12"
            >
                {/* ── Header ── */}
                <div className="flex items-start justify-between mb-8">
                    <div>
                        <div className="flex items-center gap-2 mb-1.5">
                            <div
                                className="w-6 h-6 rounded-lg flex items-center justify-center"
                                style={{ background: surface2 }}
                            >
                                <Zap className="w-3.5 h-3.5" style={{ color: accent }} />
                            </div>
                            <span className="text-xs font-medium uppercase tracking-widest" style={{ color: textMuted }}>
                                Simulation Engine
                            </span>
                        </div>
                        <h1
                            className="text-3xl font-black tracking-tight"
                            style={{ color: textPri, fontFamily: 'var(--font-orbitron, var(--font-oxanium))' }}
                        >
                            What-If Predictions
                        </h1>
                        <p className="text-sm mt-1" style={{ color: textMuted }}>
                            Model lifestyle changes and forecast health trajectory impact
                        </p>
                    </div>

                    <AnimatePresence>
                        {changedCount > 0 && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.8, x: 8 }}
                                animate={{ opacity: 1, scale: 1, x: 0 }}
                                exit={{ opacity: 0, scale: 0.8, x: 8 }}
                                className="flex items-center gap-2 px-3 py-1.5 rounded-xl border"
                                style={{ background: surface2, borderColor: borderAccent }}
                            >
                                <Activity className="w-3.5 h-3.5" style={{ color: accent }} />
                                <span className="text-xs font-semibold" style={{ color: accent }}>
                                    {changedCount} param{changedCount !== 1 ? 's' : ''} modified
                                </span>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* ── Time Horizon ── */}
                <div
                    className="flex items-center gap-4 px-4 py-3 rounded-2xl mb-8 border"
                    style={{
                        ...glass,
                        border: `1px solid ${border}`,
                    }}>
                    <div className="flex items-center gap-2">
                        <Clock className="w-3.5 h-3.5" style={{ color: textMuted }} />
                        <span className="text-xs font-medium uppercase tracking-wider" style={{ color: textMuted }}>
                            Horizon
                        </span>
                    </div>
                    <div className="flex gap-1.5">
                        {([7, 30, 90, 180] as TimeHorizon[]).map((h) => (
                            <button
                                key={h}
                                onClick={() => setSelectedHorizon(h)}
                                className="px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200"
                                style={{
                                    background: selectedHorizon === h ? surface2 : 'transparent',
                                    color: selectedHorizon === h ? accent : textMuted,
                                    border: selectedHorizon === h
                                        ? `1px solid ${borderAccent}`
                                        : `1px solid ${border}`,
                                }}
                            >
                                {h}d
                            </button>
                        ))}
                    </div>
                    <div className="ml-auto text-xs hidden md:block" style={{ color: textMuted, opacity: 0.5 }}>
                        Confidence intervals widen over longer horizons
                    </div>
                </div>

                {/* ── Main Grid ── */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                    {/* LEFT — parameters */}
                    <div>
                        <div className="flex items-center justify-between mb-4">
                            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: textMuted }}>
                                Parameters
                            </span>
                            <button
                                onClick={resetScenarios}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-all"
                                style={{
                                    ...glassSunken,
                                    color: textMuted,
                                }}>
                                <RotateCcw className="w-3 h-3" />
                                Reset
                            </button>
                        </div>

                        <div className="space-y-2.5">
                            {mockScenarios.map((scenario, index) => (
                                <ScenarioSlider
                                    key={scenario.id}
                                    scenario={scenario}
                                    value={scenarioValues[scenario.id]}
                                    onChange={(value) =>
                                        setScenarioValues((prev) => ({ ...prev, [scenario.id]: value }))
                                    }
                                    index={index}
                                />
                            ))}
                        </div>

                        {/* Run Button */}
                        <motion.button
                            whileHover={changedCount > 0 ? { scale: 1.015 } : {}}
                            whileTap={changedCount > 0 ? { scale: 0.985 } : {}}
                            onClick={runSimulation}
                            disabled={isRunning || changedCount === 0}
                            className="relative w-full mt-5 py-3.5 rounded-2xl font-bold text-sm overflow-hidden transition-all duration-300"
                            style={{
                                background: changedCount === 0
                                    ? surface
                                    : `linear-gradient(135deg, color-mix(in srgb, ${accent} 80%, #6366f1), ${accent})`,
                                color: changedCount === 0 ? textMuted : 'rgba(0,0,0,0.85)',
                                border: changedCount === 0 ? `1px solid ${border}` : 'none',
                                cursor: changedCount === 0 ? 'not-allowed' : 'pointer',
                                opacity: changedCount === 0 ? 0.6 : 1,
                            }}
                        >
                            <AnimatePresence mode="wait">
                                {isRunning ? (
                                    <motion.span
                                        key="running"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="flex items-center justify-center gap-2"
                                    >
                                        <motion.div
                                            animate={{ rotate: 360 }}
                                            transition={{ repeat: Infinity, duration: 0.8, ease: 'linear' }}
                                            className="w-4 h-4 border-2 rounded-full"
                                            style={{ borderColor: 'rgba(0,0,0,0.25)', borderTopColor: 'rgba(0,0,0,0.75)' }}
                                        />
                                        Simulating…
                                    </motion.span>
                                ) : (
                                    <motion.span
                                        key="idle"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="flex items-center justify-center gap-2"
                                    >
                                        <Play className="w-4 h-4" />
                                        {changedCount === 0
                                            ? 'Adjust a parameter to simulate'
                                            : `Run Simulation · ${changedCount} change${changedCount !== 1 ? 's' : ''}`}
                                    </motion.span>
                                )}
                            </AnimatePresence>
                        </motion.button>
                    </div>

                    {/* RIGHT — results */}
                    <div>
                        <span className="text-xs font-bold uppercase tracking-widest mb-4 block" style={{ color: textMuted }}>
                            Predicted Impact
                        </span>

                        <AnimatePresence mode="wait">
                            {!hasRun ? (
                                <motion.div
                                    key="empty"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="flex flex-col items-center justify-center py-16 rounded-2xl border text-center"
                                    style={{ background: surface, borderColor: border, borderStyle: 'dashed' }}
                                >
                                    <div
                                        className="w-12 h-12 rounded-2xl flex items-center justify-center mb-4"
                                        style={{ background: surface2 }}
                                    >
                                        <Sparkles className="w-5 h-5" style={{ color: textMuted, opacity: 0.5 }} />
                                    </div>
                                    <p className="text-sm max-w-[220px] leading-relaxed" style={{ color: textMuted }}>
                                        Adjust the parameters and run simulation to forecast health trajectory changes
                                    </p>
                                </motion.div>
                            ) : results.length === 0 ? (
                                <motion.div
                                    key="no-results"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="flex flex-col items-center justify-center py-16 rounded-2xl border text-center"
                                    style={{
                                    ...glass,
                                    border: `1px solid ${border}`,
                                }}>
                                    <p className="text-sm" style={{ color: textMuted }}>
                                        No significant changes. Try larger parameter shifts.
                                    </p>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="results"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="space-y-2.5"
                                >
                                    {/* Summary */}
                                    <motion.div
                                        initial={{ opacity: 0, y: -8 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="flex items-center justify-between px-4 py-3 rounded-2xl border"
                                        style={{
                                            ...glass,
                                            border: `1px solid ${border}`,
                                        }}
                                    >
                                        <span className="text-xs font-medium uppercase tracking-wider" style={{ color: textMuted }}>
                                            Summary
                                        </span>
                                        <div className="flex items-center gap-4">
                                            <div className="flex items-center gap-1.5">
                                                <div className="w-2 h-2 rounded-full" style={{ background: '#16a34a' }} />
                                                <span className="text-sm font-bold" style={{ color: textPri }}>{improving}</span>
                                                <span className="text-xs" style={{ color: textMuted }}>improving</span>
                                            </div>
                                            <div className="w-px h-4" style={{ background: border }} />
                                            <div className="flex items-center gap-1.5">
                                                <div className="w-2 h-2 rounded-full" style={{ background: '#dc2626' }} />
                                                <span className="text-sm font-bold" style={{ color: textPri }}>{atRisk}</span>
                                                <span className="text-xs" style={{ color: textMuted }}>at risk</span>
                                            </div>
                                        </div>
                                    </motion.div>

                                    {topResults.map((result, i) => (
                                        <ScenarioResultCard key={result.agentId} result={result} index={i} />
                                    ))}

                                    {topResults.length > 0 && (
                                        <div className="pt-2">
                                            <span
                                                className="text-xs uppercase tracking-widest mb-3 font-medium block"
                                                style={{ color: textMuted, opacity: 0.7 }}
                                            >
                                                Trajectory Projection
                                            </span>
                                            <div className="grid grid-cols-2 gap-2.5">
                                                {topResults.slice(0, 2).map((result) => {
                                                    const traj = generateTrajectory(
                                                        result.baselineRisk,
                                                        result.projectedRisk,
                                                        selectedHorizon,
                                                    );
                                                    return (
                                                        <MiniTrajectoryChart
                                                            key={result.agentId}
                                                            trajectories={traj}
                                                            agentColor={result.agentColor}
                                                            agentName={result.agentName}
                                                            isImprovement={result.absoluteChange < 0}
                                                        />
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>

                {/* ── Disclaimer ── */}
                <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="text-center text-[11px] mt-10 leading-relaxed"
                    style={{ color: textMuted, opacity: 0.5 }}
                >
                    Predictions are based on historical patterns and may not reflect actual outcomes.
                    <br />
                    Confidence intervals represent model uncertainty, not medical certainty.
                </motion.p>
            </motion.div>
        </div>
    );
}