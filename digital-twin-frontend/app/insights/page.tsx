'use client';

/**
 * Insights page — theme-aware (dark + light).
 *
 * Every colour token comes from the ThemeRegistry CSS variables that
 * useApplyTheme() writes onto :root:
 *
 *   --accent        e.g. #38bdf8  (Clinic dark) / #0284c7 (Clinic light)
 *   --textPrimary   e.g. #e6f6ff  / #0b1f2a
 *   --textMuted     e.g. #6b8ca3  / #4b6b7f
 *   --surface       e.g. rgba(56,189,248,0.08) / rgba(2,132,199,0.08)
 */

import { motion } from 'framer-motion';
import {
    Activity, TrendingUp, TrendingDown, Link2, Calendar,
    AlertTriangle, CheckCircle, Info, Lightbulb, Clock,
    Sparkles, ArrowRight, Zap,
} from 'lucide-react';
import { mockInsights, mockTrends, mockCorrelations, mockWeeklySummary } from './components/mockData';
import {useEffect} from "react";
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
    boxShadow: `
        0 4px 24px rgba(0,0,0,0.12),
        inset 0 1px 0 rgba(255,255,255,0.04)
    `,
};

const glassSunken = {
    background: 'color-mix(in srgb, var(--surface) 55%, transparent)',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    border: `1px solid ${border}`,
};

/* ─── InsightCard ────────────────────────────────────────────────────────── */
function InsightCard({ insight, index }: { insight: any; index: number }) {

    const {setSubsystem} = useSubsystem();
    useEffect(() => {
        setSubsystem("Insights");
    }, [setSubsystem]);

    const styles = {
        warning: {
            tint: 'rgba(220,38,38,0.42)',
            border: 'rgba(220,38,38,0.55)',
            icon: AlertTriangle,
            color: '#dc2626',
        },
        success: {
            tint: 'rgba(22,163,74,0.40)',
            border: 'rgba(22,163,74,0.55)',
            icon: CheckCircle,
            color: '#16a34a',
        },
        action: {
            tint: 'rgba(168,85,247,0.40)',
            border: 'rgba(168,85,247,0.55)',
            icon: Lightbulb,
            color: '#a855f7',
        },
        info: {
            tint: 'rgba(59,130,246,0.40)',
            border: 'rgba(59,130,246,0.55)',
            icon: Info,
            color: '#3b82f6',
        },
    };
    const style = styles[insight.type as keyof typeof styles] || styles.info;
    const Icon = style.icon;

    const timeAgo = (date: Date) => {
        const mins = Math.round((Date.now() - date.getTime()) / 60000);
        if (mins < 60) return `${mins}m ago`;
        const hrs = Math.round(mins / 60);
        if (hrs < 24) return `${hrs}h ago`;
        return `${Math.round(hrs / 24)}d ago`;
    };

    return (
        <motion.div
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            className="relative p-4 rounded-2xl border overflow-hidden"
            style={{
                ...glass,
                border: `1px solid ${style.border}`,
                background: `
                    linear-gradient(
                        140deg,
                        color-mix(in srgb, ${style.tint} 95%, transparent),
                        color-mix(in srgb, ${style.tint} 60%, var(--surface))
                    )`,
                    }}>
            <div className="flex items-start gap-3.5">
                <div
                    className="p-2 rounded-xl flex-shrink-0"
                    style={{ ...glassSunken }}
                >
                    <Icon className="w-4 h-4" style={{ color: style.color }} />
                </div>
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-semibold leading-snug" style={{ color: textPri }}>
                            {insight.title}
                        </span>
                        {insight.agentColor && (
                            <div
                                className="w-2 h-2 rounded-full flex-shrink-0"
                                style={{ background: insight.agentColor }}
                            />
                        )}
                    </div>
                    <p className="text-xs leading-relaxed mb-2.5" style={{ color: textMuted }}>
                        {insight.description}
                    </p>
                    <div className="flex items-center justify-between">
                        <span className="text-[11px] flex items-center gap-1" style={{ color: textMuted, opacity: 0.6 }}>
                            <Clock className="w-3 h-3" />
                            {timeAgo(insight.timestamp)}
                        </span>
                        {insight.actionable && insight.action && (
                            <button
                                className="flex items-center gap-1 text-[11px] font-semibold transition-colors"
                                style={{ color: accent }}
                            >
                                {insight.action.label}
                                <ArrowRight className="w-3 h-3" />
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

/* ─── TrendCard ──────────────────────────────────────────────────────────── */
function TrendCard({ trend, index }: { trend: any; index: number }) {
    const isNegativeGood = ['Resting HR', 'Screen Time', 'Metabolic Risk'].some((s) => trend.signal.includes(s));
    const isGood = isNegativeGood ? trend.trend === 'down' : trend.trend === 'up';
    const color = trend.trend === 'stable' ? textMuted : isGood ? '#16a34a' : '#dc2626';

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.03 }}
            className="p-3.5 rounded-2xl border"
            style={{
                ...glass,
                border: `1px solid ${border}`,
            }}>
            <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium" style={{ color: textPri, opacity: 0.85 }}>
                    {trend.signal}
                </span>
                <div style={{ color }}>
                    {trend.trend === 'up' ? <TrendingUp className="w-3.5 h-3.5" /> :
                     trend.trend === 'down' ? <TrendingDown className="w-3.5 h-3.5" /> :
                     <Activity className="w-3.5 h-3.5" />}
                </div>
            </div>
            <div className="flex items-end justify-between">
                <div>
                    <span className="text-xl font-bold tabular-nums" style={{ color: textPri }}>
                        {trend.current}
                    </span>
                    <span className="text-[11px] ml-1" style={{ color: textMuted }}>
                        {trend.unit}
                    </span>
                </div>
                <span className="text-xs font-semibold tabular-nums" style={{ color }}>
                    {trend.change > 0 ? '+' : ''}{trend.change.toFixed(1)}%
                </span>
            </div>
        </motion.div>
    );
}

/* ─── CorrelationItem ─────────────────────────────────────────────────────── */
function CorrelationItem({ corr, index }: { corr: any; index: number }) {
    const barColor = Math.abs(corr.correlation) > 0.6 ? '#16a34a' : Math.abs(corr.correlation) > 0.3 ? '#ca8a04' : '#6b7280';

    return (
        <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="p-3 rounded-xl border"
            style={{
                ...glass,
                border: `1px solid ${border}`,
            }}>
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5 text-[11px]" style={{ color: textMuted }}>
                    <span>{corr.signal1}</span>
                    <span style={{ opacity: 0.4 }}>↔</span>
                    <span>{corr.signal2}</span>
                </div>
                <span
                    className="text-[10px] font-medium px-1.5 py-0.5 rounded-md capitalize"
                    style={{ background: `color-mix(in srgb, ${barColor} 15%, transparent)`, color: barColor }}
                >
                    {corr.strength}
                </span>
            </div>
            <div className="relative h-1.5 rounded-full overflow-hidden" style={{
                    background: 'color-mix(in srgb, var(--surface) 40%, transparent)'
                }}>
                <div
                    className="absolute top-0 bottom-0 left-0 rounded-full transition-all"
                    style={{
                        width: `${Math.abs(corr.correlation) * 100}%`,
                        background: corr.correlation > 0 ? barColor : '#dc2626',
                    }}
                />
            </div>
            <div className="flex justify-between mt-1.5">
                <span className="text-[10px] tabular-nums" style={{ color: textMuted, opacity: 0.6 }}>
                    r = {corr.correlation.toFixed(2)}
                </span>
                <span className="text-[10px]" style={{ color: textMuted, opacity: 0.6 }}>
                    {corr.correlation > 0 ? 'positive' : 'negative'}
                </span>
            </div>
        </motion.div>
    );
}

/* ─── Main Page ──────────────────────────────────────────────────────────── */
export default function InsightsPage() {
    const sortedInsights = [...mockInsights].sort((a, b) => {
        const priority = { high: 0, medium: 1, low: 2 };
        return priority[a.priority as keyof typeof priority] - priority[b.priority as keyof typeof priority];
    });

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
                                style={glassSunken}
                            >
                                <Sparkles className="w-3.5 h-3.5" style={{ color: accent }} />
                            </div>
                            <span className="text-xs font-medium uppercase tracking-widest" style={{ color: textMuted }}>
                                AI-Powered Analysis
                            </span>
                        </div>
                        <h1
                            className="text-3xl font-black tracking-tight"
                            style={{ color: textPri, fontFamily: 'var(--font-orbitron, var(--font-oxanium))' }}
                        >
                            Health Insights
                        </h1>
                        <p className="text-sm mt-1" style={{ color: textMuted }}>
                            Personalized recommendations based on your health data patterns
                        </p>
                    </div>
                </div>

                {/* ── Weekly Summary Banner ── */}
                <motion.div
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="relative p-5 rounded-2xl border mb-8 overflow-hidden"
                    style={{
                        ...glass,
                        border: `1px solid ${border}`,
                    }}>
                    <div
                        className="absolute -top-12 -right-12 w-40 h-40 rounded-full pointer-events-none"
                        style={{ background: accent, opacity: 0.04, filter: 'blur(32px)' }}
                    />
                    <div className="relative">
                        <div className="flex items-center gap-2 mb-4">
                            <Calendar className="w-4 h-4" style={{ color: accent }} />
                            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: textMuted }}>
                                Weekly Summary
                            </span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                            <div>
                                <div className="text-[11px] font-medium uppercase tracking-wider mb-1" style={{ color: textMuted }}>
                                    Overall Score
                                </div>
                                <div className="text-5xl font-black" style={{ color: accent }}>
                                    {mockWeeklySummary.overallScore}
                                </div>
                                <div className="text-xs mt-0.5" style={{ color: textMuted }}>out of 100</div>
                            </div>
                            <div>
                                <div className="text-[11px] font-medium uppercase tracking-wider mb-2" style={{ color: textMuted }}>
                                    Improvements
                                </div>
                                <div className="space-y-1">
                                    {mockWeeklySummary.topImprovements.map((item, i) => (
                                        <div key={i} className="flex items-center gap-1.5 text-xs" style={{ color: '#16a34a' }}>
                                            <TrendingDown className="w-3 h-3" />
                                            {item}
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <div className="text-[11px] font-medium uppercase tracking-wider mb-2" style={{ color: textMuted }}>
                                    Declines
                                </div>
                                <div className="space-y-1">
                                    {mockWeeklySummary.topDeclines.map((item, i) => (
                                        <div key={i} className="flex items-center gap-1.5 text-xs" style={{ color: '#dc2626' }}>
                                            <TrendingUp className="w-3 h-3" />
                                            {item}
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <div className="text-[11px] font-medium uppercase tracking-wider mb-2" style={{ color: textMuted }}>
                                    Key Insights
                                </div>
                                <div className="space-y-1">
                                    {mockWeeklySummary.keyInsights.map((item, i) => (
                                        <div key={i} className="text-xs" style={{ color: textMuted }}>• {item}</div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* ── Main Grid ── */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Insights Column */}
                    <div className="lg:col-span-2">
                        <div className="flex items-center gap-2 mb-4">
                            <Activity className="w-4 h-4" style={{ color: accent }} />
                            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: textMuted }}>
                                Recent Insights
                            </span>
                        </div>
                        <div
                            className="space-y-2.5 max-h-[560px] overflow-y-auto pr-1"
                            style={{ scrollbarWidth: 'thin', scrollbarColor: `${accent} transparent` }}
                        >
                            {sortedInsights.map((insight, i) => (
                                <InsightCard key={insight.id} insight={insight} index={i} />
                            ))}
                        </div>
                    </div>

                    {/* Side Panel */}
                    <div className="space-y-6">
                        {/* Trends */}
                        <div>
                            <div className="flex items-center gap-2 mb-3">
                                <TrendingUp className="w-4 h-4" style={{ color: '#16a34a' }} />
                                <span className="text-xs font-bold uppercase tracking-widest" style={{ color: textMuted }}>
                                    Weekly Trends
                                </span>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                {mockTrends.slice(0, 6).map((trend, i) => (
                                    <TrendCard key={trend.signal} trend={trend} index={i} />
                                ))}
                            </div>
                        </div>

                        {/* Correlations */}
                        <div>
                            <div className="flex items-center gap-2 mb-3">
                                <Link2 className="w-4 h-4" style={{ color: '#ea580c' }} />
                                <span className="text-xs font-bold uppercase tracking-widest" style={{ color: textMuted }}>
                                    Correlations
                                </span>
                            </div>
                            <div className="space-y-2">
                                {mockCorrelations.slice(0, 4).map((corr, i) => (
                                    <CorrelationItem key={`${corr.signal1}-${corr.signal2}`} corr={corr} index={i} />
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── Footer ── */}
                <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="text-center text-[11px] mt-10"
                    style={{ color: textMuted, opacity: 0.5 }}
                >
                    Insights are generated from pattern analysis across your health data signals
                </motion.p>
            </motion.div>
        </div>
    );
}
