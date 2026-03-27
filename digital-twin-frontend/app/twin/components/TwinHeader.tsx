'use client';

import { motion } from 'framer-motion';
import { Activity, Clock, Database, Shield, AlertTriangle } from 'lucide-react';
import { TwinSummary } from './types';

interface TwinHeaderProps {
    summary: TwinSummary;
    mode: 'mvp' | 'full';
}

function getRiskLevelColor(level: TwinSummary['riskLevel']): string {
    switch (level) {
        case 'low': return '#22c55e';
        case 'moderate': return '#eab308';
        case 'elevated': return '#f97316';
        case 'high': return '#ef4444';
    }
}

function getRiskLevelBg(level: TwinSummary['riskLevel']): string {
    switch (level) {
        case 'low': return 'rgba(34, 197, 94, 0.15)';
        case 'moderate': return 'rgba(234, 179, 8, 0.15)';
        case 'elevated': return 'rgba(249, 115, 22, 0.15)';
        case 'high': return 'rgba(239, 68, 68, 0.15)';
    }
}

export function TwinHeader({ summary, mode }: TwinHeaderProps) {
    const riskColor = getRiskLevelColor(summary.riskLevel);
    const riskBg = getRiskLevelBg(summary.riskLevel);
    const timeSinceSync = Math.round((Date.now() - summary.lastSync.getTime()) / 60000);

    return (
        <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6"
        >
            {/* Title Row */}
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h1 className="text-2xl md:text-3xl font-bold text-white/90" style={{ fontFamily: 'var(--font-oxanium)' }}>
                        Your Digital Twin
                    </h1>
                    <p className="text-sm text-white/50 mt-1">
                        Real-time health trajectory analysis
                    </p>
                </div>
                <div className="flex items-center gap-2 text-xs text-white/40">
                    <Clock className="w-3.5 h-3.5" />
                    <span>Synced {timeSinceSync}m ago</span>
                </div>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {/* Overall Risk */}
                <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="p-4 rounded-xl backdrop-blur-md border"
                    style={{
                        background: riskBg,
                        borderColor: `${riskColor}30`,
                    }}
                >
                    <div className="flex items-center gap-2 mb-1">
                        <Shield className="w-4 h-4" style={{ color: riskColor }} />
                        <span className="text-xs text-white/50 uppercase tracking-wider">Overall Risk</span>
                    </div>
                    <div className="text-2xl font-bold" style={{ color: riskColor }}>
                        {(summary.overallRisk * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs capitalize mt-0.5" style={{ color: riskColor }}>
                        {summary.riskLevel}
                    </div>
                </motion.div>

                {/* Active Agents */}
                <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="p-4 rounded-xl bg-white/5 backdrop-blur-md border border-white/10"
                >
                    <div className="flex items-center gap-2 mb-1">
                        <Activity className="w-4 h-4 text-cyan-400" />
                        <span className="text-xs text-white/50 uppercase tracking-wider">Agents</span>
                    </div>
                    <div className="text-2xl font-bold text-cyan-400">
                        {summary.activeAgents}
                    </div>
                    <div className="text-xs text-white/40 mt-0.5">
                        {mode === 'mvp' ? 'Core only' : 'All active'}
                    </div>
                </motion.div>

                {/* Data Quality */}
                <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="p-4 rounded-xl bg-white/5 backdrop-blur-md border border-white/10"
                >
                    <div className="flex items-center gap-2 mb-1">
                        <Database className="w-4 h-4 text-purple-400" />
                        <span className="text-xs text-white/50 uppercase tracking-wider">Data Quality</span>
                    </div>
                    <div className="text-2xl font-bold text-purple-400">
                        {summary.dataQuality}%
                    </div>
                    <div className="text-xs text-white/40 mt-0.5">
                        Good coverage
                    </div>
                </motion.div>

                {/* Alerts */}
                <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="p-4 rounded-xl bg-white/5 backdrop-blur-md border border-white/10"
                >
                    <div className="flex items-center gap-2 mb-1">
                        <AlertTriangle className="w-4 h-4 text-orange-400" />
                        <span className="text-xs text-white/50 uppercase tracking-wider">Alerts</span>
                    </div>
                    <div className="text-2xl font-bold text-orange-400">
                        2
                    </div>
                    <div className="text-xs text-white/40 mt-0.5">
                        Need attention
                    </div>
                </motion.div>
            </div>
        </motion.div>
    );
}

export default TwinHeader;
