'use client';

import { motion } from 'framer-motion';
import { AlertTriangle, Info, CheckCircle, ArrowRight, Lightbulb, Clock } from 'lucide-react';
import Link from 'next/link';
import { HealthInsight } from './types';

interface InsightCardProps {
    insight: HealthInsight;
    index: number;
}

function getTypeIcon(type: HealthInsight['type']) {
    switch (type) {
        case 'warning':
            return <AlertTriangle className="w-4 h-4" />;
        case 'success':
            return <CheckCircle className="w-4 h-4" />;
        case 'action':
            return <Lightbulb className="w-4 h-4" />;
        default:
            return <Info className="w-4 h-4" />;
    }
}

function getTypeColor(type: HealthInsight['type']) {
    switch (type) {
        case 'warning':
            return { bg: 'rgba(239, 68, 68, 0.15)', border: 'rgba(239, 68, 68, 0.3)', text: '#ef4444' };
        case 'success':
            return { bg: 'rgba(34, 197, 94, 0.15)', border: 'rgba(34, 197, 94, 0.3)', text: '#22c55e' };
        case 'action':
            return { bg: 'rgba(168, 85, 247, 0.15)', border: 'rgba(168, 85, 247, 0.3)', text: '#a855f7' };
        default:
            return { bg: 'rgba(59, 130, 246, 0.15)', border: 'rgba(59, 130, 246, 0.3)', text: '#3b82f6' };
    }
}

function formatTimeAgo(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.round(diffMs / 60000);
    const diffHours = Math.round(diffMs / 3600000);
    const diffDays = Math.round(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
}

export function InsightCard({ insight, index }: InsightCardProps) {
    const colors = getTypeColor(insight.type);

    return (
        <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="p-4 rounded-xl backdrop-blur-md border"
            style={{ background: colors.bg, borderColor: colors.border }}
        >
            <div className="flex items-start gap-3">
                {/* Icon */}
                <div
                    className="p-2 rounded-lg shrink-0"
                    style={{ background: `${colors.text}20` }}
                >
                    <div style={{ color: colors.text }}>
                        {getTypeIcon(insight.type)}
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-white/90">{insight.title}</h3>
                        {insight.agentColor && (
                            <div
                                className="w-2 h-2 rounded-full shrink-0"
                                style={{ background: insight.agentColor }}
                            />
                        )}
                    </div>
                    <p className="text-sm text-white/60 leading-relaxed mb-2">
                        {insight.description}
                    </p>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 text-xs text-white/40">
                            <Clock className="w-3 h-3" />
                            {formatTimeAgo(insight.timestamp)}
                        </div>
                        {insight.actionable && insight.action && (
                            <Link
                                href={insight.action.href}
                                className="flex items-center gap-1 text-xs font-medium transition-colors"
                                style={{ color: colors.text }}
                            >
                                {insight.action.label}
                                <ArrowRight className="w-3 h-3" />
                            </Link>
                        )}
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

export default InsightCard;
