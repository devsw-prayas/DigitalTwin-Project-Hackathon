'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AgentRiskData, TrajectoryPoint } from './types';
import { AgentCard } from './AgentCard';
import { getActiveAgents, AgentMode } from '@/lib/agents';

interface AgentGridProps {
    agents: AgentRiskData[];
    trajectories: Record<string, TrajectoryPoint[]>;
    mode: AgentMode;
}

export function AgentGrid({ agents, trajectories, mode }: AgentGridProps) {
    const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
    const activeAgents = getActiveAgents(mode);
    const activeAgentIds = new Set(activeAgents.map(a => a.id));

    // Filter agents based on mode
    const visibleAgents = agents.filter(a => activeAgentIds.has(a.id as any));
    const gridCols = mode === 'mvp' ? 2 : 4;

    return (
        <div className="space-y-4">
            {/* Agent Cards Grid */}
            <div
                className="grid gap-4"
                style={{
                    gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))`,
                }}
            >
                {visibleAgents.map((agent, index) => (
                    <motion.div
                        key={agent.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.05 }}
                    >
                        <AgentCard
                            agent={agent}
                            trajectory={trajectories[agent.id]}
                            isExpanded={selectedAgent === agent.id}
                            onClick={() => setSelectedAgent(
                                selectedAgent === agent.id ? null : agent.id
                            )}
                        />
                    </motion.div>
                ))}
            </div>

            {/* Mode indicator */}
            {mode === 'full' && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.4 }}
                    className="text-center text-xs text-white/30"
                >
                    Showing all 8 agents • Click a card to expand
                </motion.div>
            )}
        </div>
    );
}

export default AgentGrid;
