'use client';

import { motion } from 'framer-motion';
import { WhatIfScenario } from './types';

interface ScenarioSliderProps {
    scenario: WhatIfScenario;
    value: number;
    onChange: (value: number) => void;
    index: number;
}

export function ScenarioSlider({ scenario, value, onChange, index }: ScenarioSliderProps) {
    const progress = ((value - scenario.minValue) / (scenario.maxValue - scenario.minValue)) * 100;

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="p-4 rounded-xl bg-white/5 backdrop-blur-md border border-white/10"
        >
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <span className="text-lg">{scenario.icon}</span>
                    <div>
                        <h3 className="text-sm font-medium text-white/90">{scenario.name}</h3>
                        <p className="text-xs text-white/40">{scenario.description}</p>
                    </div>
                </div>
                <div className="text-right">
                    <span className="text-lg font-bold text-cyan-400">{value}</span>
                    <span className="text-xs text-white/40 ml-1">{scenario.unit}</span>
                </div>
            </div>

            {/* Custom slider */}
            <div className="relative">
                <input
                    type="range"
                    min={scenario.minValue}
                    max={scenario.maxValue}
                    step={scenario.step}
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-800 rounded-full appearance-none cursor-pointer"
                    style={{
                        background: `linear-gradient(to right, #38bdf8 0%, #38bdf8 ${progress}%, #1f2937 ${progress}%, #1f2937 100%)`,
                    }}
                />
                <div className="flex justify-between mt-1">
                    <span className="text-xs text-white/30">{scenario.minValue} {scenario.unit}</span>
                    <span className="text-xs text-white/30">{scenario.maxValue} {scenario.unit}</span>
                </div>
            </div>
        </motion.div>
    );
}

export default ScenarioSlider;
