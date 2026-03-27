// Forms/Transcendentals.tsx
import { DSLInput, renderDSL } from "./base";

// Exponential
export const Exp = ({ expr }: { expr: DSLInput }) =>
    `e^{${renderDSL(expr)}}`;

// Logarithm
export const Log = ({ base, expr }: { base?: DSLInput; expr: DSLInput }) =>
    base
        ? `\\log_{${renderDSL(base)}} ${renderDSL(expr)}`
        : `\\log ${renderDSL(expr)}`;

// Natural log
export const Ln = ({ expr }: { expr: DSLInput }) =>
    `\\ln ${renderDSL(expr)}`;

// Trigonometric
export const Sin = ({ expr }: { expr: DSLInput }) =>
    `\\sin ${renderDSL(expr)}`;

export const Cos = ({ expr }: { expr: DSLInput }) =>
    `\\cos ${renderDSL(expr)}`;

export const Tan = ({ expr }: { expr: DSLInput }) =>
    `\\tan ${renderDSL(expr)}`;

export const Cot = ({ expr }: { expr: DSLInput }) =>
    `\\cot ${renderDSL(expr)}`;

export const Sec = ({ expr }: { expr: DSLInput }) =>
    `\\sec ${renderDSL(expr)}`;

export const Csc = ({ expr }: { expr: DSLInput }) =>
    `\\csc ${renderDSL(expr)}`;

// Inverse trig
export const Arcsin = ({ expr }: { expr: DSLInput }) =>
    `\\arcsin ${renderDSL(expr)}`;

export const Arccos = ({ expr }: { expr: DSLInput }) =>
    `\\arccos ${renderDSL(expr)}`;

export const Arctan = ({ expr }: { expr: DSLInput }) =>
    `\\arctan ${renderDSL(expr)}`;

// Hyperbolic
export const Sinh = ({ expr }: { expr: DSLInput }) =>
    `\\sinh ${renderDSL(expr)}`;

export const Cosh = ({ expr }: { expr: DSLInput }) =>
    `\\cosh ${renderDSL(expr)}`;

export const Tanh = ({ expr }: { expr: DSLInput }) =>
    `\\tanh ${renderDSL(expr)}`;
