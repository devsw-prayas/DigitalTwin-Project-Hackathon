// Forms/Relations.tsx
import { DSLInput, renderDSL } from "./base";

export const Eq = ({ left, right }: { left: DSLInput; right: DSLInput }) =>
    `${renderDSL(left)} = ${renderDSL(right)}`;

export const Neq = ({ left, right }: { left: DSLInput; right: DSLInput }) =>
    `${renderDSL(left)} \\neq ${renderDSL(right)}`;

export const Lt = ({ left, right }: { left: DSLInput; right: DSLInput }) =>
    `${renderDSL(left)} < ${renderDSL(right)}`;

export const Gt = ({ left, right }: { left: DSLInput; right: DSLInput }) =>
    `${renderDSL(left)} > ${renderDSL(right)}`;

export const Leq = ({ left, right }: { left: DSLInput; right: DSLInput }) =>
    `${renderDSL(left)} \\leq ${renderDSL(right)}`;

export const Geq = ({ left, right }: { left: DSLInput; right: DSLInput }) =>
    `${renderDSL(left)} \\geq ${renderDSL(right)}`;
