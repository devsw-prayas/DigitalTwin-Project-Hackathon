// Forms/Algebra.tsx
import { DSLInput, renderDSL } from "./base";

export const Frac = ({ num, den }: { num: DSLInput; den: DSLInput }) =>
    `\\frac{${renderDSL(num)}}{${renderDSL(den)}}`;

export const Sqrt = ({ children }: { children: DSLInput | DSLInput[] }) =>
    `\\sqrt{${Array.isArray(children) ? children.map(renderDSL).join("") : renderDSL(children)}}`;

export const Pow = ({ base, exp }: { base: DSLInput; exp: DSLInput }) =>
    `{${renderDSL(base)}}^{${renderDSL(exp)}}`;

export const Sub = ({ base, sub }: { base: DSLInput; sub: DSLInput }) =>
    `{${renderDSL(base)}}_{${renderDSL(sub)}}`;

export const Abs = ({ children }: { children: DSLInput | DSLInput[] }) =>
    `\\left|${Array.isArray(children) ? children.map(renderDSL).join("") : renderDSL(children)}\\right|`;

export const Parens = ({ children }: { children: DSLInput | DSLInput[] }) =>
    `\\left(${Array.isArray(children) ? children.map(renderDSL).join("") : renderDSL(children)}\\right)`;

export const Piecewise = ({cases,}: {
    cases: { expr: DSLInput; condition: DSLInput }[];
}) =>
    `\\begin{cases} ${cases
        .map(c => `${renderDSL(c.expr)} & ${renderDSL(c.condition)}`)
        .join(" \\\\ ")} \\end{cases}`;