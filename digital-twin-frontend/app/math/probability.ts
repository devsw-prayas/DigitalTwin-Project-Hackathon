// Forms/Probability.tsx
import { DSLInput, renderDSL } from "./base";

export const Expectation = ({ children }: { children: DSLInput | DSLInput[] }) =>
    `\\mathbb{E}\\left[${Array.isArray(children) ? children.map(renderDSL).join("") : renderDSL(children)}\\right]`;

export const Variance = ({ children }: { children: DSLInput | DSLInput[] }) =>
    `\\mathrm{Var}\\left[${Array.isArray(children) ? children.map(renderDSL).join("") : renderDSL(children)}\\right]`;

export const Probability = ({ children }: { children: DSLInput | DSLInput[] }) =>
    `\\mathbb{P}\\left(${Array.isArray(children) ? children.map(renderDSL).join("") : renderDSL(children)}\\right)`;

export const Distribution = ({ name, params }: { name: string; params: DSLInput[] }) =>
    `${name}\\left(${params.map(renderDSL).join(", ")}\\right)`;
