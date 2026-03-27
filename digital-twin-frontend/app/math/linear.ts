// Forms/LinearAlgebra.tsx
import { DSLInput, renderDSL } from "./base";

export const Vector = ({ children }: { children: DSLInput | DSLInput[] }) =>
    `\\begin{bmatrix} ${Array.isArray(children) ? children.map(renderDSL).join(" \\\\ ") : renderDSL(children)} \\end{bmatrix}`;

export const Matrix = ({ rows }: { rows: DSLInput[][] }) =>
    `\\begin{bmatrix} ${rows
        .map(r => r.map(renderDSL).join(" & "))
        .join(" \\\\ ")} \\end{bmatrix}`;

export const Det = ({ children }: { children: DSLInput }) =>
    `\\det\\left(${renderDSL(children)}\\right)`;

export const Transpose = ({ children }: { children: DSLInput }) =>
    `{${renderDSL(children)}}^{T}`;
