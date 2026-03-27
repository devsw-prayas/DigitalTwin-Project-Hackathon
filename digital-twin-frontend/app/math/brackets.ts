import { DSLInput, renderDSL } from "./base";

export const Floor = ({ expr }: { expr: DSLInput }) =>
    `\\lfloor ${renderDSL(expr)} \\rfloor`;

export const Ceil = ({ expr }: { expr: DSLInput }) =>
    `\\lceil ${renderDSL(expr)} \\rceil`;

export const Overline = ({ expr }: { expr: DSLInput }) =>
    `\\overline{${renderDSL(expr)}}`;

export const Underline = ({ expr }: { expr: DSLInput }) =>
    `\\underline{${renderDSL(expr)}}`;
