import { DSLInput, renderDSL } from "./base";

export const Derivative = ({
                               expr,
                               variable,
                           }: { expr: DSLInput; variable: string }) =>
    `\\frac{d}{d${variable}} ${renderDSL(expr)}`;

export const Partial = ({
                            expr,
                            variable,
                        }: { expr: DSLInput; variable: string }) =>
    `\\frac{\\partial}{\\partial ${variable}} ${renderDSL(expr)}`;

export const Quadrature = ({
                               expr,
                               variable,
                               lower,
                               upper,
                           }: { expr: DSLInput; variable: string; lower?: DSLInput; upper?: DSLInput }) =>
    lower && upper
        ? `\\int_{${renderDSL(lower)}}^{${renderDSL(upper)}} ${renderDSL(expr)}\\, d${variable}`
        : `\\int ${renderDSL(expr)}\\, d${variable}`;

export const Sum = ({
                        expr,
                        index,
                        lower,
                        upper,
                    }: { expr: DSLInput; index: string; lower: DSLInput; upper: DSLInput }) =>
    `\\sum_{${index}=${renderDSL(lower)}}^{${renderDSL(upper)}} ${renderDSL(expr)}`;


export const Product = ({
                            expr,
                            index,
                            lower,
                            upper,
                        }: { expr: DSLInput; index: string; lower: DSLInput; upper: DSLInput }) =>
    `\\prod_{${index}=${renderDSL(lower)}}^{${renderDSL(upper)}} ${renderDSL(expr)}`;

export const Limit = ({
                          variable,
                          to,
                          expr,
                      }: { variable: string; to: DSLInput; expr: DSLInput }) =>
    `\\lim_{${variable} \\to ${renderDSL(to)}} ${renderDSL(expr)}`;

export const Gradient = ({ expr }: { expr: DSLInput }) =>
    `\\nabla ${renderDSL(expr)}`;

export const Divergence = ({ expr }: { expr: DSLInput }) =>
    `\\nabla \\cdot ${renderDSL(expr)}`;

export const Curl = ({ expr }: { expr: DSLInput }) =>
    `\\nabla \\times ${renderDSL(expr)}`;