import {isValidElement, ReactNode} from "react";
export type DSLInput = string | ReactNode;

export const renderDSL = (input?: DSLInput): string => {
    if (!input) return "";

    if (typeof input === "string") {
        return input;
    }

    if (Array.isArray(input)) {
        return input.map(renderDSL).join("");
    }

    if (isValidElement(input)) {
        // input.type is the function component (e.g., Frac, Quadrature, etc.)
        // call it manually with its props to get back the LaTeX string
        const { type, props } = input;
        if (typeof type === "function") {
            return (type as (props: unknown) => string)(props);
        }
    }

    // fallback (shouldn’t hit if DSL is correct)
    return String(input);
};
