// Core math symbols, operators, and constants
export const Symbols = {
    // Operators
    equals: "=",
    plus: "+",
    minus: "-",
    times: "\\times",
    divide: "\\div",
    cdot: "\\cdot",

    // Relations
    lt: "<",
    gt: ">",
    leq: "\\leq",
    geq: "\\geq",
    neq: "\\neq",

    // Set theory
    in: "\\in",
    notin: "\\notin",
    subset: "\\subset",
    subseteq: "\\subseteq",
    superset: "\\supset",
    supeseteq: "\\supseteq",
    emptyset: "\\emptyset",

    // Logic
    forall: "\\forall",
    exists: "\\exists",
    neg: "\\neg",
    wedge: "\\wedge",
    vee: "\\vee",
    implies: "\\implies",
    iff: "\\iff",

    // Calculus / operators
    integral: "\\int",
    doubleIntegral: "\\iint",
    tripleIntegral: "\\iiint",
    sum: "\\sum",
    prod: "\\prod",
    partial: "\\partial",
    nabla: "\\nabla",
    lim: "\\lim",

    // Constants
    pi: "\\pi",
    infinity: "\\infty",
    e: "e",
    hbar: "\\hbar",

    // Special
    approx: "\\approx",
    proportional: "\\propto",
    degree: "^\\circ",
} as const;


export const Infinity = ({ sign }: { sign?: "+" | "-" }) => {
    if (sign === "+") return `+\\infty`;
    if (sign === "-") return `-\\infty`;
    return `\\infty`;
};

export type SymbolName = keyof typeof Symbols;
