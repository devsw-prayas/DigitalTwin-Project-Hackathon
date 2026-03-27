// Greek letters for KaTeX/LaTeX (typed & autocomplete-friendly)
export const Greek = {
    // Lowercase
    alpha: "\\alpha",
    beta: "\\beta",
    gamma: "\\gamma",
    delta: "\\delta",
    epsilon: "\\epsilon",
    zeta: "\\zeta",
    eta: "\\eta",
    theta: "\\theta",
    iota: "\\iota",
    kappa: "\\kappa",
    lambda: "\\lambda",
    mu: "\\mu",
    nu: "\\nu",
    xi: "\\xi",
    omicron: "o",
    pi: "\\pi",
    rho: "\\rho",
    sigma: "\\sigma",
    tau: "\\tau",
    upsilon: "\\upsilon",
    phi: "\\phi",
    chi: "\\chi",
    psi: "\\psi",
    omega: "\\omega",

    // Uppercase
    Gamma: "\\Gamma",
    Delta: "\\Delta",
    Theta: "\\Theta",
    Lambda: "\\Lambda",
    Xi: "\\Xi",
    Pi: "\\Pi",
    Sigma: "\\Sigma",
    Upsilon: "\\Upsilon",
    Phi: "\\Phi",
    Psi: "\\Psi",
    Omega: "\\Omega",
} as const;

export type GreekSymbol = keyof typeof Greek;
