"use client"

import Typography from "@/app/typography";
import {BlockMath, InlineMath} from "react-katex";
import {Frac, Pow} from "@/app/math/algebra";
import {Quadrature} from "@/app/math/calculus";
import {Greek} from "@/app/math/greek";
import {Infinity} from "@/app/math/symbols";

export default function TestPage(){
    return (
        <div className="min-h-screen bg-background text-text-primary font-body">
            {/* Header */}
            <header className="border-b border-border bg-surface px-6 py-4">
    <Typography.Text variant={"h1"} className={"text-accent-cyan"}>
        StormWeaver Studios
    </Typography.Text>
    <Typography.Text variant={"p"} className={"text-text-muted"}>
        Work in progress
        </Typography.Text>
        </header>

    {/* Main Content */}
    <main className="p-8 space-y-8">
    {/* Project Card */}
    <section>
    <h2 className="text-2xl font-heading text-accent-violet mb-4">
        Projects
        </h2>
        <div className="bg-surface rounded-lg p-4 shadow-lg">
    <h3 className="text-xl font-heading text-accent-cyan">
        Spectra Renderer
    </h3>
    <p className="text-text-muted">
        Spectral path tracer with neural acceleration.
    </p>
    </div>
    </section>

    {/* Code Example */}
    <section>
        <h2 className="text-2xl font-heading text-accent-violet mb-2">
        Code Example
    </h2>
    <div className="bg-code-bg rounded-lg p-4 font-mono text-sm">
    <span className="text-[var(--color-code-keyword)]">static_cast</span>
        &lt;Derived*&gt;(this)-&gt;method(); <br />
    <span className="text-[var(--color-code-comment)]"></span>
        </div>
        </section>

    {/* Math Example */}
    <section>
        <h2 className="text-2xl font-heading text-accent-violet mb-2">
        Math Example
    </h2>
    <p className="text-text-muted">
        Inline math: <InlineMath>{"E = mc^2"}</InlineMath>
    </p>
    <BlockMath>
    {"\\int_\\Omega f_r(p, \\omega_i, \\omega_o) L_i(p, \\omega_i) (n \\cdot \\omega_i) d\\omega_i"}
    </BlockMath>
    <p className="text-xs text-text-muted text-center">
        Equation 1.1 — Spectral Rendering Integral
    </p>
    </section>

    <section>
    <Typography.Math>
        <Frac num="1" den="x" />
        </Typography.Math>

        <Typography.Math block>
    <Quadrature
        expr={
        <Frac
    num={<Pow base={"e"} exp={"x"}></Pow>}
    den={<Pow base={"1+x"} exp={"2"} />}
    />
}
    variable={Greek.Pi}
    lower={<Infinity sign={'-'}/>}
    upper={<Infinity sign={"+"} />}
    />
    </Typography.Math>

    <Typography.Math className={"text-4xl "}>
    <Quadrature expr={<Frac num="1" den="x" />} variable="x" lower="0" upper="1" />
        </Typography.Math>

        <Typography.Text variant="p">
        To call the function, use <Typography.Code>myFunction()</Typography.Code> in your code.
    </Typography.Text>

    <Typography.Code block language="typescript" className={"text-4xl justify-items-center"}>
        {`
function greet(name) {
  if (name) {
    console.log("Hello, " + name);
  }
}
                        `}
    </Typography.Code>
    </section>
    </main>

    {/* Footer */}
    <footer className="border-t border-border bg-surface px-6 py-3 text-center text-text-muted text-sm">
                © 2025 StormWeaver Studios — Prayas Bharadwaj
    </footer>
    </div>
);
}