"use client";

import { ReactNode, useEffect, useRef } from "react";
import React from "react";
import katex from "katex";
import clsx from "clsx";
import { DSLInput, renderDSL } from "@/app/math/base";
import { Highlight } from "prism-react-renderer";
import type { PrismTheme } from "prism-react-renderer";
import { dedent } from "@/app/util/dedent";
import { orbitron, oxanium, spaceGrotesk, exo2 } from "@/app/system/font/fonts";

const themeAwareCode: PrismTheme = {
    plain: {
        backgroundColor: "rgba(10, 10, 15, 0.85)",
        color: "#e8e8f0",
    },
    styles: [
        { types: ["comment", "prolog", "doctype", "cdata"], style: { color: "#7a7a9a", fontStyle: "italic" } },
        { types: ["keyword", "operator", "important", "atrule"], style: { color: "var(--accent)" } },
        { types: ["function", "class-name", "builtin"], style: { color: "var(--accent)", fontWeight: "600" } },
        { types: ["string", "char", "attr-value", "regex"], style: { color: "#a8d8a8" } },
        { types: ["number", "boolean", "constant"], style: { color: "#f0c080" } },
        { types: ["punctuation", "symbol"], style: { color: "#9090b0" } },
        { types: ["tag", "selector", "attr-name"], style: { color: "#c0c8e8" } },
        { types: ["variable", "parameter"], style: { color: "#d0c0f0", fontStyle: "italic" } },
        { types: ["deleted"], style: { color: "#e08080" } },
        { types: ["inserted"], style: { color: "#a8d8a8" } },
    ],
};


type TextProps = {
    variant?: "h1" | "h2" | "h3" | "p";
    children: ReactNode;
    className?: string;
    u?: boolean;
    i?: boolean;
    b?: boolean;
};

const Text = ({ variant = "p", children, className, u, i, b }: TextProps) => {
    const Tag = variant;

    const baseClasses: Record<NonNullable<TextProps["variant"]>, string> = {
        h1: "text-4xl font-bold leading-tight",
        h2: "text-3xl font-semibold leading-snug",
        h3: "text-2xl font-medium leading-snug",
        p: "text-base leading-relaxed",
    };

    const colorStyle: Record<NonNullable<TextProps["variant"]>, React.CSSProperties> = {
        h1: { color: "var(--textPrimary)" },
        h2: { color: "var(--textPrimary)" },
        h3: { color: "var(--textPrimary)" },
        p:  { color: "var(--textMuted)", fontWeight: 500 },
    };

    const fontClass: Record<NonNullable<TextProps["variant"]>, string> = {
        h1: exo2.className,
        h2: exo2.className,
        h3: exo2.className,
        p:  spaceGrotesk.className,
    };

    return (
        <Tag
            className={clsx(
                baseClasses[variant],
                fontClass[variant],
                u && "underline",
                i && "italic",
                b && "font-bold",
                className
            )}
            style={colorStyle[variant]}
        >
            {children}
        </Tag>
    );
};


type MathProps = {
    children: DSLInput | DSLInput[];
    className?: string;
    block?: boolean;
};

const Math = ({ children, className, block = false }: MathProps) => {
    const ref = useRef<HTMLSpanElement | null>(null);

    useEffect(() => {
        if (ref.current) {
            try {
                ref.current.innerHTML = ""; // clear old KaTeX render
                const latex = Array.isArray(children)
                    ? children.map(renderDSL).join("")
                    : renderDSL(children);

                katex.render(latex, ref.current, {
                    throwOnError: false,
                    displayMode: block,
                });
            } catch (err) {
                console.error("KaTeX render error:", err);
            }
        }
    }, [children, block]);

    return (
        <span className={clsx("inline-block align-middle", className)}>
      <span ref={ref} />
    </span>
    );
};


type CodeProps = {
    children: string;
    className?: string;
    block?: boolean;
    language?: string;
    showLineNumbers?: boolean;
    tabWidth?: number;
};

const Code = ({
                  children,
                  className,
                  block,
                  language = "tsx",
                  showLineNumbers = false,
                  tabWidth = 2,
              }: CodeProps) => {
    const isBlock =
        block ?? children.includes("\n");

    if (!isBlock) {
        return (
            <code
                className={clsx(oxanium.className, "px-1.5 py-0.5 rounded text-sm", className)}
                style={{
                    background: "rgba(10, 10, 15, 0.75)",
                    color: "#e8e8f0",
                    border: "1px solid color-mix(in srgb, var(--accent) 25%, transparent)",
                }}
            >
                {String(children).trim()}
            </code>
        );
    }

    const normalized = dedent(String(children), tabWidth);

    return (
        <Highlight code={normalized} language={language as never} theme={themeAwareCode}>
            {({ className: highlightClass, style, tokens, getLineProps, getTokenProps }) => (
                <div
                    className={clsx("relative rounded-md", className)}
                    style={{
                        border: "1px solid color-mix(in srgb, var(--accent) 20%, transparent)",
                        borderRadius: "8px",
                        backdropFilter: "blur(12px)",
                        WebkitBackdropFilter: "blur(12px)",
                        overflow: "hidden",
                    }}
                >
                    <button
                        onClick={() => navigator.clipboard.writeText(normalized)}
                        className="absolute top-2 right-2 text-xs text-gray-400 hover:text-gray-200"
                    >
                        Copy
                    </button>

                    <pre
                        className={clsx(
                            "p-3 rounded-md overflow-x-auto",
                            highlightClass
                        )}
                        style={style}
                    >
        {tokens.map((line, i) => {
            const { key, ...lineProps } = getLineProps({ line }); // destructure out key

            return (
                <div key={i} {...lineProps} className="flex">
                    {showLineNumbers && (
                        <span className="inline-block w-10 pr-4 text-right select-none text-xs text-gray-500">{i + 1}
        </span>)}
                    <div className="flex-1">
                        {line.map((token, j) => {
                            const { key: _tk, ...tokenProps } = getTokenProps({ token });
                            return (
                                <span
                                    key={`${i}-${j}-${token.content}`}
                                    {...tokenProps}
                                />
                            );
                        })}
                    </div>
                </div>
            );
        })}
                    </pre>
                </div>
            )}
        </Highlight>
    );
};


const Typography = {
    Text,
    Math,
    Code,
};
export default Typography;