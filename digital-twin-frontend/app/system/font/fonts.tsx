import { Orbitron, Oxanium, Space_Grotesk, Exo_2 } from "next/font/google";

export const orbitron = Orbitron({
    subsets: ["latin"],
    variable: "--font-orbitron",
    weight: ["400", "500", "600", "700"],
    display: "swap",
});

export const oxanium = Oxanium({
    subsets: ["latin"],
    variable: "--font-oxanium", // ✅ fixed
    weight: ["400", "600", "700"],
    display: "swap",
});

export const spaceGrotesk = Space_Grotesk({
    subsets: ["latin"],
    variable: "--font-space-grotesk",
    weight: ["300", "400", "500", "700"],
    display: "swap",
});

export const exo2 = Exo_2({
    subsets: ["latin"],
    variable: "--font-exo2",
    weight: ["300", "400", "500", "600", "700", "800"],
    display: "swap",
});