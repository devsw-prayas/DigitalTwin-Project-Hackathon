import "./globals.css"
import {ReactNode} from "react"
import {Metadata} from "next";
import {orbitron, oxanium, spaceGrotesk, exo2} from '@/app/system/font/fonts';
import {ThemeProvider} from "@/app/system/theme/theme-context";
import {SubsystemProvider} from "@/app/system/theme/subsystem-context";
import {Background} from "@/app/system/theme/background";
import NavBar from "@/app/component/navbar";
import {AgentModeProvider} from "@/app/agents/agent-mode-context";

export const metadata: Metadata = {
    metadataBase:  new URL('https://digi-twin.com'),
    title: {
        default: "Digital-Twin",
        template: "%s | Digital-Twin",
    },
    description: "Your digital twin to your health to keep you safe",
    abstract: "The home of digital twin",
    alternates: {
        canonical: "https://digi-twin.com",
        languages: {
            "en-US": "https://digi-twin.com/en-US",
        },
    },
    authors: [
        {
            name: "Prayas Bharadwaj",
            url: "https://digi-twin.com",
        },
    ],
    openGraph: {
        title: "Digital-Twin",
        description: "Your digital twin to your health to keep you safe",
        url: "https://digi-twin.com",
        siteName: "Digital-Twin",
        images: [
            {
                url: "/og-image.png",
                width: 1200,
                height: 630,
                alt: "Digital-Twin",
            },
        ],
        locale: "en_US",
        type: "website",
    },
    twitter: {
        card: "summary_large_image",
        title: "Digital-Twin",
        description: "Your digital twin to your health to keep you safe.",
        images: ["/og-image.png"],
    },
    robots: {
        index: true,
        follow: true,
        googleBot: {
            index: true,
            follow: true,
        },
    },
    icons: {
        icon: "/favicon.ico",
        shortcut: "/favicon-16x16.png",
        apple: "/apple-touch-icon.png",
    },
    manifest: "/site.webmanifest",
};

export default function RootLayout({children} :  {children: ReactNode}) {
    return (
        <html lang="en" className={`${oxanium.variable} ${spaceGrotesk.variable} ${orbitron.variable} ${exo2.variable}`}>

        <body style={{ display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
        <ThemeProvider>
            <SubsystemProvider>
                <AgentModeProvider>
                    <Background/>
                    <NavBar/>
                    <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", paddingTop: "72px" }}>
                        {children}
                    </div>
                </AgentModeProvider>
            </SubsystemProvider>
        </ThemeProvider>
        </body>
        </html>
    )
}