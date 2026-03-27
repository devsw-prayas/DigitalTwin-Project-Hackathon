import type { Config } from 'tailwindcss'

const config: Config = {
    content: [
        './app/**/*.{ts,tsx}',
        './components/**/*.{ts,tsx}',
        './app/system/**/*.{ts,tsx}',
    ],
    theme: {
        extend: {
            fontFamily: {
                body: ['var(--font-space-grotesk)'],
                heading: ['var(--font-orbitron)'],
                system: ['var(--font-oxanium)'],
            },
        },
    },
}

export default config
