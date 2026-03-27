'use client';

export const PageContainer = ({ children }: { children: React.ReactNode }) => (
    <div className="w-full h-full overflow-y-auto">
        <div className="mx-auto max-w-7xl px-6 md:px-10 py-10 space-y-10">
            {children}
        </div>
    </div>
);

export const GlassCard = ({
                              children,
                              className = "",
                          }: {
    children: React.ReactNode;
    className?: string;
}) => (
    <div
        className={`rounded-2xl p-5 border backdrop-blur-xl transition-all ${className}`}
        style={{
            background: "var(--surface)",
            borderColor: "color-mix(in srgb, var(--accent) 20%, transparent)",
            boxShadow: "0 10px 40px rgba(0,0,0,0.25)",
        }}
    >
        {children}
    </div>
);

export const SectionHeader = ({
                                  title,
                                  subtitle,
                              }: {
    title: string;
    subtitle?: string;
}) => (
    <div className="space-y-1">
        <h2 className="text-2xl font-semibold" style={{ color: "var(--textPrimary)" }}>
            {title}
        </h2>
        {subtitle && (
            <p className="text-sm" style={{ color: "var(--textMuted)" }}>
                {subtitle}
            </p>
        )}
    </div>
);