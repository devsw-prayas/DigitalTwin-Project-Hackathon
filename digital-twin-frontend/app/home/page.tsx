import Hero from "@/app/home/components/landing";


export default function LandingPage() {
    return (
        <main className="w-full flex flex-col" style={{ overflowY: "auto", flex: 1 }}>
            <Hero />
        </main>
    );
}