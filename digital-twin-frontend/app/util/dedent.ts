export function dedent(input: string, convertTabsTo = 2): string {
    if (!input) return "";
    // normalize newlines
    const raw = input.replace(/\r\n/g, "\n");
    const lines = raw.split("\n");

    // strip leading/trailing blank lines
    while (lines.length && lines[0].trim() === "") lines.shift();
    while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
    if (lines.length === 0) return "";

    // optionally convert tabs to spaces for consistent width
    const tabExpanded = lines.map(l =>
        convertTabsTo > 0 ? l.replace(/\t/g, " ".repeat(convertTabsTo)) : l
    );

    // compute minimum indent of all non-empty lines
    const indents = tabExpanded
        .filter(l => l.trim().length > 0)
        .map(l => l.match(/^ */)![0].length);

    const minIndent = indents.length ? Math.min(...indents) : 0;

    // remove common indent
    return tabExpanded.map(l => l.slice(minIndent)).join("\n");
}
