import { defaultRemarkPlugins } from "streamdown";
import type { Pluggable, PluggableList } from "unified";

/** Minimal mdast shape we touch — avoids depending on @types/mdast. */
interface MdNode {
	type: string;
	value?: string;
	url?: string;
	children?: MdNode[];
}

const MARKER = /\[(\d+)\]/g;

/** 0-based indices of citations referenced by an inline `[n]` marker. Used only
 * to decide which verified citations need the trailing "Sources" fallback. */
export function referencedCitationIndices(
	markdown: string,
	count: number,
): Set<number> {
	const refs = new Set<number>();
	for (const match of markdown.matchAll(MARKER)) {
		const n = Number(match[1]);
		if (n >= 1 && n <= count) refs.add(n - 1);
	}
	return refs;
}

function splitMarkers(value: string, count: number): MdNode[] {
	const out: MdNode[] = [];
	let last = 0;
	for (const match of value.matchAll(MARKER)) {
		const n = Number(match[1]);
		const at = match.index ?? 0;
		if (n < 1 || n > count) continue;
		if (at > last) out.push({ type: "text", value: value.slice(last, at) });
		// In-page anchor href — the safe fragment form streamdown's hardening
		// allows; the `a` component override turns it into a citation marker.
		out.push({
			type: "link",
			url: `#cite-${n}`,
			children: [{ type: "text", value: String(n) }],
		});
		last = at + match[0].length;
	}
	if (out.length === 0) return [{ type: "text", value }];
	if (last < value.length) out.push({ type: "text", value: value.slice(last) });
	return out;
}

function transform(node: MdNode, count: number): void {
	if (!node.children) return;
	const next: MdNode[] = [];
	for (const child of node.children) {
		if (child.type === "text" && typeof child.value === "string") {
			next.push(...splitMarkers(child.value, count));
		} else {
			// Never rewrite markers inside code — only prose text nodes.
			if (child.type !== "code" && child.type !== "inlineCode") {
				transform(child, count);
			}
			next.push(child);
		}
	}
	node.children = next;
}

function citationLinkPlugin(count: number): Pluggable {
	return () => (tree: MdNode) => transform(tree, count);
}

/** Streamdown's GFM defaults plus an AST plugin that turns `[n]` markers into
 * `#cite-n` links. Passing `remarkPlugins` replaces streamdown's defaults, so we
 * merge them back in. */
export function citationRemarkPlugins(count: number): PluggableList {
	return [...Object.values(defaultRemarkPlugins), citationLinkPlugin(count)];
}
