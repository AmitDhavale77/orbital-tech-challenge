import { defaultRemarkPlugins } from "streamdown";
import type { PluggableList } from "unified";

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

/** The remark attacher. `count` is taken via an options object (not closed over)
 * and the function is named at module scope ON PURPOSE: streamdown memoizes the
 * compiled remark pipeline in a module-level cache keyed by each plugin's
 * function `name` + JSON-stringified options. An anonymous `() => (tree) => …`
 * closure has an empty name and no options, so EVERY `count` collides on one
 * cache entry — whichever `count` renders first is frozen for the rest of the
 * session. (A conversation that opens with an uncited turn — a greeting,
 * "Not specified", or an empty bundle — would seed `count = 0` and then render
 * every later cited answer's `[n]` markers as plain text.) Naming it and moving
 * `count` into options makes `citationLinkPlugin:{"count":2}` distinct from
 * `:{"count":0}`, so the cache keys per count instead of colliding. */
function citationLinkPlugin(options: { count: number }) {
	return (tree: MdNode) => transform(tree, options.count);
}

/** Streamdown's GFM defaults plus an AST plugin that turns `[n]` markers into
 * `#cite-n` links. Passing `remarkPlugins` replaces streamdown's defaults, so we
 * merge them back in. `count` bounds which markers convert (markers beyond the
 * verified citations stay literal); omit it — while streaming, citations aren't
 * known yet — to convert every `[n]` to a (placeholder) marker. */
export function citationRemarkPlugins(
	count: number = Number.POSITIVE_INFINITY,
): PluggableList {
	return [
		...Object.values(defaultRemarkPlugins),
		[citationLinkPlugin, { count }],
	];
}
