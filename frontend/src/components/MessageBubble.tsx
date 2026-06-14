import { motion } from "framer-motion";
import { Bot } from "lucide-react";
import { type Components, Streamdown } from "streamdown";
import "streamdown/styles.css";
import {
	citationRemarkPlugins,
	referencedCitationIndices,
} from "../lib/citation-markers";
import type { Citation, Message } from "../types";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";

/** One inline, clickable reference marker rendered in place of a `[n]` token. */
function CitationRef({
	index,
	citation,
	onCitationClick,
}: {
	index: number;
	citation: Citation;
	onCitationClick?: (citation: Citation) => void;
}) {
	return (
		<Tooltip>
			<TooltipTrigger asChild>
				<button
					type="button"
					aria-label={`${citation.document_name} — page ${citation.page}`}
					onClick={() => onCitationClick?.(citation)}
					className="mx-0.5 inline-flex items-center rounded bg-neutral-900 px-1 align-super text-[0.65rem] font-semibold leading-tight text-white no-underline transition-colors hover:bg-neutral-700"
				>
					{index + 1}
				</button>
			</TooltipTrigger>
			<TooltipContent className="max-w-xs">
				<div className="font-medium">
					{citation.document_name} · p.{citation.page}
				</div>
				<div className="mt-0.5 italic text-neutral-300">“{citation.quote}”</div>
			</TooltipContent>
		</Tooltip>
	);
}

function AssistantAnswer({
	message,
	onCitationClick,
}: {
	message: Message;
	onCitationClick?: (citation: Citation) => void;
}) {
	const { citations } = message;
	const referenced = referencedCitationIndices(
		message.content,
		citations.length,
	);
	const orphans = citations.filter((_, i) => !referenced.has(i));

	const components: Components = {
		a: ({ href, children }) => {
			const match = /^#cite-(\d+)$/.exec(href ?? "");
			if (match) {
				const idx = Number(match[1]) - 1;
				const citation = citations[idx];
				return citation ? (
					<CitationRef
						index={idx}
						citation={citation}
						onCitationClick={onCitationClick}
					/>
				) : (
					<>{children}</>
				);
			}
			return (
				<a href={href} target="_blank" rel="noreferrer">
					{children}
				</a>
			);
		},
	};

	return (
		<>
			<div className="prose">
				<Streamdown
					components={components}
					remarkPlugins={citationRemarkPlugins(citations.length)}
				>
					{message.content}
				</Streamdown>
			</div>
			{orphans.length > 0 && (
				<div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs text-neutral-400">
					<span>Sources:</span>
					{orphans.map((citation) => (
						<CitationRef
							key={`${citation.document_id}-${citation.page}`}
							index={citations.indexOf(citation)}
							citation={citation}
							onCitationClick={onCitationClick}
						/>
					))}
				</div>
			)}
		</>
	);
}

interface MessageBubbleProps {
	message: Message;
	onCitationClick?: (citation: Citation) => void;
}

export function MessageBubble({
	message,
	onCitationClick,
}: MessageBubbleProps) {
	if (message.role === "system") {
		return (
			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ duration: 0.2 }}
				className="flex justify-center py-2"
			>
				<p className="text-xs text-neutral-400">{message.content}</p>
			</motion.div>
		);
	}

	if (message.role === "user") {
		return (
			<motion.div
				initial={{ opacity: 0, y: 8 }}
				animate={{ opacity: 1, y: 0 }}
				transition={{ duration: 0.2 }}
				className="flex justify-end py-1.5"
			>
				<div className="max-w-[75%] rounded-2xl rounded-br-md bg-neutral-100 px-4 py-2.5">
					<p className="whitespace-pre-wrap text-sm text-neutral-800">
						{message.content}
					</p>
				</div>
			</motion.div>
		);
	}

	// Assistant message
	return (
		<motion.div
			initial={{ opacity: 0, y: 8 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.2 }}
			className="flex gap-3 py-1.5"
		>
			<div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-neutral-900">
				<Bot className="h-4 w-4 text-white" />
			</div>
			<div className="min-w-0 max-w-[80%]">
				<AssistantAnswer message={message} onCitationClick={onCitationClick} />
			</div>
		</motion.div>
	);
}

interface StreamingBubbleProps {
	content: string;
}

export function StreamingBubble({ content }: StreamingBubbleProps) {
	return (
		<div className="flex gap-3 py-1.5">
			<div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-neutral-900">
				<Bot className="h-4 w-4 text-white" />
			</div>
			<div className="min-w-0 max-w-[80%]">
				{content ? (
					<div className="prose">
						<Streamdown mode="streaming">{content}</Streamdown>
					</div>
				) : (
					<div className="flex items-center gap-1 py-2">
						<span className="h-1.5 w-1.5 animate-pulse rounded-full bg-neutral-400" />
						<span
							className="h-1.5 w-1.5 animate-pulse rounded-full bg-neutral-400"
							style={{ animationDelay: "0.15s" }}
						/>
						<span
							className="h-1.5 w-1.5 animate-pulse rounded-full bg-neutral-400"
							style={{ animationDelay: "0.3s" }}
						/>
					</div>
				)}
				<span className="inline-block h-4 w-0.5 animate-pulse bg-neutral-400" />
			</div>
		</div>
	);
}
