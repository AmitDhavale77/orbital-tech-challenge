import { AnimatePresence, motion } from "framer-motion";
import {
	BookOpen,
	ChevronRight,
	FileText,
	Layers,
	Loader2,
	Search,
	Wrench,
} from "lucide-react";
import { useState } from "react";
import type { Step } from "../types";

function StepIcon({ kind }: { kind: Step["kind"] }) {
	const className = "h-3 w-3 flex-shrink-0 text-neutral-400";
	if (kind === "search") return <Search className={className} />;
	if (kind === "read") return <FileText className={className} />;
	if (kind === "summarize") return <BookOpen className={className} />;
	if (kind === "list") return <Layers className={className} />;
	return <Wrench className={className} />;
}

function StepRow({
	step,
	active,
	onStepClick,
}: {
	step: Step;
	active?: boolean;
	onStepClick?: (step: Step) => void;
}) {
	const clickable = step.kind === "read" && !!step.document_id && !!step.page;
	const body = (
		<>
			<StepIcon kind={step.kind} />
			<span className={active ? "animate-pulse" : undefined}>{step.label}</span>
		</>
	);
	if (clickable) {
		return (
			<button
				type="button"
				onClick={() => onStepClick?.(step)}
				className="flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left text-neutral-500 transition-colors hover:bg-neutral-100 hover:text-neutral-800"
			>
				{body}
			</button>
		);
	}
	return (
		<div className="px-1 py-0.5 text-neutral-500">
			<div className="flex items-center gap-1.5">{body}</div>
			{step.detail && (
				<div className="ml-[1.125rem] text-neutral-400">{step.detail}</div>
			)}
		</div>
	);
}

function summarize(steps: Step[]): string {
	const searches = steps.filter((s) => s.kind === "search").length;
	const reads = steps.filter((s) => s.kind === "read").length;
	const summarized = steps.filter((s) => s.kind === "summarize").length;
	const parts: string[] = [];
	if (searches)
		parts.push(searches === 1 ? "Searched the bundle" : `${searches} searches`);
	if (reads) parts.push(`read ${reads} page${reads === 1 ? "" : "s"}`);
	if (summarized)
		parts.push(
			`summarised ${summarized} document${summarized === 1 ? "" : "s"}`,
		);
	return (
		parts.join(" · ") || `${steps.length} step${steps.length === 1 ? "" : "s"}`
	);
}

interface AgentStepsProps {
	steps: Step[];
	/** Live = the agent is still working (shimmer the latest step). */
	live?: boolean;
	onStepClick?: (step: Step) => void;
}

export function AgentSteps({ steps, live, onStepClick }: AgentStepsProps) {
	const [open, setOpen] = useState(false);
	if (steps.length === 0) return null;

	if (live) {
		return (
			<div className="mb-2 rounded-lg border border-neutral-200 bg-neutral-50 px-2 py-1.5 text-xs">
				<div className="flex items-center gap-1.5 font-medium text-neutral-500">
					<Loader2 className="h-3 w-3 animate-spin" />
					Working…
				</div>
				<div className="mt-1 space-y-0.5">
					<AnimatePresence initial={false}>
						{steps.map((step, i) => (
							<motion.div
								key={`${step.kind}-${i}`}
								initial={{ opacity: 0, y: -2 }}
								animate={{ opacity: 1, y: 0 }}
								transition={{ duration: 0.15 }}
							>
								<StepRow
									step={step}
									active={i === steps.length - 1}
									onStepClick={onStepClick}
								/>
							</motion.div>
						))}
					</AnimatePresence>
				</div>
			</div>
		);
	}

	// Finished: a compact, expandable summary that replays the steps.
	return (
		<div className="mb-1.5 text-xs">
			<button
				type="button"
				onClick={() => setOpen((o) => !o)}
				className="flex items-center gap-1 text-neutral-400 transition-colors hover:text-neutral-600"
			>
				<ChevronRight
					className={`h-3 w-3 transition-transform ${open ? "rotate-90" : ""}`}
				/>
				{summarize(steps)}
			</button>
			{open && (
				<div className="mt-1 space-y-0.5 border-l border-neutral-200 pl-2">
					{steps.map((step, i) => (
						<StepRow
							key={`${step.kind}-${i}`}
							step={step}
							onStepClick={onStepClick}
						/>
					))}
				</div>
			)}
		</div>
	);
}
