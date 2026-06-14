import { ChevronLeft, ChevronRight, FileText, Loader2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Document as PDFDocument, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import { getDocumentUrl } from "../lib/api";
import type { Document } from "../types";
import { Button } from "./ui/button";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
	"pdfjs-dist/build/pdf.worker.min.mjs",
	import.meta.url,
).toString();

const MIN_WIDTH = 280;
const MAX_WIDTH = 700;
const DEFAULT_WIDTH = 400;

interface ViewerTarget {
	documentId: string;
	page: number;
	rects?: number[][];
	pageWidth?: number | null;
	pageHeight?: number | null;
}

interface DocumentViewerProps {
	documents: Document[];
	/** When set, the viewer opens this document at this page (citation click)
	 * and highlights the quote's rects if present. */
	target?: ViewerTarget | null;
}

export function DocumentViewer({ documents, target }: DocumentViewerProps) {
	const [activeId, setActiveId] = useState<string | null>(null);
	const [currentPage, setCurrentPage] = useState(1);
	const [numPages, setNumPages] = useState(0);
	// Track which document the loaded page-count / error belong to, so switching
	// documents never shows the previous document's state.
	const [loadedId, setLoadedId] = useState<string | null>(null);
	const [error, setError] = useState<{ id: string; message: string } | null>(
		null,
	);
	const [width, setWidth] = useState(DEFAULT_WIDTH);
	const [dragging, setDragging] = useState(false);
	const containerRef = useRef<HTMLDivElement>(null);

	// Open the cited document at the cited page. Depends on the target object
	// identity, so re-clicking the same citation re-triggers the jump.
	useEffect(() => {
		if (target) {
			setActiveId(target.documentId);
			setCurrentPage(target.page >= 1 ? target.page : 1);
		}
	}, [target]);

	const handleMouseDown = useCallback(
		(e: React.MouseEvent) => {
			e.preventDefault();
			setDragging(true);
			const startX = e.clientX;
			const startWidth = width;
			const handleMouseMove = (moveEvent: MouseEvent) => {
				const delta = startX - moveEvent.clientX;
				setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidth + delta)));
			};
			const handleMouseUp = () => {
				setDragging(false);
				window.removeEventListener("mousemove", handleMouseMove);
				window.removeEventListener("mouseup", handleMouseUp);
			};
			window.addEventListener("mousemove", handleMouseMove);
			window.addEventListener("mouseup", handleMouseUp);
		},
		[width],
	);

	const pdfPageWidth = width - 48; // account for px-4 padding on each side
	const active =
		documents.find((d) => d.id === activeId) ?? documents[0] ?? null;

	if (!active) {
		return (
			<div
				style={{ width }}
				className="flex h-full flex-shrink-0 flex-col items-center justify-center border-l border-neutral-200 bg-neutral-50"
			>
				<FileText className="mb-3 h-10 w-10 text-neutral-300" />
				<p className="text-sm text-neutral-400">No documents uploaded</p>
			</div>
		);
	}

	const pdfUrl = getDocumentUrl(active.id);
	const isLoaded = loadedId === active.id;
	const errorMessage = error?.id === active.id ? error.message : null;

	// Highlight the cited quote only while its own page of its own document is shown.
	const displayedPage = numPages
		? Math.min(currentPage, numPages)
		: currentPage;
	const showHighlights =
		!!target &&
		target.documentId === active.id &&
		displayedPage === target.page &&
		!!target.rects?.length &&
		!!target.pageWidth;
	const highlightScale = showHighlights
		? pdfPageWidth / (target.pageWidth as number)
		: 1;

	const selectDocument = (id: string) => {
		setActiveId(id);
		setCurrentPage(1);
	};

	return (
		<div
			ref={containerRef}
			style={{ width }}
			className="relative flex h-full flex-shrink-0 flex-col border-l border-neutral-200 bg-white"
		>
			{/* Resize handle */}
			<div
				className={`absolute top-0 left-0 z-10 h-full w-1.5 cursor-col-resize transition-colors hover:bg-neutral-300 ${
					dragging ? "bg-neutral-400" : ""
				}`}
				onMouseDown={handleMouseDown}
			/>

			{/* Header: document switcher (or single filename) */}
			<div className="border-b border-neutral-100 px-4 py-3">
				{documents.length > 1 ? (
					<select
						value={active.id}
						onChange={(e) => selectDocument(e.target.value)}
						className="w-full truncate rounded-md border border-neutral-200 bg-neutral-50 px-2 py-1.5 text-sm font-medium text-neutral-800 outline-none focus:border-neutral-400"
					>
						{documents.map((d) => (
							<option key={d.id} value={d.id}>
								{d.filename}
							</option>
						))}
					</select>
				) : (
					<p className="truncate text-sm font-medium text-neutral-800">
						{active.filename}
					</p>
				)}
				<p className="mt-1 text-xs text-neutral-400">
					{active.page_count} page{active.page_count !== 1 ? "s" : ""}
				</p>
			</div>

			{/* PDF content */}
			<div className="flex-1 overflow-y-auto p-4">
				{errorMessage && (
					<div className="rounded-lg bg-red-50 p-3 text-sm text-red-600">
						{errorMessage}
					</div>
				)}

				<PDFDocument
					key={active.id}
					file={pdfUrl}
					onLoadSuccess={({ numPages: pages }) => {
						setNumPages(pages);
						setLoadedId(active.id);
					}}
					onLoadError={(e) =>
						setError({
							id: active.id,
							message: `Failed to load PDF: ${e.message}`,
						})
					}
					loading={
						<div className="flex items-center justify-center py-12">
							<Loader2 className="h-6 w-6 animate-spin text-neutral-400" />
						</div>
					}
				>
					{isLoaded && !errorMessage && (
						<div className="relative inline-block">
							<Page
								pageNumber={Math.min(currentPage, numPages)}
								width={pdfPageWidth}
								loading={
									<div className="flex items-center justify-center py-12">
										<Loader2 className="h-5 w-5 animate-spin text-neutral-300" />
									</div>
								}
							/>
							{showHighlights &&
								target?.rects?.map((r) => {
									const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = r;
									return (
										<div
											key={`${x0}-${y0}-${x1}-${y1}`}
											className="pointer-events-none absolute rounded-[1px] bg-yellow-300/40 ring-1 ring-yellow-500/40"
											style={{
												left: x0 * highlightScale,
												top: y0 * highlightScale,
												width: (x1 - x0) * highlightScale,
												height: (y1 - y0) * highlightScale,
											}}
										/>
									);
								})}
						</div>
					)}
				</PDFDocument>
			</div>

			{/* Page navigation */}
			{isLoaded && numPages > 0 && (
				<div className="flex items-center justify-center gap-3 border-t border-neutral-100 px-4 py-2.5">
					<Button
						variant="ghost"
						size="icon"
						className="h-7 w-7"
						disabled={currentPage <= 1}
						onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
					>
						<ChevronLeft className="h-4 w-4" />
					</Button>
					<span className="text-xs text-neutral-500">
						Page {Math.min(currentPage, numPages)} of {numPages}
					</span>
					<Button
						variant="ghost"
						size="icon"
						className="h-7 w-7"
						disabled={currentPage >= numPages}
						onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))}
					>
						<ChevronRight className="h-4 w-4" />
					</Button>
				</div>
			)}
		</div>
	);
}
