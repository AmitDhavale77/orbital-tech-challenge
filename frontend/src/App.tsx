import { useCallback, useState } from "react";
import { ChatSidebar } from "./components/ChatSidebar";
import { ChatWindow } from "./components/ChatWindow";
import { DocumentViewer } from "./components/DocumentViewer";
import { TooltipProvider } from "./components/ui/tooltip";
import { useConversations } from "./hooks/use-conversations";
import { useDocuments } from "./hooks/use-documents";
import { useMessages } from "./hooks/use-messages";
import type { Citation, Step } from "./types";

interface ViewerTarget {
	documentId: string;
	page: number;
	rects?: number[][];
	pageWidth?: number | null;
	pageHeight?: number | null;
}

export default function App() {
	const {
		conversations,
		selectedId,
		loading: conversationsLoading,
		create,
		select,
		remove,
		refresh: refreshConversations,
	} = useConversations();

	const {
		messages,
		loading: messagesLoading,
		error: messagesError,
		streaming,
		streamingContent,
		streamingSteps,
		streamingReasoning,
		send,
	} = useMessages(selectedId);

	const {
		documents,
		upload,
		error: documentsError,
		refresh: refreshDocuments,
	} = useDocuments(selectedId);

	const [viewerTarget, setViewerTarget] = useState<ViewerTarget | null>(null);

	const handleCitationClick = useCallback((citation: Citation) => {
		// New object each click so the viewer re-jumps even to the same page.
		setViewerTarget({
			documentId: citation.document_id,
			page: citation.page,
			rects: citation.rects,
			pageWidth: citation.page_width,
			pageHeight: citation.page_height,
		});
	}, []);

	const handleStepClick = useCallback((step: Step) => {
		if (step.document_id && step.page) {
			setViewerTarget({ documentId: step.document_id, page: step.page });
		}
	}, []);

	const handleSend = useCallback(
		async (content: string) => {
			await send(content);
			refreshConversations();
		},
		[send, refreshConversations],
	);

	const handleUpload = useCallback(
		async (file: File) => {
			const doc = await upload(file);
			if (doc) {
				refreshDocuments();
				refreshConversations();
			}
		},
		[upload, refreshDocuments, refreshConversations],
	);

	const handleCreate = useCallback(async () => {
		await create();
	}, [create]);

	return (
		<TooltipProvider delayDuration={200}>
			<div className="flex h-screen bg-neutral-50">
				<ChatSidebar
					conversations={conversations}
					selectedId={selectedId}
					loading={conversationsLoading}
					onSelect={select}
					onCreate={handleCreate}
					onDelete={remove}
				/>

				<ChatWindow
					messages={messages}
					loading={messagesLoading}
					error={messagesError ?? documentsError}
					streaming={streaming}
					streamingContent={streamingContent}
					streamingSteps={streamingSteps}
					streamingReasoning={streamingReasoning}
					hasDocument={documents.length > 0}
					conversationId={selectedId}
					onSend={handleSend}
					onUpload={handleUpload}
					onCitationClick={handleCitationClick}
					onStepClick={handleStepClick}
				/>

				<DocumentViewer documents={documents} target={viewerTarget} />
			</div>
		</TooltipProvider>
	);
}
