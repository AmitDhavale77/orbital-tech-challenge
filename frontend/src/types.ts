export interface Conversation {
	id: string;
	title: string;
	created_at: string;
	updated_at: string;
	has_document: boolean;
}

export interface Citation {
	document_id: string;
	document_name: string;
	page: number;
	quote: string;
	// Server-computed for on-PDF highlighting; rects are [x0, y0, x1, y1] in PDF points.
	rects: number[][];
	page_width: number | null;
	page_height: number | null;
}

export interface Step {
	kind: "search" | "read" | "list" | "summarize" | "tool";
	label: string;
	document_id: string | null;
	page: number | null;
}

export interface Message {
	id: string;
	conversation_id: string;
	role: "user" | "assistant" | "system";
	content: string;
	sources_cited: number;
	citations: Citation[];
	steps: Step[];
	created_at: string;
}

export interface Document {
	id: string;
	conversation_id: string;
	filename: string;
	page_count: number;
	uploaded_at: string;
}

export interface ConversationDetail extends Conversation {
	documents: Document[];
}
