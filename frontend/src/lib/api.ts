import type {
	Conversation,
	ConversationDetail,
	Document,
	Message,
} from "../types";

const BASE = "/api";

async function handleResponse<T>(response: Response): Promise<T> {
	if (!response.ok) {
		throw new Error(await errorMessage(response));
	}
	return response.json() as Promise<T>;
}

/** Pull a human-readable message out of an error response: prefer the API's
 * `{ detail }`, falling back to the raw body or the status. */
async function errorMessage(response: Response): Promise<string> {
	const text = await response.text().catch(() => "");
	if (text) {
		try {
			const detail = (JSON.parse(text) as { detail?: unknown }).detail;
			if (typeof detail === "string" && detail) return detail;
		} catch {
			// Body wasn't JSON — fall through to the raw text.
		}
		return text;
	}
	return `Request failed (${response.status})`;
}

export async function fetchConversations(): Promise<Conversation[]> {
	const res = await fetch(`${BASE}/conversations`);
	return handleResponse<Conversation[]>(res);
}

export async function createConversation(): Promise<Conversation> {
	const res = await fetch(`${BASE}/conversations`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ title: "New conversation" }),
	});
	return handleResponse<Conversation>(res);
}

export async function deleteConversation(id: string): Promise<void> {
	const res = await fetch(`${BASE}/conversations/${id}`, {
		method: "DELETE",
	});
	if (!res.ok) {
		throw new Error(await errorMessage(res));
	}
}

export async function fetchConversation(
	id: string,
): Promise<ConversationDetail> {
	const res = await fetch(`${BASE}/conversations/${id}`);
	return handleResponse<ConversationDetail>(res);
}

export async function fetchMessages(
	conversationId: string,
): Promise<Message[]> {
	const res = await fetch(`${BASE}/conversations/${conversationId}/messages`);
	return handleResponse<Message[]>(res);
}

export async function sendMessage(
	conversationId: string,
	content: string,
): Promise<Response> {
	const res = await fetch(`${BASE}/conversations/${conversationId}/messages`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ content }),
	});
	if (!res.ok) {
		throw new Error(await errorMessage(res));
	}
	return res;
}

export async function uploadDocument(
	conversationId: string,
	file: File,
): Promise<Document> {
	const formData = new FormData();
	formData.append("file", file);
	const res = await fetch(`${BASE}/conversations/${conversationId}/documents`, {
		method: "POST",
		body: formData,
	});
	return handleResponse<Document>(res);
}

export function getDocumentUrl(documentId: string): string {
	return `${BASE}/documents/${documentId}/content`;
}
