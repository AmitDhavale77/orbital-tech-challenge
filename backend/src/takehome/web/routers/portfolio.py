from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.session import get_session
from takehome.services.conversation import get_conversation
from takehome.services.portfolio import PortfolioAnswer, analyze_portfolio

router = APIRouter(tags=["portfolio"])


class PortfolioRequest(BaseModel):
    question: str


@router.post(
    "/api/conversations/{conversation_id}/portfolio",
    response_model=PortfolioAnswer,
)
async def portfolio_analysis(
    conversation_id: str,
    body: PortfolioRequest,
    session: AsyncSession = Depends(get_session),
) -> PortfolioAnswer:
    """Run map-reduce fan-out analysis across the whole bundle.

    The breadth/aggregation counterpart to chat: maps every document in parallel,
    then synthesises one grounded answer plus a per-document finding row (the
    "answer table"). See services/portfolio.py.
    """
    conversation = await get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return await analyze_portfolio(conversation_id, body.question)
