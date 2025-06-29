
# Updated FastAPI with Laravel Security Integration
# secure_api_laravel.py
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from python_laravel_security_client import LaravelSecurityClient
from rag_system import VTRailroadRAG
import asyncio

app = FastAPI(title="Virginia & Truckee Railroad Chatbot API")

# Initialize clients
security_client = LaravelSecurityClient()
rag_system = None

class SecurityCheck:
async def __call__(self, request: Request, chat_request: ChatRequest):
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "unknown")
        message = chat_request.message

        # Validate through Laravel security service
        allowed, reason = await security_client.validate_request(client_ip, user_agent, message)

        if not allowed:
            raise HTTPException(status_code=429, detail=f"Request blocked: {reason}")

        return {"ip": client_ip, "user_agent": user_agent, "message": message}

security_check = SecurityCheck()

@app.post("/chat")
async def chat_endpoint(
                request: Request,
                chat_request: ChatRequest,
                security_data: dict = Depends(security_check)
            ):
    """
    Secure chat endpoint using Laravel security validation
    """
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system not available")

        # Process the query
        result = rag_system.query(chat_request.message)

        # Calculate estimated cost
        input_tokens = len(chat_request.message.split()) * 1.3
        output_tokens = len(result["answer"].split()) * 1.3
        total_cost_cents = int(input_tokens * 0.0003 + output_tokens * 0.0015)

        # Log successful request through Laravel
        await security_client.log_request(
                security_data["ip"],
                security_data["user_agent"],
                chat_request.message,
                response_tokens=int(output_tokens),
                cost_cents=total_cost_cents,
                blocked=False
            )

        return ChatResponse(
            response=result["answer"],
            sources=result["context_sources"],
            session_id=chat_request.session_id
        )

    except Exception as e:
        # Log error through Laravel
        await security_client.log_request(
                security_data["ip"],
                security_data["user_agent"],
                chat_request.message,
                blocked=True,
                block_reason=f"System error: {str(e)}"
        )

        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check with Laravel integration status"""
    stats = await security_client.get_usage_stats()

    return {
                "status": "healthy",
        "rag_system": "available" if rag_system else "unavailable",
        "laravel_security": "connected" if stats else "disconnected",
        "usage_stats": stats
    }
