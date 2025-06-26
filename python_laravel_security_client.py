// Updated Python Security Client
// python_laravel_security_client.py
import httpx
import os
from typing import Dict, Optional, Tuple
import asyncio

class LaravelSecurityClient:
    """
    Secure client for communicating with Laravel security API
    Much more secure than direct database calls
    """

    def __init__(self, laravel_base_url: str = None, api_key: str = None):
        self.base_url = laravel_base_url or os.getenv('LARAVEL_BASE_URL', 'https://virginiatruckee.com')
        self.api_key = api_key or os.getenv('LARAVEL_API_KEY')

        if not self.api_key:
            raise ValueError("LARAVEL_API_KEY environment variable required")

        # HTTP client with security settings
        self.client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )

    async def validate_request(self, ip_address: str, user_agent: str, message: str) -> Tuple[bool, Optional[str]]:
        """
        Validate request through Laravel security service
        Returns (allowed: bool, reason: Optional[str])
        """
        try:
            response = await self.client.post(
    f"{self.base_url}/api/chatbot-security/validate",
                json={
    'ip_address': ip_address,
                    'user_agent': user_agent,
                    'message': message,
                    'api_key': self.api_key
                },
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('allowed', False), data.get('reason')
            else:
                # Fail secure on API errors
                return False, "Security service unavailable"

        except Exception as e:
            print(f"Security validation error: {e}")
            return False, "Security validation failed"

    async def log_request(self, ip_address: str, user_agent: str, message: str,
                response_tokens: int = 0, cost_cents: int = 0,
                blocked: bool = False, block_reason: str = None) -> bool:
        """
        Log request through Laravel security service
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chatbot-security/log",
                json={
                'ip_address': ip_address,
                    'user_agent': user_agent,
                    'message': message,
                    'response_tokens': response_tokens,
                    'cost_cents': cost_cents,
                    'blocked': blocked,
                    'block_reason': block_reason,
                    'api_key': self.api_key
                }
            )

            return response.status_code == 200

        except Exception as e:
            print(f"Request logging error: {e}")
            return False

    async def get_usage_stats(self) -> Dict:
        """
        Get usage statistics from Laravel
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/api/chatbot-security/stats",
                params={'api_key': self.api_key}
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {}

        except Exception as e:
            print(f"Stats fetch error: {e}")
            return {}

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
