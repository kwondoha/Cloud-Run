import random
import uuid
from locust import HttpUser, task, between

class ProductionAgentUser(HttpUser):
    """Elasticity test user for the Production ADK Agent."""

    wait_time = between(1, 3)  # Faster requests to trigger scaling

    def on_start(self):
        """Set up user session when starting."""
        self.user_id = f"user_{uuid.uuid4()}"
        self.session_id = f"session_{uuid.uuid4()}"

        # Create session for the Gemma agent using proper ADK API format
        session_data = {"state": {"user_type": "elasticity_test_user"}}

        self.client.post(
            f"/apps/production_agent/users/{self.user_id}/sessions/{self.session_id}",
            headers={"Content-Type": "application/json"},
            json=session_data,
        )

    @task(4)
    def test_conversations(self):
        """Test conversational capabilities - high frequency."""
        topics = [
            "What do red pandas typically eat in the wild?",
            "Can you tell me an interesting fact about snow leopards?",
            "Why are poison dart frogs so brightly colored?",
            "Where can I find the new baby kangaroo in the zoo?",
            "What is the name of your oldest gorilla?",
            "What time is the penguin feeding today?"
        ]

        # Use proper ADK API format for sending messages
        message_data = {
            "app_name": "production_agent",
            "user_id": self.user_id,
            "session_id": self.session_id,
            "new_message": {
                "role": "user",
                "parts": [{
                    "text": random.choice(topics)
                }]
            }
        }

        self.client.post(
            "/run",
            headers={"Content-Type": "application/json"},
            json=message_data,
        )

    @task(1)
    def health_check(self):
        """Test the health endpoint."""
        self.client.get("/health")

# 세션 생성: /apps/production_agent/users/{user_id}/sessions/{session_id}에 POST를 사용하여 적절한 ADK API 형식을 사용합니다. session_id 및 user_id를 만든 후 에이전트에 요청할 수 있습니다.
# 메시지 형식: app_name, user_id, session_id, 구조화된 new_message 객체를 사용하여 ADK 사양을 따름
# 대화 엔드포인트: /run 엔드포인트를 사용하여 모든 이벤트를 한 번에 수집합니다 (부하 테스트에 권장).
# Realistic Load: 대기 시간이 짧은 대화형 로드를 만듭니다.