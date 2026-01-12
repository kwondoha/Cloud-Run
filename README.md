### 1. Cloud Run에 보안 MCP 서버를 배포하는 방법
  - MCP : AI 애플리케이션을 위한 표준화된 양방향 연결을 생성하여, LLM이 다양한 데이터 소스 및 도구에 쉽게 연결하도록 함
  - FastMCP를 사용하여 **get_animals_by_species** 및 **get_animal_details**의 두 도구가 있는 동물원 MCP 서버를 만들기
  - 모든 요청에 인증을 요구하여 승인된 클라이언트와 에이전트만 서버의 엔드포인트와 통신할 수 있도록 보호하기
  - Gemini CLI에서 보안 MCP 서버 엔드포인트에 연결하기
### 2. Cloud Run에서 MCP 서버를 사용하는 ADK 에이전트 빌드 및 배포
  - 클라이언트 에이전트 서비스의 구현 및 배포
  - ADK 배포를 위해 Python 프로젝트 구성
  - google-adk로 도구 사용 에이전트 구현
  - Python 애플리케이션을 서버리스 컨테이너로 Cloud Run에 배포
### 3. GPU를 사용하여 Cloud Run에 ADK 에이전트 배포
