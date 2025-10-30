
from src.infrastructure.llm_client import get_llm_client
import asyncio

class PrecautionAgent:
    """Generates tailored precautions for festivals."""
    def __init__(self):
        # Use a slightly more creative temperature for generation
        self.llm = get_llm_client(temperature=0.5)

    async def generate_precautions(self, festival_name, detailed_category, prohibited_behaviors):
        """Orchestrates the generation of the final precaution message."""
        # Step 1 & 2: Generate precautions for category and behaviors in parallel
        category_task = self._generate_category_precautions(festival_name, detailed_category)
        behavior_task = self._generate_behavior_precautions(festival_name, prohibited_behaviors)
        
        category_precautions, behavior_precautions = await asyncio.gather(category_task, behavior_task)
        
        # Step 3: Combine them into a final, coherent message
        final_precautions = await self._combine_precautions(festival_name, category_precautions, behavior_precautions)
        
        return final_precautions

    async def _generate_category_precautions(self, festival_name, detailed_category):
        """Generates general advice based on the festival's detailed category."""
        if not detailed_category or detailed_category == "기타 전통문화":
            return ""

        prompt = f"""
        당신은 특정 유형의 한국 전통 문화 행사에 대한 맞춤형 주의사항을 생성하는 AI입니다.
        '{festival_name}' 축제는 '{detailed_category}' 유형으로 분류되었습니다.
        이 유형의 행사를 방문할 때 관광객이 일반적으로 명심해야 할 핵심적인 에티켓이나 주의사항을 2-3가지 항목으로 요약해주세요.
        친절하고 이해하기 쉬운 어투로 설명해주세요.
        """
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating category precautions: {e}")
            return ""

    async def _generate_behavior_precautions(self, festival_name, prohibited_behaviors):
        """Generates specific warnings based on a list of prohibited behaviors."""
        if not prohibited_behaviors or prohibited_behaviors == "일반적인 관광 예절 준수":
            return ""
            
        prompt = f"""
        당신은 특정 행동 규칙 목록을 바탕으로 관광객을 위한 주의사항을 생성하는 AI입니다.
        '{festival_name}' 축제에서는 다음 행동들이 문제가 될 수 있습니다: {prohibited_behaviors}.
        이 목록을 바탕으로, 방문객이 왜 이런 행동을 조심해야 하는지, 그리고 어떻게 행동하는 것이 좋은지 2-3가지 핵심 항목으로 요약해주세요.
        친절하고 이해하기 쉬운 어투로 설명해주세요.
        """
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating behavior precautions: {e}")
            return ""

    async def _combine_precautions(self, festival_name, category_precautions, behavior_precautions):
        """Combines two sets of precautions into a single, final guide."""
        if not category_precautions and not behavior_precautions:
            return "### 👑 AI 기반 에티켓 가이드\n특별한 주의사항은 없지만, 기본적인 관광 에티켓을 지켜주시면 모두가 즐거운 축제가 될 거예요!"

        prompt = f"""
        당신은 여러 주의사항 정보를 하나로 통합하여 최종 안내문을 작성하는 AI입니다.
        '{festival_name}' 축제에 대한 두 가지 종류의 주의사항 정보가 있습니다.

        [유형별 주의사항]
        {category_precautions}

        [행동별 주의사항]
        {behavior_precautions}

        [요청]
        위 두 정보를 자연스럽게 통합하여, 방문객을 위한 최종 주의사항 안내문을 작성해주세요.
        - "### 👑 {festival_name} 방문 시 이것만은 꼭! (AI 기반 에티켓 가이드)" 라는 제목으로 시작해주세요.
        - 각 주의사항은 글머리 기호(•)를 사용하여 명확하게 구분해주세요.
        - 전체적으로 일관성 있고 친절한 톤을 유지해주세요.
        - 불필요한 내용은 제거하고 핵심만 간결하게 정리해주세요.
        """
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error combining precautions: {e}")
            return "주의사항을 종합하는 데 실패했습니다."
