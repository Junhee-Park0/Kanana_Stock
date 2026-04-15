from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, ConfigDict

class InitialOutput(BaseModel):
    """에이전트 초기 의견 출력 형식"""
    text: str = Field(..., description = "초기 의견 텍스트")
    tool_calls: List[Dict[str, Any]] = Field(..., description = "도구 호출 기록 리스트")

class DebateOutput(BaseModel):
    """에이전트 토론 중 의견 출력 형식"""
    text: str = Field(..., description = "토론 중 의견 텍스트")
    tool_calls: List[Dict[str, Any]] = Field(..., description = "도구 호출 기록 리스트")

class ConsensusOutput(BaseModel):
    """중재자 에이전트 최종 결론 출력 형식"""
    pros: str = Field(..., description = "핵심 기회 요인 요약")
    cons: str = Field(..., description = "핵심 리스크 요인 요약")
    recommendation: Literal["매수", "매도", "보류"] = Field(..., description = "최종 투자 의견")
    conclusion: str = Field(..., description = "종합 결론 본문")

    @property
    def to_report_text(self) -> str:
        """보고서 텍스트 형식으로 변환"""
        report = []
        report.append(f"**토론 흐름 요약**")
        if self.pros:
            report.append(f"[핵심 기회 요인 (Pros)]\n{self.pros}")
        if self.cons:
            report.append(f"[핵심 리스크 요인 (Cons)]\n{self.cons}")

        report.append(f"**최종 투자 의견:** {self.recommendation}")
        if self.conclusion:
            report.append(f"[종합 결론]\n{self.conclusion}")
        return "\n\n".join(report) if report else "합의안 도출에 실패했습니다."


