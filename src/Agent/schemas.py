from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, ConfigDict

class EvidenceItem(BaseModel):
    """에이전트가 참고한 근거 하나의 형식"""
    source: Literal["news", "filing"] = Field(..., description = "근거 출처(뉴스 기사 또는 SEC 공시)")
    source_id: str = Field(..., description = "근거 ID (기사 ID 또는 공시 ID)")
    summary: str = Field(..., description = "근거 내용(핵심 사실 한 줄)")

    @property
    def to_text(self) -> str:
        """근거 텍스트 형식으로 변환"""
        if self.source == "news":
            return f"[근거 - News] source_id = {self.source_id} summary = {self.summary}"
        else:
            return f"[근거 - Filing] source_id = {self.source_id} summary = {self.summary}"

class InitialOutput(BaseModel):
    """에이전트 초기 의견 출력 형식"""
    text: str = Field(..., description = "초기 의견 텍스트")
    evidence: List[EvidenceItem] = Field(..., description = "초기 의견 근거 리스트")
    tool_calls: List[Dict[str, Any]] = Field(..., description = "도구 호출 기록 리스트")

class DebateOutput(BaseModel):
    """에이전트 토론 중 의견 출력 형식"""
    text: str = Field(..., description = "토론 중 의견 텍스트")
    opponent_text: str = Field(..., description = "상대방 의견 요약")
    evidence: List[EvidenceItem] = Field(..., description = "토론 중 의견 근거 리스트")
    tool_calls: List[Dict[str, Any]] = Field(..., description = "도구 호출 기록 리스트")

class ConsensusOutput(BaseModel):
    """중재자 에이전트 최종 결론 출력 형식"""
    pros: str = Field(..., description = "핵심 기회 요인 요약")
    cons: str = Field(..., description = "핵심 리스크 요인 요약")
    recommendation: Literal["매수", "매도", "보류"] = Field(..., description = "최종 투자 의견")
    conclusion: str = Field(..., description = "종합 결론 본문")
    evidence: List[EvidenceItem] = Field(..., description = "최종 결론 근거 리스트")

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
        report.append(f"[최종 결론 근거]")
        if self.evidence:
            for evidence_item in self.evidence:
                report.append(f"{evidence_item.to_text}")
        else:
            report.append("(근거 정보를 추출하지 못했습니다.)")
        return "\n\n".join(report) if report else "합의안 도출에 실패했습니다."


