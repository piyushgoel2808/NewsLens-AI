"""Unit and integration tests for Dynamic Answer Blueprint Architecture."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.answer_verifier import AnswerVerificationResult
from app.agent.evaluator import EvaluationVerdict
from app.agent.graph import AgentWorkflow
from app.agent.models import (
    AgentPlan,
    AnswerBlueprint,
    SectionSpec,
    ToolCallSpec,
)
from app.agent.planner import QueryPlanner, build_heuristic_answer_blueprint
from app.agent.synthesizer import AnswerSynthesizer, compile_structure_from_blueprint
from app.providers.base import ModelResponse


class TestAnswerBlueprintModels:
    """Test Pydantic model serialization and validation."""

    def test_section_spec_defaults(self) -> None:
        sec = SectionSpec(
            title="### ⚡ Executive Summary",
            format_type="narrative",
            content_focus="Main developments",
            target_length="120 words",
        )
        assert sec.title == "### ⚡ Executive Summary"
        assert sec.format_type == "narrative"
        d = sec.model_dump()
        assert d["target_length"] == "120 words"

    def test_answer_blueprint_serialization(self) -> None:
        bp = AnswerBlueprint(
            user_intent="single_article_word_limited_summary",
            overall_tone="executive_brief",
            target_word_count=150,
            sections=[
                SectionSpec(
                    title="### ⚡ Executive Summary: BRICS 2026",
                    format_type="narrative",
                    content_focus="Operational details",
                    target_length="150 words maximum",
                ),
                SectionSpec(
                    title="### 📌 Key Takeaways",
                    format_type="bullet_list",
                    content_focus="Personnel numbers",
                    target_length="3 bullets",
                ),
            ],
            table_columns=None,
            prohibited_elements=["robotic catalog tables", "conversational filler"],
        )
        data = bp.model_dump()
        assert data["target_word_count"] == 150
        assert len(data["sections"]) == 2
        assert "robotic catalog tables" in data["prohibited_elements"]

        # Round-trip deserialization
        reconstructed = AnswerBlueprint(**data)
        assert reconstructed.target_word_count == 150
        assert reconstructed.sections[0].title == "### ⚡ Executive Summary: BRICS 2026"


class TestPlannerBlueprintGeneration:
    """Test heuristic and structured planner blueprint generation."""

    def test_word_count_and_single_article_detection(self) -> None:
        query = "Security heightened, key areas beautified for Brics?, find this article and SUMMARISE it IN 150 WORDS"
        bp = build_heuristic_answer_blueprint(query, "factual_lookup")
        assert bp.target_word_count == 150
        assert bp.user_intent == "single_article_summary"
        assert bp.overall_tone == "executive_brief"
        assert len(bp.sections) >= 2
        assert any("Executive Summary" in s.title for s in bp.sections)
        assert any(s.format_type == "bullet_list" for s in bp.sections)
        assert "robotic catalog tables" in bp.prohibited_elements

    def test_table_comparison_request(self) -> None:
        query = "Compare how HT and The Goan covered inflation in a comparison table"
        bp = build_heuristic_answer_blueprint(query, "cross_newspaper_comparison")
        assert bp.user_intent == "cross_newspaper_tabular_comparison"
        assert bp.table_columns is not None
        assert "Publication" in bp.table_columns
        assert any(s.format_type == "markdown_table" for s in bp.sections)

    def test_scalar_count_metric_blueprint(self) -> None:
        query = "How many advertisements are there in Hindustan Times 2026-09-11?"
        bp = build_heuristic_answer_blueprint(query, "quantitative_trend")
        assert bp.user_intent == "scalar_count_metric"
        assert bp.overall_tone == "concise_atomic"
        assert bp.sections[0].title == "### ⚡ Direct Finding"
        assert bp.sections[1].title == "### 📊 Key Computed Metrics"
        assert any("catalog tables" in p for p in bp.prohibited_elements)

    def test_structured_planner_attaches_blueprint(self) -> None:
        planner = QueryPlanner()
        plan_obj = AgentPlan(
            thought_process="User wants concise summary of Article 42983",
            archetype="factual_lookup",
            tool_calls=[
                ToolCallSpec(
                    tool_name="hybrid_search",
                    arguments={"query": "Brics summit", "top_k": 3},
                    purpose="Retrieve target article",
                )
            ],
            answer_blueprint=AnswerBlueprint(
                user_intent="custom_intent",
                overall_tone="executive_brief",
                target_word_count=100,
                sections=[
                    SectionSpec(
                        title="### ⚡ Direct Summary",
                        format_type="narrative",
                        content_focus="Summary",
                        target_length="100 words",
                    )
                ],
            ),
        )
        plan_res = planner._build_plan_from_structured_model(
            query="summarise in 100 words",
            plan_obj=plan_obj,
        )
        assert plan_res.answer_blueprint is not None
        assert plan_res.answer_blueprint.target_word_count == 100
        assert plan_res.answer_blueprint.sections[0].title == "### ⚡ Direct Summary"


class TestSynthesizerBlueprintCompilation:
    """Test dynamic compilation of AnswerBlueprint into system prompt instructions."""

    def test_compile_structure_from_blueprint(self) -> None:
        bp = AnswerBlueprint(
            user_intent="test_intent",
            overall_tone="authoritative_journalistic",
            target_word_count=120,
            sections=[
                SectionSpec(
                    title="### ⚡ Operational Overview",
                    format_type="narrative",
                    content_focus="Anti-drone deployment and security perimeter",
                    target_length="80 words",
                ),
                SectionSpec(
                    title="### 📌 Key Takeaways",
                    format_type="bullet_list",
                    content_focus="Personnel counts and street beautification",
                    target_length="3 bullets",
                ),
            ],
            prohibited_elements=["robotic catalog tables", "fake metrics"],
        )
        prompt_snippet = compile_structure_from_blueprint(bp)
        assert "### ⚡ Operational Overview" in prompt_snippet
        assert "### 📌 Key Takeaways" in prompt_snippet
        assert "Target Length: 80 words." in prompt_snippet
        assert "Anti-drone deployment" in prompt_snippet
        assert "STRICT CITATION RULE" in prompt_snippet
        assert "120 words" in prompt_snippet
        assert "DO NOT output robotic catalog tables." in prompt_snippet

    def test_synthesizer_uses_blueprint_when_provided(self) -> None:
        synth = AnswerSynthesizer(provider=MagicMock())
        bp = AnswerBlueprint(
            user_intent="tabular_comparison",
            overall_tone="analytical_comparison",
            sections=[
                SectionSpec(
                    title="### 📊 Cross-Newspaper Comparison Table",
                    format_type="markdown_table",
                    content_focus="Fiscal policy stances",
                    target_length="1 row per paper",
                )
            ],
            table_columns=["Publication", "Headline", "Focus"],
            prohibited_elements=["crime stories"],
        )
        sys_prompt = synth._build_synthesizer_system_prompt(
            archetype="factual_lookup",
            query="compare in a table",
            evidence_items=[],
            answer_blueprint=bp,
        )
        assert "### 📊 Cross-Newspaper Comparison Table" in sys_prompt
        assert "Format as Markdown Table with columns: | Publication | Headline | Focus |" in sys_prompt
        assert "DO NOT output crime stories." in sys_prompt

    def test_synthesizer_falls_back_when_blueprint_is_none(self) -> None:
        synth = AnswerSynthesizer(provider=MagicMock())
        sys_prompt = synth._build_synthesizer_system_prompt(
            archetype="thematic_timeline",
            query="how did the case evolve over time?",
            evidence_items=[],
            answer_blueprint=None,
        )
        # Should use standard thematic_timeline structure
        assert "### 📅 Milestone Timeline & Event Progression" in sys_prompt


@pytest.mark.asyncio
class TestEndToEndWorkflowWithBlueprint:
    """Test full workflow cycle populates and preserves AnswerBlueprint."""

    async def test_workflow_runs_and_populates_blueprint(self) -> None:
        mock_session_factory = MagicMock()
        mock_db = MagicMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_session_factory.return_value.__aenter__.return_value = mock_db

        workflow = AgentWorkflow(session_factory=mock_session_factory)
        workflow._cache.get_query = AsyncMock(return_value=None)

        mock_provider = MagicMock()
        mock_provider.provider_name = "mock"
        mock_provider.complete = AsyncMock(
            return_value=ModelResponse(
                text="### ⚡ Direct Finding\nThere are 10 advertisements in Hindustan Times.",
                input_tokens=50,
                output_tokens=20,
            )
        )
        workflow._synthesizer._provider = mock_provider
        workflow._planner._provider = mock_provider
        workflow._verifier._provider = mock_provider
        workflow._evaluator._provider = mock_provider
        workflow._executor.execute_tools = AsyncMock(return_value=([], [], {}))
        workflow._evaluator.evaluate_evidence_async = AsyncMock(
            return_value=EvaluationVerdict(
                is_sufficient=True,
                quality_score=1.0,
                recommended_action="proceed_to_synthesis",
            )
        )
        workflow._verifier.verify_answer_async = AsyncMock(
            return_value=AnswerVerificationResult(
                is_valid=True,
                recommended_action="accept",
            )
        )
        if workflow._tool_maker:
            workflow._tool_maker.generate_and_execute = AsyncMock(
                return_value=MagicMock(success=True, evidence_items=[])
            )

        # Query asking for ad count
        state = await workflow.run(
            query="how many advertisements in Hindustan Times 2026-09-11",
            user_id="test_user",
        )
        assert state["answer_blueprint"] is not None
        assert state["answer_blueprint"]["user_intent"] == "scalar_count_metric"
        assert state["answer_blueprint"]["sections"][0]["title"] == "### ⚡ Direct Finding"
