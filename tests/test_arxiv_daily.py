from arxiv_daily import (
    Paper,
    ScoredPaper,
    SUMMARY_LIMIT_MESSAGE,
    deduplicate_papers,
    generate_markdown_report,
    score_paper,
    with_summaries,
)


def make_paper(
    arxiv_id: str = "2501.00001",
    title: str = "Photon polarization and helicity amplitudes",
    abstract: str = "We study entanglement in collider observables.",
    categories: list[str] | None = None,
) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=["A. Author"],
        abstract=abstract,
        categories=categories or ["hep-ph"],
        published="2025-01-01T00:00:00Z",
        updated="2025-01-01T00:00:00Z",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def test_score_paper_gives_high_score_for_core_keywords() -> None:
    paper = make_paper()

    scored = score_paper(paper)

    assert scored.relevance == "high"
    assert scored.score >= 20
    assert "photon polarization" in scored.matched_keywords["title"]
    assert "helicity amplitudes" in scored.matched_keywords["title"]
    assert "entanglement" in scored.matched_keywords["abstract"]


def test_deduplicate_papers_removes_duplicate_arxiv_ids() -> None:
    first = make_paper(arxiv_id="2501.00001", title="First")
    duplicate = make_paper(arxiv_id="2501.00001", title="Duplicate")
    second = make_paper(arxiv_id="2501.00002", title="Second")

    papers = deduplicate_papers([first, duplicate, second])

    assert papers == [first, second]


def test_markdown_report_contains_title_url_and_score() -> None:
    paper = make_paper()
    scored = ScoredPaper(
        paper=paper,
        score=42,
        relevance="high",
        matched_keywords={"title": ["photon polarization"]},
        summary="summary failed",
    )

    markdown = generate_markdown_report([scored])

    assert "Photon polarization and helicity amplitudes" in markdown
    assert "https://arxiv.org/abs/2501.00001" in markdown
    assert "Relevance score: 42" in markdown
    assert "Abstract" in markdown
    assert "### hep-ph" in markdown


def test_summary_limit_is_separate_from_summary_failure(monkeypatch) -> None:
    def fake_summary(paper: Paper) -> str:
        return f"summary for {paper.arxiv_id}"

    monkeypatch.setattr("arxiv_daily.summarize_paper", fake_summary)
    monkeypatch.setenv("ENABLE_LLM_SUMMARIES", "1")
    monkeypatch.setenv("MAX_HIGH_SUMMARIES", "10")
    scored_papers = [
        ScoredPaper(
            paper=make_paper(arxiv_id=f"2501.{index:05d}", title=f"High paper {index}"),
            score=50 - index,
            relevance="high",
            matched_keywords={"title": ["photon polarization"]},
        )
        for index in range(11)
    ]

    enriched = with_summaries(scored_papers)

    assert enriched[9].summary == "summary for 2501.00009"
    assert enriched[10].summary == SUMMARY_LIMIT_MESSAGE


def test_llm_summaries_are_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_LLM_SUMMARIES", raising=False)
    scored = ScoredPaper(
        paper=make_paper(),
        score=42,
        relevance="high",
        matched_keywords={"title": ["model independent"]},
    )

    enriched = with_summaries([scored])

    assert enriched[0].summary is None


def test_model_independent_keywords_score_highly() -> None:
    paper = make_paper(
        title="A model-independent helicity amplitude framework",
        abstract="We develop basis-independent collider observables using sum rules.",
    )

    scored = score_paper(paper)

    assert scored.relevance == "high"
    assert "model-independent" in scored.matched_keywords["title"]


def test_report_separates_hep_ph_and_hep_th() -> None:
    hep_ph = ScoredPaper(
        paper=make_paper(arxiv_id="2501.00001", title="hep-ph candidate", categories=["hep-ph"]),
        score=30,
        relevance="high",
        matched_keywords={"title": ["model independent"]},
    )
    hep_th = ScoredPaper(
        paper=make_paper(arxiv_id="2501.00002", title="hep-th candidate", categories=["hep-th"]),
        score=29,
        relevance="high",
        matched_keywords={"title": ["symmetry"]},
    )

    markdown = generate_markdown_report([hep_ph, hep_th])

    assert "### hep-ph" in markdown
    assert "hep-ph candidate" in markdown
    assert "### hep-th" in markdown
    assert "hep-th candidate" in markdown
