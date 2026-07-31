from __future__ import annotations

import html
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
from openai import OpenAI


ARXIV_API_URL = "https://export.arxiv.org/api/query"
DEFAULT_CATEGORIES = ("hep-ph", "hep-th")
DEFAULT_MAX_RESULTS = 80
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_HIGH_SUMMARY_LIMIT = 3
DEFAULT_MEDIUM_SUMMARY_LIMIT = 0
DEFAULT_NOTION_PARENT_PAGE_ID = "39751c6c82cb817bb84ce5cd2b9fdef7"
DEFAULT_NOTION_DATABASE_ID = "b8a5a2eb179a49e3800ba185263d57e4"
NOTION_VERSION = "2022-06-28"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
KST = timezone(timedelta(hours=9), "KST")
SUMMARY_LIMIT_MESSAGE = "요약 없음: MVP 요약 한도 초과로 이번 실행에서는 LLM 요약을 생성하지 않았습니다."
SUMMARY_DISABLED_MESSAGE = "요약 없음: LLM 요약은 비활성화되어 있습니다. 후보 논문은 title/abstract 기반으로 선별했습니다."


KEYWORD_GROUPS: dict[str, list[str]] = {
    "very_high": [
        "model independent",
        "model-independent",
        "model agnostic",
        "model-agnostic",
        "photon polarization",
        "two-photon density matrix",
        "density matrix",
        "peres-horodecki",
        "peres horodecki",
        "ppt criterion",
        "positive partial transpose",
        "helicity amplitudes",
        "compton scattering",
        "e+e- -> gamma gamma",
        "e+ e- -> gamma gamma",
        "e+e- to gamma gamma",
        "gamma gamma",
        "entanglement",
    ],
    "high": [
        "effective field theory",
        "eft",
        "smeft",
        "operator basis",
        "basis independent",
        "basis-independent",
        "general formalism",
        "general framework",
        "sum rule",
        "sum rules",
        "null test",
        "null tests",
        "polarization",
        "polarisation",
        "helicity",
        "spin correlation",
        "spin correlations",
        "scattering amplitudes",
        "angular correlations",
        "collider observables",
        "observable",
        "observables",
        "kinematic",
        "kinematics",
        "symmetry",
        "symmetries",
        "quantum information",
    ],
    "medium": [
        "bounds",
        "constraints",
        "parameterization",
        "parametrization",
        "analysis framework",
        "particle physics phenomenology",
        "phenomenology",
        "collider",
        "scattering",
        "amplitude",
        "amplitudes",
        "spin",
        "photon",
        "two-photon",
        "bell inequality",
        "quantum",
    ],
}

GROUP_WEIGHTS = {
    "very_high": 8,
    "high": 5,
    "medium": 2,
}
TITLE_MULTIPLIER = 3
ABSTRACT_MULTIPLIER = 1
HIGH_RELEVANCE_THRESHOLD = 20
MEDIUM_RELEVANCE_THRESHOLD = 8


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str
    updated: str
    abs_url: str
    pdf_url: str


@dataclass(frozen=True)
class ScoredPaper:
    paper: Paper
    score: int
    relevance: str
    matched_keywords: dict[str, list[str]]
    summary: str | None = None
    scoop_score: int = 0
    scoop_level: str = "normal"
    scoop_reasons: tuple[str, ...] = ()


SCOOP_DIMENSIONS = {
    "same_system": (28, ("photon pair", "photon pairs", "two-photon", "gamma gamma", "annihilation photon")),
    "same_environment": (18, ("lepton collider", "lepton colliders", "e+e-", "e+ e-", "electron-positron")),
    "same_state": (22, ("density matrix", "two-qubit", "spin density matrix", "quantum state tomography")),
    "same_question": (20, ("entanglement", "bell inequality", "bell non-locality", "quantum information")),
    "same_method": (12, ("helicity amplitude", "helicity amplitudes", "model-independent", "general framework", "factorization framework", "stokes-mueller")),
}


def score_scoop_risk(paper: Paper) -> tuple[int, str, tuple[str, ...]]:
    """Estimate direct-overlap risk; this is not an allegation of idea theft."""
    text = f"{paper.title} {paper.abstract}".lower()
    score = 0
    reasons: list[str] = []
    for dimension, (weight, phrases) in SCOOP_DIMENSIONS.items():
        matches = sorted({phrase for phrase in phrases if keyword_matches(text, phrase)})
        if matches:
            score += weight
            reasons.append(f"{dimension}: {', '.join(matches[:3])}")
    score = min(score, 100)
    level = "critical" if score >= 75 else "high" if score >= 55 else "watch" if score >= 35 else "normal"
    return score, level, tuple(reasons)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def extract_arxiv_id(entry_id: str) -> str:
    return entry_id.rstrip("/").split("/")[-1]


def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        print(f"[warn] invalid integer for {name}={value!r}; using {default}")
        return default
    return max(parsed, 0)


def fetch_arxiv_papers(category: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Paper]:
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[warn] failed to fetch arXiv category {category}: {exc}")
        return []

    feed = feedparser.parse(response.text)
    if getattr(feed, "bozo", False):
        print(f"[warn] feedparser reported a parse issue for {category}: {feed.bozo_exception}")

    papers: list[Paper] = []
    for entry in feed.entries:
        abs_url = entry.get("id", "")
        arxiv_id = extract_arxiv_id(abs_url)
        pdf_url = next(
            (
                link.get("href", "")
                for link in entry.get("links", [])
                if link.get("title") == "pdf" or link.get("type") == "application/pdf"
            ),
            f"https://arxiv.org/pdf/{arxiv_id}",
        )

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=normalize_whitespace(entry.get("title", "")),
                authors=[author.get("name", "") for author in entry.get("authors", [])],
                abstract=normalize_whitespace(entry.get("summary", "")),
                categories=[tag.get("term", "") for tag in entry.get("tags", [])],
                published=entry.get("published", ""),
                updated=entry.get("updated", ""),
                abs_url=abs_url,
                pdf_url=pdf_url,
            )
        )

    return papers


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    deduplicated: list[Paper] = []
    for paper in papers:
        if paper.arxiv_id in seen:
            continue
        seen.add(paper.arxiv_id)
        deduplicated.append(paper)
    return deduplicated


def keyword_matches(text: str, keyword: str) -> bool:
    escaped = re.escape(keyword.lower())
    if re.search(r"[a-z0-9]", keyword.lower()):
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text.lower()) is not None
    return keyword.lower() in text.lower()


def score_paper(paper: Paper) -> ScoredPaper:
    title = paper.title.lower()
    abstract = paper.abstract.lower()
    score = 0
    matched_keywords: dict[str, list[str]] = {"title": [], "abstract": []}

    for group, keywords in KEYWORD_GROUPS.items():
        group_weight = GROUP_WEIGHTS[group]
        for keyword in keywords:
            matched = False
            if keyword_matches(title, keyword):
                score += group_weight * TITLE_MULTIPLIER
                matched_keywords["title"].append(keyword)
                matched = True
            if keyword_matches(abstract, keyword):
                score += group_weight * ABSTRACT_MULTIPLIER
                matched_keywords["abstract"].append(keyword)
                matched = True
            if matched:
                continue

    if score >= HIGH_RELEVANCE_THRESHOLD:
        relevance = "high"
    elif score >= MEDIUM_RELEVANCE_THRESHOLD:
        relevance = "medium"
    else:
        relevance = "low"

    scoop_score, scoop_level, scoop_reasons = score_scoop_risk(paper)
    return ScoredPaper(
        paper=paper,
        score=score,
        relevance=relevance,
        matched_keywords={
            location: sorted(set(keywords))
            for location, keywords in matched_keywords.items()
            if keywords
        },
        scoop_score=scoop_score,
        scoop_level=scoop_level,
        scoop_reasons=scoop_reasons,
    )


def summarize_paper(paper: Paper) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "요약 실패: OPENAI_API_KEY가 설정되어 있지 않습니다."

    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    client = OpenAI(api_key=api_key)
    prompt = f"""
아래 arXiv 논문의 title과 abstract만 근거로 한국어 요약을 작성하세요.
논문 전체를 읽은 것처럼 쓰지 말고, abstract에 없는 내용은 추정하지 마세요.

Title: {paper.title}
Abstract: {paper.abstract}

형식:
1. 한 줄 요약
2. 이 논문이 하는 일
3. 내 연구 관심사와의 관련성
4. 읽어볼 가치: 높음 / 중간 / 낮음 + 이유
""".strip()

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You summarize high-energy physics arXiv abstracts in Korean for a researcher.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_output_tokens=700,
        )
        summary = response.output_text.strip()
        return summary or "요약 실패: LLM 응답이 비어 있습니다."
    except Exception as exc:
        print(f"[warn] LLM summary failed for {paper.arxiv_id}: {exc}")
        return "요약 실패: LLM 요약 생성 중 오류가 발생했습니다."


def format_keywords(matched_keywords: dict[str, list[str]]) -> str:
    if not matched_keywords:
        return "none"
    parts = []
    for location in ("title", "abstract"):
        keywords = matched_keywords.get(location, [])
        if keywords:
            parts.append(f"{location}: {', '.join(keywords)}")
    return "; ".join(parts)


def format_abstract(abstract: str) -> str:
    return abstract or "No abstract available."


def format_paper_block(scored: ScoredPaper, include_summary: bool) -> str:
    paper = scored.paper
    authors = ", ".join(paper.authors) if paper.authors else "Unknown authors"
    categories = ", ".join(paper.categories) if paper.categories else "unknown"
    alert = {
        "critical": "🚨 **SCOOP RISK — 즉시 원문 비교 필요**",
        "high": "⚠️ **높은 직접 중복 가능성**",
        "watch": "🟡 **연구 중복 관찰 대상**",
    }.get(scored.scoop_level)
    lines = [
        alert,
        f"### {paper.title}",
        "",
        f"- arXiv: [{paper.arxiv_id}]({paper.abs_url}) | [PDF]({paper.pdf_url})",
        f"- Authors: {authors}",
        f"- Categories: {categories}",
        f"- Published: {paper.published or 'unknown'}",
        f"- Updated: {paper.updated or 'unknown'}",
        f"- Relevance score: {scored.score} ({scored.relevance})",
        f"- Scoop-risk score: {scored.scoop_score}/100 ({scored.scoop_level})",
        f"- Scoop-risk reasons: {'; '.join(scored.scoop_reasons) or 'none'}",
        f"- Matched keywords: {format_keywords(scored.matched_keywords)}",
        "",
        "**Abstract**",
        "",
        format_abstract(paper.abstract),
    ]
    if include_summary:
        lines.extend(["", scored.summary or "요약 실패: LLM 요약 결과가 없습니다."])
    return "\n".join(line for line in lines if line is not None)


def primary_category(paper: Paper) -> str:
    if "hep-ph" in paper.categories:
        return "hep-ph"
    if "hep-th" in paper.categories:
        return "hep-th"
    return "other/mixed"


def append_scored_section(lines: list[str], title: str, papers: list[ScoredPaper]) -> None:
    lines.extend(["", f"## {title}", ""])
    if not papers:
        lines.append(f"No {title.lower()} papers found.")
        return

    for category in ("hep-ph", "hep-th", "other/mixed"):
        category_papers = [scored for scored in papers if primary_category(scored.paper) == category]
        if not category_papers:
            continue
        lines.extend([f"### {category}", ""])
        lines.append("\n\n".join(format_paper_block(scored, include_summary=True) for scored in category_papers))
        lines.append("")


def append_low_section(lines: list[str], low: list[ScoredPaper]) -> None:
    lines.extend(["", "## Low Relevance", ""])
    if not low:
        lines.append("No low relevance papers found.")
        return

    for category in ("hep-ph", "hep-th", "other/mixed"):
        category_papers = [scored for scored in low if primary_category(scored.paper) == category]
        if not category_papers:
            continue
        lines.extend([f"### {category}", ""])
        for scored in category_papers:
            paper = scored.paper
            lines.append(
                f"- [{paper.title}]({paper.abs_url}) | [PDF]({paper.pdf_url}) "
                f"- score: {scored.score}, arXiv: {paper.arxiv_id}"
            )
        lines.append("")


def generate_markdown_report(scored_papers: list[ScoredPaper], report_date: str | None = None) -> str:
    today = report_date or datetime.now(KST).strftime("%Y-%m-%d")
    sorted_papers = sorted(scored_papers, key=lambda item: item.score, reverse=True)
    high = [paper for paper in sorted_papers if paper.relevance == "high"]
    medium = [paper for paper in sorted_papers if paper.relevance == "medium"]
    low = [paper for paper in sorted_papers if paper.relevance == "low"]

    lines = [
        f"# Daily arXiv Candidate Screening - {today}",
        "",
        "Categories: hep-ph, hep-th",
        "",
        "- Screening basis: title and abstract keyword matching",
        f"- LLM summaries enabled: {get_bool_env('ENABLE_LLM_SUMMARIES', False)}",
        "",
        f"- High relevance: {len(high)}",
        f"- Medium relevance: {len(medium)}",
        f"- Low relevance: {len(low)}",
    ]

    append_scored_section(lines, "High Relevance", high)
    append_scored_section(lines, "Medium Relevance", medium)
    append_low_section(lines, low)

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The bot does not read full PDFs. Screening is based on arXiv titles and abstracts.",
            "- LLM summaries are disabled by default to avoid unnecessary API cost.",
            "- arXiv API submittedDate ordering may differ from the daily announcement page.",
            "- Relevance is keyword-based and may miss relevant papers.",
            "",
            "## TODO",
            "",
            "- Embedding-based recommendation",
            "- PDF full-text parsing",
            "- Email sending",
            "- Notion integration",
            "- Slack/Discord integration",
            "- Web dashboard",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def save_report(markdown: str, date: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dated_report = REPORTS_DIR / f"arxiv_{date}.md"
    latest_report = REPORTS_DIR / "latest.md"
    dated_report.write_text(markdown, encoding="utf-8")
    shutil.copyfile(dated_report, latest_report)
    return dated_report


def notion_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def get_or_create_notion_database() -> str | None:
    if not os.getenv("NOTION_TOKEN"):
        return None
    configured = os.getenv("NOTION_DATABASE_ID", DEFAULT_NOTION_DATABASE_ID)
    if configured:
        return configured.replace("-", "")

    headers = notion_headers()
    search = requests.post(
        "https://api.notion.com/v1/search",
        headers=headers,
        json={"query": "논문 레이더", "filter": {"property": "object", "value": "database"}},
        timeout=30,
    )
    search.raise_for_status()
    for item in search.json().get("results", []):
        title = "".join(part.get("plain_text", "") for part in item.get("title", []))
        if title == "논문 레이더":
            return item["id"].replace("-", "")

    parent_id = os.getenv("NOTION_PARENT_PAGE_ID", DEFAULT_NOTION_PARENT_PAGE_ID)
    payload = {
        "parent": {"type": "page_id", "page_id": parent_id},
        "title": [{"type": "text", "text": {"content": "논문 레이더"}}],
        "properties": {
            "논문명": {"title": {}},
            "arXiv ID": {"rich_text": {}},
            "발표일": {"date": {}},
            "관련도": {"number": {}},
            "SCOOP 위험": {"select": {"options": [
                {"name": "🚨 Critical", "color": "red"},
                {"name": "⚠️ High", "color": "orange"},
                {"name": "🟡 Watch", "color": "yellow"},
                {"name": "Normal", "color": "gray"},
            ]}},
            "위험점수": {"number": {}},
            "원문": {"url": {}},
            "검토상태": {"select": {"options": [
                {"name": "즉시 비교", "color": "red"},
                {"name": "읽기 대기", "color": "yellow"},
                {"name": "보관", "color": "gray"},
            ]}},
        },
    }
    response = requests.post("https://api.notion.com/v1/databases", headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["id"].replace("-", "")


def upload_to_notion(scored_papers: list[ScoredPaper]) -> None:
    database_id = get_or_create_notion_database()
    if not database_id:
        print("[info] NOTION_TOKEN not set; skipping Notion upload")
        return
    headers = notion_headers()
    for scored in scored_papers:
        if scored.relevance == "low" and scored.scoop_level == "normal":
            continue
        paper = scored.paper
        query = requests.post(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            headers=headers,
            json={"filter": {"property": "arXiv ID", "rich_text": {"equals": paper.arxiv_id}}},
            timeout=30,
        )
        query.raise_for_status()
        if query.json().get("results"):
            continue
        risk_label = {"critical": "🚨 Critical", "high": "⚠️ High", "watch": "🟡 Watch"}.get(scored.scoop_level, "Normal")
        review = "즉시 비교" if scored.scoop_level in {"critical", "high"} else "읽기 대기" if scored.relevance != "low" else "보관"
        published_date = paper.published[:10] if paper.published else None
        body = (
            f"## 자동 판정\n\n- 직접 중복 위험: **{scored.scoop_score}/100 ({scored.scoop_level})**\n"
            f"- 판정 근거: {'; '.join(scored.scoop_reasons) or '직접 중복 신호 없음'}\n"
            f"- 일반 관련도: {scored.score} ({scored.relevance})\n\n"
            f"> 이 표시는 제목·초록 기반의 경쟁 가능성 경보다. 실제 스쿱 여부는 원문과 현재 원고의 claim-by-claim 비교로 확정한다.\n\n"
            f"## Abstract\n\n{paper.abstract[:1800]}\n\n[arXiv 원문]({paper.abs_url}) · [PDF]({paper.pdf_url})"
        )
        payload = {
            "parent": {"database_id": database_id},
            "properties": {
                "논문명": {"title": [{"text": {"content": paper.title[:2000]}}]},
                "arXiv ID": {"rich_text": [{"text": {"content": paper.arxiv_id}}]},
                "발표일": {"date": {"start": published_date}} if published_date else {"date": None},
                "관련도": {"number": scored.score},
                "SCOOP 위험": {"select": {"name": risk_label}},
                "위험점수": {"number": scored.scoop_score},
                "원문": {"url": paper.abs_url},
                "검토상태": {"select": {"name": review}},
            },
            "children": [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": body[:2000]}}]}}],
        }
        response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        print(f"[info] uploaded {paper.arxiv_id} to Notion")


def with_summaries(scored_papers: list[ScoredPaper]) -> list[ScoredPaper]:
    if not get_bool_env("ENABLE_LLM_SUMMARIES", False):
        return [
            ScoredPaper(
                paper=scored.paper,
                score=scored.score,
                relevance=scored.relevance,
                matched_keywords=scored.matched_keywords,
                summary=None,
                scoop_score=scored.scoop_score,
                scoop_level=scored.scoop_level,
                scoop_reasons=scored.scoop_reasons,
            )
            for scored in scored_papers
        ]

    high_limit = get_int_env("MAX_HIGH_SUMMARIES", DEFAULT_HIGH_SUMMARY_LIMIT)
    medium_limit = get_int_env("MAX_MEDIUM_SUMMARIES", DEFAULT_MEDIUM_SUMMARY_LIMIT)
    high_count = 0
    medium_count = 0
    enriched: list[ScoredPaper] = []

    for scored in scored_papers:
        should_summarize = False
        if scored.relevance == "high" and high_count < high_limit:
            should_summarize = True
            high_count += 1
        elif scored.relevance == "medium" and medium_count < medium_limit:
            should_summarize = True
            medium_count += 1

        if should_summarize:
            summary = summarize_paper(scored.paper)
            enriched.append(
                ScoredPaper(
                    paper=scored.paper,
                    score=scored.score,
                    relevance=scored.relevance,
                    matched_keywords=scored.matched_keywords,
                    summary=summary,
                    scoop_score=scored.scoop_score,
                    scoop_level=scored.scoop_level,
                    scoop_reasons=scored.scoop_reasons,
                )
            )
        elif scored.relevance in {"high", "medium"}:
            enriched.append(
                ScoredPaper(
                    paper=scored.paper,
                    score=scored.score,
                    relevance=scored.relevance,
                    matched_keywords=scored.matched_keywords,
                    summary=SUMMARY_LIMIT_MESSAGE,
                    scoop_score=scored.scoop_score,
                    scoop_level=scored.scoop_level,
                    scoop_reasons=scored.scoop_reasons,
                )
            )
        else:
            enriched.append(scored)

    return enriched


def main() -> None:
    all_papers: list[Paper] = []
    for index, category in enumerate(DEFAULT_CATEGORIES):
        print(f"[info] fetching arXiv category {category}")
        all_papers.extend(fetch_arxiv_papers(category, DEFAULT_MAX_RESULTS))
        if index < len(DEFAULT_CATEGORIES) - 1:
            time.sleep(3)

    papers = deduplicate_papers(all_papers)
    scored = sorted((score_paper(paper) for paper in papers), key=lambda item: item.score, reverse=True)
    summarized = with_summaries(scored)
    report_date = datetime.now(KST).strftime("%Y-%m-%d")
    markdown = generate_markdown_report(summarized, report_date=report_date)
    report_path = save_report(markdown, report_date)
    upload_to_notion(summarized)
    print(f"[info] wrote report to {report_path}")


if __name__ == "__main__":
    main()

