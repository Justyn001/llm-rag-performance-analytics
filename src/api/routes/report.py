"""
PDF Report endpoint — generates detailed benchmark report.

POST /api/report → Generates PDF from session data
"""

import io
import logging
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryData(BaseModel):
    """Single query data from frontend."""

    question: str
    answer: str
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    answer_len: int
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_per_second: float = 0.0
    eval_correct: bool | None = None
    eval_faithful: bool | None = None
    eval_context: bool | None = None


class PanelData(BaseModel):
    """Data from a single panel (one model)."""

    model_name: str
    queries: list[QueryData]


class ReportRequest(BaseModel):
    """Request to generate a PDF report."""

    panels: list[PanelData]
    rag_config: str = "basic-512"
    dataset: str = "wiki"
    language: str = "en"
    top_k: int = 5


def _tok_speed_str(queries) -> str:
    """Mean tokens/s as string, or em-dash if no data."""
    import numpy as np

    speeds = [q.tokens_per_second for q in queries if q.tokens_per_second > 0]
    return f"{np.mean(speeds):.1f}" if speeds else "\u2014"


@router.post("/report")
async def generate_report(request: ReportRequest):
    """Generate a PDF benchmark report from session data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # ---- TITLE PAGE ----
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        fig.patch.set_facecolor("#0a0a0f")

        ax.text(
            0.5,
            0.7,
            "RAG Performance Analytics",
            transform=ax.transAxes,
            fontsize=32,
            fontweight="bold",
            color="#e8e8f0",
            ha="center",
            va="center",
        )
        ax.text(
            0.5,
            0.58,
            "Benchmark Report",
            transform=ax.transAxes,
            fontsize=20,
            color="#a29bfe",
            ha="center",
            va="center",
        )

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        model_names = ", ".join(p.model_name for p in request.panels)
        info_lines = [
            f"Date: {now}",
            f"Models: {model_names}",
            f"RAG Config: {request.rag_config}",
            f"Dataset: {request.dataset}",
            f"Language: {request.language.upper()}",
            f"Top-K: {request.top_k}",
            f"Queries per model: {max(len(p.queries) for p in request.panels)}",
        ]
        info_text = "\n".join(info_lines)
        ax.text(
            0.5,
            0.35,
            info_text,
            transform=ax.transAxes,
            fontsize=12,
            color="#9898b0",
            ha="center",
            va="center",
            family="monospace",
            linespacing=1.8,
        )

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ---- PER-MODEL PAGES ----
        colors = ["#6c5ce7", "#00cec9", "#fdcb6e", "#ff6b6b"]

        for idx, panel in enumerate(request.panels):
            if not panel.queries:
                continue

            queries = panel.queries
            n = len(queries)
            color = colors[idx % len(colors)]

            retrieval = [q.retrieval_ms for q in queries]
            generation = [q.generation_ms for q in queries]
            total = [q.total_ms for q in queries]
            lengths = [q.answer_len for q in queries]

            # --- Stats Page ---
            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            fig.patch.set_facecolor("#0a0a0f")
            fig.suptitle(
                f"Model: {panel.model_name}",
                fontsize=18,
                fontweight="bold",
                color="#e8e8f0",
                y=0.96,
            )

            # Stacked bar: retrieval vs generation
            ax = axes[0, 0]
            ax.set_facecolor("#12121a")
            x = np.arange(n)
            ax.bar(x, retrieval, color="#a29bfe", label="Retrieval")
            ax.bar(
                x,
                generation,
                bottom=retrieval,
                color=color,
                alpha=0.8,
                label="Generation",
            )
            ax.set_xlabel("Query", color="#9898b0", fontsize=9)
            ax.set_ylabel("Time (ms)", color="#9898b0", fontsize=9)
            ax.set_title(
                "Response Time Breakdown",
                color="#e8e8f0",
                fontsize=11,
            )
            ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333")
            ax.tick_params(colors="#686888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333")

            # Answer length bar
            ax = axes[0, 1]
            ax.set_facecolor("#12121a")
            ax.bar(x, lengths, color=color, alpha=0.7)
            ax.set_xlabel("Query", color="#9898b0", fontsize=9)
            ax.set_ylabel("Characters", color="#9898b0", fontsize=9)
            ax.set_title(
                "Answer Length",
                color="#e8e8f0",
                fontsize=11,
            )
            ax.tick_params(colors="#686888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333")

            # Boxplot of timings
            ax = axes[1, 0]
            ax.set_facecolor("#12121a")
            ax.boxplot(
                [retrieval, generation, total],
                labels=["Retrieval", "Generation", "Total"],
                patch_artist=True,
                boxprops=dict(facecolor="#1a1a2e", color=color),
                medianprops=dict(color="#fdcb6e"),
                whiskerprops=dict(color="#686888"),
                capprops=dict(color="#686888"),
                flierprops=dict(
                    markerfacecolor="#ff6b6b",
                    markersize=4,
                ),
            )
            ax.set_ylabel("Time (ms)", color="#9898b0", fontsize=9)
            ax.set_title(
                "Timing Distribution",
                color="#e8e8f0",
                fontsize=11,
            )
            ax.tick_params(colors="#686888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333")

            # Stats table
            ax = axes[1, 1]
            ax.set_facecolor("#12121a")
            ax.set_axis_off()

            stats_data = [
                [
                    "Metric",
                    "Retrieval",
                    "Generation",
                    "Total",
                    "Tok/s",
                ],
                [
                    "Mean",
                    f"{np.mean(retrieval):.0f}ms",
                    f"{np.mean(generation):.0f}ms",
                    f"{np.mean(total):.0f}ms",
                    _tok_speed_str(queries),
                ],
                [
                    "Median",
                    f"{np.median(retrieval):.0f}ms",
                    f"{np.median(generation):.0f}ms",
                    f"{np.median(total):.0f}ms",
                    "—",
                ],
                [
                    "Std Dev",
                    f"{np.std(retrieval):.1f}",
                    f"{np.std(generation):.1f}",
                    f"{np.std(total):.1f}",
                    "—",
                ],
                [
                    "Min",
                    f"{np.min(retrieval):.0f}ms",
                    f"{np.min(generation):.0f}ms",
                    f"{np.min(total):.0f}ms",
                    "—",
                ],
                [
                    "Max",
                    f"{np.max(retrieval):.0f}ms",
                    f"{np.max(generation):.0f}ms",
                    f"{np.max(total):.0f}ms",
                    "—",
                ],
                [
                    "Count",
                    f"{n}",
                    f"{n}",
                    f"{n}",
                    f"{n}",
                ],
            ]

            table = ax.table(
                cellText=stats_data,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)

            for key, cell in table.get_celld().items():
                cell.set_facecolor("#1a1a2e")
                cell.set_edgecolor("#333")
                cell.set_text_props(color="#e8e8f0")
                if key[0] == 0:
                    cell.set_facecolor("#2a2a4e")
                    cell.set_text_props(
                        color="#a29bfe",
                        fontweight="bold",
                    )

            ax.set_title(
                "Detailed Statistics",
                color="#e8e8f0",
                fontsize=11,
                pad=20,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

            # --- Query Details Page ---
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_axis_off()

            ax.text(
                0.5,
                0.97,
                f"Query Details — {panel.model_name}",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                color="#e8e8f0",
                ha="center",
                va="top",
            )

            query_data = [
                ["#", "Question", "Ret.", "Gen.", "Total", "Tok/s", "Eval"],
            ]
            for i, q in enumerate(queries):
                qtext = q.question[:40] + "..." if len(q.question) > 40 else q.question
                eval_str = ""
                if q.eval_correct is not None:
                    eval_str += "C" if q.eval_correct else "c"
                if q.eval_faithful is not None:
                    eval_str += "F" if q.eval_faithful else "f"
                if q.eval_context is not None:
                    eval_str += "X" if q.eval_context else "x"
                query_data.append(
                    [
                        f"{i + 1}",
                        qtext,
                        f"{q.retrieval_ms:.0f}",
                        f"{q.generation_ms:.0f}",
                        f"{q.total_ms:.0f}",
                        f"{q.tokens_per_second:.1f}" if q.tokens_per_second > 0 else "—",
                        eval_str or "—",
                    ]
                )

            table = ax.table(
                cellText=query_data,
                loc="center",
                cellLoc="center",
                colWidths=[0.04, 0.38, 0.09, 0.09, 0.09, 0.09, 0.08],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)

            for key, cell in table.get_celld().items():
                cell.set_facecolor("#1a1a2e")
                cell.set_edgecolor("#333")
                cell.set_text_props(color="#e8e8f0")
                if key[0] == 0:
                    cell.set_facecolor("#2a2a4e")
                    cell.set_text_props(
                        color="#a29bfe",
                        fontweight="bold",
                    )
                if key[1] == 1:
                    cell.set_text_props(ha="left")

            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

        # ---- COMPARISON PAGE ----
        if len(request.panels) > 1:
            panels_with_data = [p for p in request.panels if p.queries]
            if len(panels_with_data) > 1:
                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(11.69, 8.27),
                )
                fig.patch.set_facecolor("#0a0a0f")
                fig.suptitle(
                    "Model Comparison",
                    fontsize=18,
                    fontweight="bold",
                    color="#e8e8f0",
                    y=0.96,
                )

                names = [p.model_name for p in panels_with_data]
                avg_ret = [np.mean([q.retrieval_ms for q in p.queries]) for p in panels_with_data]
                avg_gen = [np.mean([q.generation_ms for q in p.queries]) for p in panels_with_data]
                avg_len = [np.mean([q.answer_len for q in p.queries]) for p in panels_with_data]

                # Grouped bar: avg times
                ax = axes[0]
                ax.set_facecolor("#12121a")
                x = np.arange(len(names))
                w = 0.35
                ax.bar(
                    x - w / 2,
                    avg_ret,
                    w,
                    label="Retrieval",
                    color="#a29bfe",
                )
                ax.bar(
                    x + w / 2,
                    avg_gen,
                    w,
                    label="Generation",
                    color="#00cec9",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [n[:15] for n in names],
                    rotation=15,
                    fontsize=8,
                )
                ax.set_ylabel(
                    "Avg Time (ms)",
                    color="#9898b0",
                    fontsize=9,
                )
                ax.set_title(
                    "Average Response Time",
                    color="#e8e8f0",
                    fontsize=11,
                )
                ax.legend(
                    fontsize=8,
                    facecolor="#1a1a2e",
                    edgecolor="#333",
                )
                ax.tick_params(colors="#686888", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#333")

                # Avg answer length
                ax = axes[1]
                ax.set_facecolor("#12121a")
                bar_colors = [colors[i % len(colors)] for i in range(len(names))]
                ax.bar(x, avg_len, color=bar_colors, alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [n[:15] for n in names],
                    rotation=15,
                    fontsize=8,
                )
                ax.set_ylabel(
                    "Avg Length (chars)",
                    color="#9898b0",
                    fontsize=9,
                )
                ax.set_title(
                    "Average Answer Length",
                    color="#e8e8f0",
                    fontsize=11,
                )
                ax.tick_params(colors="#686888", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#333")

                plt.tight_layout(rect=[0, 0, 1, 0.94])
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)

        # ---- HUMAN EVALUATION PAGE ----
        panels_with_eval = [
            p
            for p in request.panels
            if p.queries
            and any(
                q.eval_correct is not None
                or q.eval_faithful is not None
                or q.eval_context is not None
                for q in p.queries
            )
        ]
        if panels_with_eval:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#12121a")
            fig.suptitle(
                "Human Evaluation",
                fontsize=18,
                fontweight="bold",
                color="#e8e8f0",
                y=0.96,
            )

            names = [p.model_name for p in panels_with_eval]
            x_pos = np.arange(len(names))

            def eval_pct(panel_data, field):
                rated = [
                    getattr(q, field) for q in panel_data.queries if getattr(q, field) is not None
                ]
                if not rated:
                    return 0
                return sum(1 for v in rated if v) / len(rated) * 100

            correct_pcts = [eval_pct(p, "eval_correct") for p in panels_with_eval]
            faithful_pcts = [eval_pct(p, "eval_faithful") for p in panels_with_eval]
            context_pcts = [eval_pct(p, "eval_context") for p in panels_with_eval]

            w = 0.25
            ax.bar(
                x_pos - w,
                correct_pcts,
                w,
                label="Correctness",
                color="#00cec9",
            )
            ax.bar(
                x_pos,
                faithful_pcts,
                w,
                label="Faithfulness",
                color="#a29bfe",
            )
            ax.bar(
                x_pos + w,
                context_pcts,
                w,
                label="Context Rel.",
                color="#fdcb6e",
            )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [n[:15] for n in names],
                rotation=15,
                fontsize=9,
            )
            ax.set_ylabel(
                "Score (%)",
                color="#9898b0",
                fontsize=10,
            )
            ax.set_ylim(0, 110)
            ax.legend(
                fontsize=9,
                facecolor="#1a1a2e",
                edgecolor="#333",
            )
            ax.tick_params(colors="#686888", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#333")

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

    buf.seek(0)
    filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
