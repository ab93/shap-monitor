"""Textual TUI application for live SHAP monitoring."""

from __future__ import annotations

from datetime import date, timedelta

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Input

from shapmonitor.cli._common import get_backend, parse_period, psi_label, psi_style


class WatchApp(App):
    """Live SHAP monitoring dashboard."""

    TITLE = "shapmonitor watch"
    CSS_PATH = None

    CSS = """
    #filter-input {
        dock: top;
        margin: 0 1;
        height: 3;
    }
    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("/", "focus_filter", "Filter"),
        Binding("escape", "clear_filter", "Clear filter"),
    ]

    def __init__(
        self,
        data_dir: str,
        refresh_interval: float = 5.0,
        period_spec: str = "last-7d",
    ) -> None:
        super().__init__()
        self._data_dir = data_dir
        self._refresh_interval = refresh_interval
        self._period_spec = period_spec
        self._filter_text = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="Type to filter features...", id="filter-input")
        yield DataTable(id="shap-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Feature", "Mean |SHAP|", "Std", "PSI (drift)", "Status")
        table.cursor_type = "row"

        self._load_data()
        self.set_interval(self._refresh_interval, self._load_data)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value.lower()
        self._refresh_table()

    def action_focus_filter(self) -> None:
        self.query_one("#filter-input", Input).focus()

    def action_clear_filter(self) -> None:
        inp = self.query_one("#filter-input", Input)
        inp.value = ""
        self._filter_text = ""
        self._refresh_table()

    def action_refresh(self) -> None:
        self._load_data()

    def _load_data(self) -> None:
        """Fetch fresh data from the backend and refresh the table."""
        from shapmonitor.analysis import SHAPAnalyzer

        backend = get_backend(self._data_dir)
        analyzer = SHAPAnalyzer(backend)

        period = parse_period(self._period_spec)

        try:
            summary = analyzer.summary(start_dt=period.start, end_dt=period.end)
        except Exception:
            summary = None

        # Drift: compare first half vs second half of the period
        try:
            mid = period.start + (period.end - period.start) / 2
            if isinstance(mid, date) and not hasattr(mid, "hour"):
                from shapmonitor.types import Period

                ref_period = Period(start=period.start, end=mid)
                curr_period = Period(start=mid + timedelta(days=1), end=period.end)
            else:
                from shapmonitor.types import Period

                ref_period = Period(start=period.start, end=mid)
                curr_period = Period(start=mid, end=period.end)

            drift = analyzer.compare_time_periods(ref_period, curr_period)
        except Exception:
            drift = None

        self._summary = summary
        self._drift = drift
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Rebuild the DataTable rows from cached data."""
        table = self.query_one(DataTable)
        table.clear()

        if self._summary is None or self._summary.empty:
            return

        import numpy as np

        # Sort by PSI descending so the most-drifting features rise to the top.
        # Features with no PSI (no overlap with reference period) go to the bottom.
        view = self._summary.copy()
        if self._drift is not None:
            view["_psi"] = self._drift["psi"].reindex(view.index)
        else:
            view["_psi"] = np.nan
        view = view.sort_values("_psi", ascending=False, na_position="last")

        from rich.text import Text

        for feat, row in view.iterrows():
            feat_str = str(feat)
            if self._filter_text and self._filter_text not in feat_str.lower():
                continue

            psi_val = row["_psi"]

            if np.isnan(psi_val):
                psi_str = "—"
                status = "n/a"
                style = "dim"
            else:
                psi_str = f"{psi_val:.4f}"
                status = psi_label(psi_val)
                style = psi_style(psi_val)

            table.add_row(
                feat_str,
                f"{row['mean_abs']:.4f}",
                f"{row['std']:.4f}",
                Text(psi_str, style=style),
                Text(status, style=style),
            )
