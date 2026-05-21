from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
from html import escape
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

# Allow running directly from a cloned repository without installation.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lingappan_mli import (  # noqa: E402
    DEFAULT_CALIBRATION_SOURCE,
    DEFAULT_PIXEL_HEIGHT_UM,
    DEFAULT_PIXEL_WIDTH_UM,
    UNRECORDED_CALIBRATION_SOURCE,
    AnalysisParams,
    process_files,
)
from lingappan_mli.ui import config as ui_config  # noqa: E402
from lingappan_mli.ui import previews as ui_previews  # noqa: E402
from lingappan_mli.ui import validation as ui_validation  # noqa: E402

logger = logging.getLogger(__name__)
_SESSION_EXPORT_RUNS: dict[str, set[str]] = {}
_SESSION_EXPORT_LOCK = threading.Lock()

GITHUB_README_URL = "https://github.com/lingappan-lab/mli#readme"
LOGO_SVG_PATH = ROOT / "assets" / "lingappan-mli-logo.svg"

SVG_ICONS = {
    "arrow-left": '<path d="M19 12H5"/><path d="m12 19-7-7 7-7"/>',
    "check": '<path d="m20 6-11 11-5-5"/>',
    "download": '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
    "error": '<circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/>',
    "filter": '<path d="M4 5h16"/><path d="M7 12h10"/><path d="M10 19h4"/>',
    "grid": '<path d="M4 4h16v16H4z"/><path d="M4 10h16"/><path d="M4 16h16"/><path d="M10 4v16"/><path d="M16 4v16"/>',
    "image": '<rect x="4" y="5" width="16" height="14" rx="2"/><path d="m4 15 4-4 4 4 3-3 5 5"/><circle cx="9" cy="10" r="1.5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7h.01"/>',
    "microscope": '<path d="M10 4h5"/><path d="M13 4v4"/><path d="m9 8 5 5"/><path d="M8 12a5 5 0 0 0 7 7"/><path d="M5 21h14"/><path d="M9 21v-3"/>',
    "play": '<path d="M8 5v14l11-7z"/>',
    "refresh": '<path d="M20 12a8 8 0 0 1-14.9 4"/><path d="M4 16h5v5"/><path d="M4 12A8 8 0 0 1 18.9 8"/><path d="M20 8h-5V3"/>',
    "ruler": '<path d="M4 17 17 4l3 3L7 20z"/><path d="m8 13 2 2"/><path d="m11 10 2 2"/><path d="m14 7 2 2"/>',
    "table": '<path d="M4 5h16v14H4z"/><path d="M4 10h16"/><path d="M4 15h16"/><path d="M10 5v14"/>',
    "upload": '<path d="M12 21V9"/><path d="m7 14 5-5 5 5"/><path d="M5 5h14"/>',
    "warning": '<path d="M12 3 2.5 20h19z"/><path d="M12 9v5"/><path d="M12 17h.01"/>',
}


def svg_icon(name: str, css_class: str = "app-icon") -> str:
    """Return a decorative inline SVG icon."""
    paths = SVG_ICONS.get(name, SVG_ICONS["info"])
    return (
        f'<svg class="{escape(css_class)}" viewBox="0 0 24 24" aria-hidden="true" '
        f'focusable="false" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )


def icon_label(icon: str, text: str, css_class: str = "icon-label") -> str:
    return f'<span class="{escape(css_class)}">{svg_icon(icon)}<span>{escape(text)}</span></span>'


def brand_logo_svg() -> str:
    """Return the app logo SVG, falling back to the microscope icon if the asset is unavailable."""
    try:
        logo = LOGO_SVG_PATH.read_text(encoding="utf-8")
    except OSError:
        return svg_icon("microscope")
    return logo.replace("<svg ", '<svg class="brand-logo" aria-hidden="true" focusable="false" ', 1)


def section_heading(step: str, icon: str, title: str, body: str) -> str:
    return f"""
    <div class="section-heading">
      <span class="step-index">{escape(step)}</span>
      <div>
        <h3>{icon_label(icon, title, "section-title-line")}</h3>
        <p>{escape(body)}</p>
      </div>
    </div>
    """


CUSTOM_JS = """
() => {
  const enableMultipleDirectoryPickers = () => {
    document.querySelectorAll('input[type="file"][directory], input[type="file"][mozdirectory], input[type="file"][webkitdirectory]').forEach((input) => {
      input.multiple = true;
      input.webkitdirectory = true;
      input.setAttribute('multiple', '');
      input.setAttribute('webkitdirectory', '');
      input.setAttribute('directory', '');
      input.setAttribute('mozdirectory', '');
    });
  };
  enableMultipleDirectoryPickers();
  new MutationObserver(enableMultipleDirectoryPickers).observe(document.body, { childList: true, subtree: true });
}
"""


CUSTOM_CSS = """
:root {
  color-scheme: light;
  --surface-base: oklch(97% 0.006 205);
  --surface-paper: oklch(99% 0.004 205);
  --surface-panel: oklch(94.5% 0.009 205);
  --surface-muted: oklch(91.5% 0.012 205);
  --surface-inset: oklch(96% 0.007 205);
  --ink: oklch(22% 0.026 215);
  --ink-soft: oklch(34% 0.028 215);
  --muted: oklch(47% 0.024 215);
  --subtle: oklch(58% 0.018 215);
  --border: oklch(83% 0.015 205);
  --border-strong: oklch(72% 0.02 205);
  --primary: oklch(39% 0.082 187);
  --primary-strong: oklch(31% 0.086 187);
  --primary-soft: oklch(88% 0.035 187);
  --primary-faint: oklch(94% 0.02 187);
  --success: oklch(43% 0.092 155);
  --success-soft: oklch(91% 0.035 155);
  --warning: oklch(55% 0.11 75);
  --warning-soft: oklch(93% 0.045 75);
  --danger: oklch(47% 0.12 28);
  --danger-soft: oklch(93% 0.035 28);
  --focus: oklch(58% 0.12 188 / 0.34);
  --shadow-low: 0 1px 2px oklch(22% 0.026 215 / 0.07), 0 12px 30px oklch(22% 0.026 215 / 0.06);
  --shadow-panel: 0 18px 52px oklch(22% 0.026 215 / 0.09);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 18px;
  --radius-xl: 24px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-7: 48px;
  --ease-out: cubic-bezier(0.25, 1, 0.5, 1);
}

html,
body,
.gradio-container {
  background:
    linear-gradient(180deg, oklch(95.5% 0.012 205) 0%, var(--surface-base) 42%, oklch(95% 0.008 205) 100%) !important;
  color: var(--ink) !important;
}

body {
  min-width: 320px;
}

.gradio-container {
  width: 100% !important;
  max-width: 1480px !important;
  min-width: 0 !important;
  margin: 0 auto !important;
  padding: 32px 28px 48px !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif !important;
  font-size: 16px !important;
  line-height: 1.5 !important;
}

.gradio-container main.app,
.gradio-container .app,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .form,
.gradio-container .auto-margin {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
}

.gradio-container main.app {
  padding: 0 !important;
}

.gradio-container .auto-margin {
  margin-right: 0 !important;
  margin-left: 0 !important;
}

.gradio-container * {
  box-sizing: border-box;
}

.app-icon {
  width: 1em;
  height: 1em;
  flex: 0 0 auto;
  color: currentColor;
  vector-effect: non-scaling-stroke;
}

.icon-label,
.section-title-line,
.result-title-line,
.empty-title-line {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.icon-label span,
.section-title-line span,
.result-title-line span,
.empty-title-line span {
  min-width: 0;
}

.section-title-line .app-icon,
.result-title-line .app-icon,
.empty-title-line .app-icon {
  color: var(--primary);
}

.app-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: var(--space-5);
  align-items: end;
  margin-bottom: var(--space-5);
  padding: var(--space-6);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  background: var(--surface-paper);
  box-shadow: var(--shadow-low);
}

.brand-row {
  display: flex;
  align-items: flex-start;
  gap: var(--space-4);
}

.brand-mark {
  display: grid;
  place-items: center;
  width: 76px;
  height: 76px;
  flex: 0 0 auto;
  border-radius: 22px;
}

.brand-logo {
  display: block;
  width: 100%;
  height: 100%;
}

.brand-mark .app-icon {
  width: 38px;
  height: 38px;
  stroke-width: 1.8;
}

.panel-kicker,
.status-eyebrow,
.method-eyebrow {
  margin: 0 0 6px;
  color: var(--primary-strong);
  font-size: 0.72rem;
  font-weight: 760;
  letter-spacing: 0.12em;
  overflow-wrap: anywhere;
  text-transform: uppercase;
}

.app-header h1 {
  margin: 0;
  color: var(--ink);
  font-size: 2.15rem;
  font-weight: 760;
  letter-spacing: -0.035em;
  line-height: 1.05;
  text-wrap: balance;
}

.header-copy {
  margin: var(--space-3) 0 0;
  color: var(--muted);
  font-size: 1rem;
}

.header-links {
  margin: var(--space-3) 0 0;
  color: var(--muted);
  font-size: 0.94rem;
}

.header-links a {
  color: var(--primary-strong);
  font-weight: 700;
  text-decoration: none;
}

.header-links a:hover {
  text-decoration: underline;
}

.workflow-card {
  width: 100% !important;
  max-width: none !important;
  margin: 0 !important;
  padding: var(--space-6) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-xl) !important;
  background: var(--surface-paper) !important;
  box-shadow: var(--shadow-panel) !important;
}

.workflow-intro {
  margin-bottom: var(--space-5);
}

.workflow-intro h2 {
  margin: 0;
  color: var(--ink);
  font-size: 1.35rem;
  font-weight: 740;
  letter-spacing: -0.02em;
  line-height: 1.18;
}

.workflow-intro h2 .app-icon {
  width: 22px;
  height: 22px;
  color: var(--primary);
}

.workflow-intro p:not(.panel-kicker) {
  max-width: 68ch;
  margin: var(--space-2) 0 0;
  color: var(--muted);
  font-size: 0.95rem;
}

.setup-section {
  padding: var(--space-4) 0 !important;
  border-top: 1px solid var(--border) !important;
  background: transparent !important;
  box-shadow: none !important;
}

.setup-section:first-of-type {
  padding-top: 0 !important;
  border-top: 0 !important;
}

.section-heading {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--space-3);
  align-items: start;
  margin-bottom: var(--space-3);
}

.step-index {
  display: inline-grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border: 1px solid var(--border-strong);
  border-radius: 50%;
  background: var(--surface-panel);
  color: var(--ink-soft);
  font-size: 0.78rem;
  font-weight: 780;
  font-variant-numeric: tabular-nums;
}

.section-heading h3 {
  margin: 0;
  color: var(--ink);
  font-size: 1rem;
  font-weight: 740;
  letter-spacing: -0.01em;
}

.section-heading h3 .app-icon {
  width: 18px;
  height: 18px;
  stroke-width: 2.1;
}

.section-heading p {
  margin: 2px 0 0;
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.35;
}

.control-note {
  margin: var(--space-2) 0 0;
  color: var(--ink-soft);
  font-size: 0.82rem;
  line-height: 1.35;
}

.analysis-walkthrough {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.analysis-walkthrough .stepper-wrapper {
  padding-top: var(--space-2) !important;
  padding-bottom: var(--space-6) !important;
}

.analysis-walkthrough .stepper-container {
  padding: 0 !important;
  gap: var(--space-2) !important;
}

.analysis-walkthrough .step-button {
  color: var(--ink-soft) !important;
  opacity: 1 !important;
}

.analysis-walkthrough .step-button:disabled {
  cursor: default !important;
  opacity: 1 !important;
}

.analysis-walkthrough .step-number {
  width: 36px !important;
  height: 36px !important;
  border: 1px solid var(--border-strong) !important;
  background: var(--surface-paper) !important;
  color: var(--ink-soft) !important;
  font-weight: 800 !important;
}

.analysis-walkthrough .step-button.active .step-number,
.analysis-walkthrough .step-button.completed .step-number {
  border-color: var(--primary-strong) !important;
  background: var(--primary) !important;
  color: var(--surface-paper) !important;
}

.analysis-walkthrough .step-button.active .step-number {
  box-shadow: 0 0 0 4px var(--focus) !important;
}

.analysis-walkthrough .step-connector {
  background: var(--border) !important;
  transform: translateY(17px) !important;
}

.analysis-walkthrough .step-connector.completed {
  background: var(--primary-soft) !important;
}

.analysis-walkthrough .step-label.visible {
  color: var(--ink-soft) !important;
  font-weight: 720 !important;
}

.wizard-panel {
  padding-top: var(--space-5) !important;
  border-top: 1px solid var(--border) !important;
}

.wizard-nav {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  justify-content: space-between;
  margin-top: var(--space-4);
}

button.secondary-action,
.secondary-action button {
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius-md) !important;
  background: var(--surface-inset) !important;
  color: var(--ink-soft) !important;
  font-weight: 740 !important;
  box-shadow: none !important;
}

@media (hover: hover) {
  button.secondary-action:hover,
  .secondary-action button:hover {
    border-color: var(--primary) !important;
    background: var(--primary-faint) !important;
    color: var(--primary-strong) !important;
    box-shadow: none !important;
    transform: none !important;
  }
}

.example-preview-group {
  gap: 0 !important;
  margin: var(--space-3) 0 !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background: var(--surface-paper) !important;
  overflow: hidden !important;
}

.example-inline {
  margin: var(--space-3) 0;
  padding: var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface-inset);
}

.example-preview-group .prose,
.example-preview-group .gradio-style {
  background: transparent !important;
}

.example-preview-group .example-inline {
  margin: 0;
  padding: var(--space-3) var(--space-4) var(--space-2);
  border: 0;
  border-radius: 0;
  background: transparent;
}

.example-inline h4 {
  margin: 0 0 6px;
  color: var(--ink);
  font-size: 0.94rem;
  font-weight: 740;
}

.example-inline p,
.example-inline li {
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.35;
}

.example-inline p,
.example-inline ul {
  margin: 0;
}

.example-inline ul {
  padding-left: 18px;
}

.slide-separator-help {
  margin-top: var(--space-3);
}

.slide-separator-help code {
  padding: 1px 5px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-paper);
  color: var(--ink-soft);
  font-size: 0.82rem;
}

.separator-rule {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  align-items: center;
  margin: var(--space-2) 0;
  color: var(--muted);
  font-size: 0.84rem;
}

.separator-examples {
  width: 100%;
  margin: var(--space-2) 0;
  border-collapse: collapse;
  font-size: 0.82rem;
}

.separator-examples th,
.separator-examples td {
  padding: 7px 8px;
  border-top: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}

.separator-examples th {
  color: var(--ink-soft);
  font-weight: 740;
}

.separator-examples td {
  color: var(--muted);
}

.separator-note {
  color: var(--muted);
}

.example-preview {
  margin: var(--space-3) 0 !important;
}

.example-preview-group .example-preview {
  margin: 0 !important;
}

.example-preview-group .example-preview,
.example-preview-group .example-preview .block,
.example-preview-group .example-preview .wrap,
.example-preview-group .example-preview .image-container,
.example-preview-group .example-preview img {
  border-radius: 0 !important;
}

.example-preview-group .example-preview .block {
  border: 0 !important;
  background: var(--surface-paper) !important;
}

.upload-choice-grid {
  align-items: stretch !important;
  gap: var(--space-4) !important;
}

.upload-choice-grid > * {
  min-width: 280px !important;
}

.file-upload-control,
.folder-upload-control {
  height: 100% !important;
}

.file-upload-control,
.file-upload-control .block,
.folder-upload-control,
.folder-upload-control .block {
  border-color: var(--border-strong) !important;
}

.parameter-grid,
.calibration-grid {
  gap: var(--space-3) !important;
}

.run-block {
  padding-top: var(--space-5) !important;
  border-top: 1px solid var(--border) !important;
}

.run-guidance {
  margin: 0 0 var(--space-3);
  color: var(--muted);
  font-size: 0.9rem;
}

.post-run-actions {
  margin-top: var(--space-3) !important;
}

.run-action-row {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
  gap: var(--space-3) !important;
  align-items: stretch !important;
  justify-content: stretch !important;
}

.run-action-row > * {
  width: 100% !important;
  min-width: 0 !important;
}

.run-action-row button,
.run-action-row a {
  width: 100% !important;
}

.run-action-row .download-action {
  margin: 0 !important;
}

.run-action button,
.download-action button,
.download-action a,
.restart-action button,
.secondary-action button,
button.secondary-action {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 8px !important;
}

.run-action button::before,
.download-action button::before,
.download-action a::before,
.restart-action button::before,
.secondary-action:not(.restart-action) button::before,
button.secondary-action:not(.restart-action)::before {
  content: "";
  width: 16px;
  height: 16px;
  flex: 0 0 16px;
  background: currentColor;
  -webkit-mask-position: center;
  mask-position: center;
  -webkit-mask-repeat: no-repeat;
  mask-repeat: no-repeat;
  -webkit-mask-size: contain;
  mask-size: contain;
}

.run-action button::before {
  -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M8 5v14l11-7z'/%3E%3C/svg%3E");
  mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M8 5v14l11-7z'/%3E%3C/svg%3E");
}

.download-action button::before,
.download-action a::before {
  -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' d='M12 3v12m-5-5 5 5 5-5M5 21h14'/%3E%3C/svg%3E");
  mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' d='M12 3v12m-5-5 5 5 5-5M5 21h14'/%3E%3C/svg%3E");
}

.restart-action button::before {
  -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' d='M20 12a8 8 0 0 1-14.9 4M4 16h5v5M4 12A8 8 0 0 1 18.9 8M20 8h-5V3'/%3E%3C/svg%3E");
  mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' d='M20 12a8 8 0 0 1-14.9 4M4 16h5v5M4 12A8 8 0 0 1 18.9 8M20 8h-5V3'/%3E%3C/svg%3E");
}

.secondary-action:not(.restart-action) button::before,
button.secondary-action:not(.restart-action)::before {
  -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' d='M19 12H5m7 7-7-7 7-7'/%3E%3C/svg%3E");
  mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' d='M19 12H5m7 7-7-7 7-7'/%3E%3C/svg%3E");
}

.restart-action button,
button.restart-action {
  min-height: 52px !important;
}

.restart-action button:disabled,
button.restart-action:disabled {
  border-color: var(--border) !important;
  background: var(--surface-muted) !important;
  color: var(--subtle) !important;
  cursor: not-allowed !important;
  box-shadow: none !important;
}

.run-action button,
.run-action .primary,
button.primary {
  min-height: 48px !important;
  border: 1px solid var(--primary-strong) !important;
  border-radius: var(--radius-md) !important;
  background: var(--primary) !important;
  color: var(--surface-paper) !important;
  font-weight: 760 !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 10px 22px oklch(39% 0.082 187 / 0.18) !important;
  transition: background-color 160ms var(--ease-out), box-shadow 160ms var(--ease-out), transform 160ms var(--ease-out) !important;
}

.run-action button:disabled,
.run-action .primary:disabled,
button.primary:disabled {
  border-color: var(--border-strong) !important;
  background: var(--surface-muted) !important;
  color: var(--subtle) !important;
  cursor: not-allowed !important;
  box-shadow: none !important;
  transform: none !important;
}

@media (hover: hover) {
  .run-action button:not(:disabled):hover,
  button.primary:not(:disabled):hover {
    background: var(--primary-strong) !important;
    box-shadow: 0 12px 28px oklch(39% 0.082 187 / 0.22) !important;
    transform: translateY(-1px);
  }
}

button:active {
  transform: translateY(0) !important;
}

label,
label span,
.label-wrap span,
.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap span {
  color: var(--ink-soft) !important;
  font-size: 0.86rem !important;
  font-weight: 700 !important;
  opacity: 1 !important;
}

.info,
.gradio-container .info,
.gradio-container .form .info,
[data-testid="block-info"],
.gradio-container [data-testid="block-info"] {
  color: var(--ink-soft) !important;
  font-size: 0.8rem !important;
  line-height: 1.35 !important;
  font-weight: 680 !important;
  opacity: 1 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  border-color: var(--border) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--surface-paper) !important;
  color: var(--ink) !important;
  font-variant-numeric: tabular-nums;
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px var(--focus) !important;
}

.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container [role="tab"]:focus-visible {
  outline: 3px solid var(--focus) !important;
  outline-offset: 2px !important;
}

.gradio-container .block {
  border-color: var(--border) !important;
  border-radius: var(--radius-md) !important;
}

.upload-control {
  border-radius: var(--radius-lg) !important;
}

.advanced-drawer {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background: var(--surface-inset) !important;
}

.workspace-stack {
  gap: var(--space-4) !important;
}

.method-summary-card,
.method-strip,
.empty-output {
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background: var(--surface-inset);
}

.method-strip {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  align-items: center;
  margin-bottom: var(--space-3);
  padding: var(--space-3);
}

.method-strip strong {
  margin-right: var(--space-1);
  color: var(--ink);
  font-size: 0.86rem;
}

.method-strip span {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 4px 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-paper);
  color: var(--ink-soft);
  font-size: 0.8rem;
  font-weight: 650;
}

.method-summary-card {
  padding: var(--space-5);
  margin-bottom: var(--space-4);
}

.method-summary-card header,
.status-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.method-summary-card h3,
.status-card h3 {
  margin: 0;
  color: var(--ink);
  font-size: 1.05rem;
  font-weight: 740;
  letter-spacing: -0.015em;
}

.method-summary-card p,
.status-card p {
  margin: var(--space-2) 0 0;
  color: var(--muted);
  font-size: 0.92rem;
}

.method-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
  gap: var(--space-3);
}

.method-grid div {
  min-height: 82px;
  padding: var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface-paper);
}

.method-grid dt {
  margin: 0 0 5px;
  color: var(--subtle);
  font-size: 0.72rem;
  font-weight: 760;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.method-grid dd {
  margin: 0;
  color: var(--ink-soft);
  font-size: 0.92rem;
  font-weight: 680;
  line-height: 1.35;
}

.status-card {
  padding: var(--space-4) 0 var(--space-3);
  margin-bottom: var(--space-4);
  border-top: 1px solid var(--border-strong);
  border-bottom: 1px solid var(--border);
  background: transparent;
}

.status-card .status-header {
  margin-bottom: 0;
}

.status-token {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 30px;
  padding: 5px 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-inset);
  color: var(--ink-soft);
  font-size: 0.78rem;
  font-weight: 760;
}

.status-token .app-icon {
  width: 15px;
  height: 15px;
  stroke-width: 2.2;
}

.status-card.ready .status-token {
  border-color: var(--border-strong);
}

.status-card.staged .status-token {
  border-color: oklch(68% 0.07 187);
  background: var(--primary-faint);
  color: var(--primary-strong);
}

.status-card.complete .status-token {
  border-color: oklch(64% 0.08 155);
  background: var(--success-soft);
  color: var(--success);
}

.status-card.warning .status-token {
  border-color: oklch(70% 0.09 75);
  background: var(--warning-soft);
  color: oklch(36% 0.075 75);
}

.status-card.error .status-token {
  border-color: oklch(66% 0.095 28);
  background: var(--danger-soft);
  color: var(--danger);
}

.status-groups {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: var(--space-5);
  margin-top: var(--space-4);
}

.status-group {
  min-width: 0;
}

.status-group h4 {
  margin: 0 0 var(--space-2);
  color: var(--ink);
  font-size: 0.86rem;
  font-weight: 760;
  letter-spacing: -0.005em;
}

.status-group-list {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  border-top: 1px solid var(--border);
}

.status-detail {
  display: grid;
  grid-template-columns: minmax(124px, 0.42fr) minmax(0, 1fr);
  gap: var(--space-3);
  align-items: start;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
}

.status-detail dt {
  color: var(--subtle);
  font-size: 0.72rem;
  font-weight: 760;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.status-detail dd {
  margin: 0;
  color: var(--ink);
  font-size: 0.92rem;
  font-weight: 700;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.status-detail.is-warning dd {
  color: oklch(36% 0.075 75);
}

.status-detail.is-error dd {
  color: var(--danger);
}

.status-list {
  display: grid;
  gap: 0;
  margin: var(--space-3) 0 0;
  padding: 0;
  border-top: 1px solid var(--border);
  list-style: none;
}

.status-list li {
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
  background: transparent;
  color: var(--ink-soft);
  font-size: 0.88rem;
  font-weight: 650;
}

.results-tabs {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background: var(--surface-paper) !important;
  overflow: hidden;
}

.results-tabs [role="tabpanel"],
.results-tabs .tabitem,
.results-tabs .block,
.results-tabs .wrap,
.results-tabs .table-container,
.results-tabs .table-wrap,
.results-tabs table,
.results-tabs thead,
.results-tabs tbody,
.results-tabs tr,
.results-tabs th,
.results-tabs td {
  background: var(--surface-paper) !important;
  color: var(--ink) !important;
  border-color: var(--border) !important;
}

.results-tabs [role="tabpanel"] {
  padding: var(--space-3) !important;
}

.empty-output {
  padding: var(--space-5);
  color: var(--muted);
}

.empty-output h3 {
  margin: 0 0 var(--space-2);
  color: var(--ink);
  font-size: 1rem;
}

.empty-output h3 .app-icon {
  width: 18px;
  height: 18px;
}

.empty-output p {
  margin: 0;
  max-width: 56ch;
  color: var(--muted);
}

.results-tabs button[role="tab"] {
  color: var(--muted) !important;
  font-weight: 720 !important;
}

.results-tabs button[role="tab"][aria-selected="true"] {
  color: var(--primary-strong) !important;
}

.gradio-container table,
.gradio-container .dataframe {
  font-size: 0.86rem !important;
  font-variant-numeric: tabular-nums !important;
}

.gradio-container th {
  color: var(--ink-soft) !important;
  font-weight: 760 !important;
}

.results-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface-paper);
}

.field-summary-table {
  max-height: 420px;
  overflow: auto;
}

.table-truncation-note {
  margin: 0 0 var(--space-2);
  color: var(--muted);
  font-size: 0.88rem;
  font-weight: 650;
}

.results-table .col-slide-id,
.results-table .col-slide {
  min-width: 240px;
  width: 240px;
}

.results-table .col-field-id,
.results-table .col-field {
  min-width: 180px;
  width: 180px;
}

.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
  font-variant-numeric: tabular-nums;
}

.results-table th,
.results-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}

.results-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--surface-panel);
  color: var(--ink-soft);
  font-weight: 760;
  white-space: nowrap;
}

.results-table tr:last-child td {
  border-bottom: 0;
}

.results-table td {
  color: var(--ink);
  overflow-wrap: anywhere;
}

.gallery-panel {
  background: var(--surface-inset) !important;
}

.qc-preview-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: var(--space-3);
  margin: 0;
  padding: 0;
  list-style: none;
}

.qc-preview-item {
  display: block;
  height: 100%;
  padding: var(--space-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface-inset);
  color: var(--ink);
  transition: border-color 140ms var(--ease-out), box-shadow 140ms var(--ease-out), transform 140ms var(--ease-out);
}

.qc-preview-item:hover {
  border-color: var(--primary);
  box-shadow: var(--shadow-low);
  transform: translateY(-1px);
}

.qc-preview-summary {
  display: grid;
  gap: var(--space-2);
  align-items: start;
  cursor: zoom-in;
  list-style: none;
}

.qc-preview-summary::-webkit-details-marker {
  display: none;
}

.qc-preview-item[open] .qc-preview-summary {
  margin-bottom: var(--space-2);
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--border);
}

.qc-preview-thumb,
.qc-preview-large {
  display: block;
  width: 100%;
  object-fit: contain;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: oklch(99% 0.004 205);
}

.qc-preview-thumb {
  aspect-ratio: 5 / 3;
}

.qc-preview-large {
  max-height: 70vh;
}

.qc-preview-name {
  display: block;
  color: var(--ink);
  font-size: 0.88rem;
  font-weight: 760;
  line-height: 1.2;
  overflow-wrap: anywhere;
}

.qc-preview-action,
.qc-preview-note {
  display: block;
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.25;
}

.qc-preview-expanded {
  display: grid;
  gap: var(--space-2);
}

.results-stack {
  gap: var(--space-4) !important;
  padding-top: var(--space-4) !important;
}

.results-section-title {
  margin: var(--space-4) 0 var(--space-2);
  padding-top: var(--space-3);
  border-top: 1px solid var(--border);
  color: var(--ink);
  font-size: 1rem;
  font-weight: 760;
  letter-spacing: -0.01em;
}

.results-section-title .app-icon {
  width: 18px;
  height: 18px;
}

.results-section-title:first-child {
  margin-top: 0;
  padding-top: 0;
  border-top: 0;
}

.download-action {
  margin: var(--space-3) 0 var(--space-4) !important;
}

button.primary.download-action,
a.primary.download-action,
button.download-action,
a.download-action,
.download-action button,
.download-action a {
  min-height: 52px !important;
  padding: 0 22px !important;
  border: 1px solid var(--primary) !important;
  border-radius: var(--radius-md) !important;
  background: var(--surface-paper) !important;
  color: var(--primary-strong) !important;
  font-weight: 800 !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 10px 22px oklch(39% 0.082 187 / 0.12) !important;
}

button.primary.download-action:not(:disabled):hover,
a.primary.download-action:hover,
button.download-action:not(:disabled):hover,
a.download-action:hover,
.download-action button:not(:disabled):hover,
.download-action a:hover {
  border-color: var(--primary-strong) !important;
  background: var(--primary-faint) !important;
  color: var(--primary-strong) !important;
  box-shadow: 0 12px 28px oklch(39% 0.082 187 / 0.18) !important;
  transform: translateY(-1px);
}

.run-action-row .run-action button,
.run-action-row .download-action button,
.run-action-row .download-action a,
.run-action-row .restart-action button,
.run-action-row button.secondary-action,
.run-action-row .secondary-action button {
  min-height: 52px !important;
  height: 52px !important;
  padding: 0 22px !important;
  margin: 0 !important;
  line-height: 1.1 !important;
}

input[type="number"],
input[type="text"],
textarea,
select,
[data-testid="textbox"] {
  min-height: 42px !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius-sm) !important;
  background: oklch(97.5% 0.006 205) !important;
  color: var(--ink) !important;
  box-shadow: inset 0 1px 0 oklch(100% 0 0 / 0.72) !important;
}

input[type="number"]:hover,
input[type="text"]:hover,
textarea:hover,
select:hover,
[data-testid="textbox"]:hover {
  border-color: var(--primary) !important;
  background: var(--surface-paper) !important;
}

label[data-testid$="-radio-label"],
label[data-testid$="-checkbox-label"] {
  border: 1px solid var(--border) !important;
  background: oklch(97.5% 0.006 205) !important;
}

label[data-testid$="-radio-label"].selected,
label[data-testid$="-checkbox-label"].selected,
label[data-testid$="-radio-label"]:has(input[type="radio"]:checked),
label[data-testid$="-checkbox-label"]:has(input[type="checkbox"]:checked) {
  border-color: oklch(78% 0.035 187) !important;
  background: var(--primary-faint) !important;
  color: var(--primary-strong) !important;
}

label[data-testid$="-radio-label"] input[type="radio"],
label[data-testid$="-checkbox-label"] input[type="checkbox"],
.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"] {
  appearance: none !important;
  display: inline-grid !important;
  place-items: center !important;
  width: 16px !important;
  height: 16px !important;
  flex: 0 0 16px !important;
  margin: 0 !important;
  border: 1.5px solid var(--border-strong) !important;
  background: var(--surface-paper) !important;
  box-shadow: inset 0 1px 0 oklch(100% 0 0 / 0.72) !important;
}

label[data-testid$="-radio-label"] input[type="radio"],
.gradio-container input[type="radio"] {
  border-radius: 50% !important;
}

label[data-testid$="-checkbox-label"] input[type="checkbox"],
.gradio-container input[type="checkbox"] {
  border-radius: 4px !important;
}

label[data-testid$="-radio-label"] input[type="radio"]:checked,
.gradio-container input[type="radio"]:checked {
  border-color: var(--primary) !important;
  background:
    radial-gradient(circle at center, var(--primary) 0 43%, transparent 46%),
    var(--surface-paper) !important;
}

label[data-testid$="-checkbox-label"] input[type="checkbox"]:checked,
.gradio-container input[type="checkbox"]:checked {
  border-color: var(--primary) !important;
  background: var(--primary) !important;
}

label[data-testid$="-checkbox-label"] input[type="checkbox"]:checked::after,
.gradio-container input[type="checkbox"]:checked::after {
  content: "";
  width: 8px;
  height: 5px;
  border: solid var(--surface-paper);
  border-width: 0 0 2px 2px;
  transform: rotate(-45deg) translateY(-1px);
}

.gradio-container label.checkbox-container {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  width: 100% !important;
  min-height: 46px !important;
  padding: 10px 12px !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius-sm) !important;
  background: oklch(97.5% 0.006 205) !important;
  color: var(--ink) !important;
  box-shadow: inset 0 1px 0 oklch(100% 0 0 / 0.7) !important;
  cursor: pointer !important;
  transition: background 160ms ease-out, border-color 160ms ease-out, box-shadow 160ms ease-out;
}

.gradio-container label.checkbox-container:hover {
  border-color: oklch(61% 0.052 187) !important;
  background: var(--surface-paper) !important;
}

.gradio-container label.checkbox-container:focus-within {
  border-color: var(--primary) !important;
  outline: 3px solid oklch(39% 0.082 187 / 0.18) !important;
  outline-offset: 2px !important;
}

.gradio-container label.checkbox-container:has(input[type="checkbox"]:checked) {
  border-color: oklch(67% 0.052 187) !important;
  background: var(--primary-faint) !important;
  color: var(--ink) !important;
  box-shadow:
    inset 0 0 0 1px oklch(78% 0.035 187 / 0.55),
    0 8px 18px oklch(39% 0.082 187 / 0.08) !important;
}

.gradio-container label.checkbox-container .label-text {
  color: inherit !important;
  font-weight: 750 !important;
  letter-spacing: 0.005em !important;
}

.gradio-container label.checkbox-container:has(input[type="checkbox"]:checked) .label-text {
  color: var(--primary-strong) !important;
}

.gradio-container label.checkbox-container input[type="checkbox"] {
  -webkit-appearance: none !important;
  appearance: none !important;
  display: inline-grid !important;
  place-items: center !important;
  box-sizing: border-box !important;
  width: 18px !important;
  min-width: 18px !important;
  height: 18px !important;
  min-height: 18px !important;
  flex: 0 0 18px !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 1.5px solid var(--border-strong) !important;
  border-radius: 5px !important;
  background-color: var(--surface-paper) !important;
  background-image: none !important;
  background-position: center !important;
  background-repeat: no-repeat !important;
  background-size: 0 0 !important;
  box-shadow: inset 0 1px 0 oklch(100% 0 0 / 0.72) !important;
  color: transparent !important;
  accent-color: var(--primary) !important;
}

.gradio-container label.checkbox-container input[type="checkbox"]:checked {
  border-color: var(--primary-strong) !important;
  background-color: var(--primary) !important;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 9'%3E%3Cpath d='M1 4.6 4.4 8 11 1' fill='none' stroke='%23f8fcfb' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
  background-size: 12px 9px !important;
  box-shadow: 0 0 0 2px oklch(39% 0.082 187 / 0.14) !important;
}

.gradio-container label.checkbox-container input[type="checkbox"]:checked::before,
.gradio-container label.checkbox-container input[type="checkbox"]:checked::after {
  content: none !important;
  display: none !important;
}

.gradio-container label.checkbox-container svg,
.gradio-container label.checkbox-container svg * {
  color: var(--surface-paper) !important;
  stroke: var(--surface-paper) !important;
}

@media (max-width: 980px) {
  .gradio-container {
    padding: 22px 16px 36px !important;
  }

  .app-header {
    grid-template-columns: 1fr;
    padding: var(--space-5);
  }

  .app-header h1 {
    font-size: 1.74rem;
  }

  .workflow-card {
    padding: var(--space-5) !important;
  }
}

@media (max-width: 680px) {
  .brand-row {
    flex-direction: column;
  }

  .brand-mark {
    width: 64px;
    height: 64px;
  }

  .app-header,
  .workflow-card,
  .wizard-panel {
    overflow: hidden;
  }

  .workflow-card {
    padding: var(--space-4) !important;
    border-radius: var(--radius-lg) !important;
  }

  .analysis-walkthrough .stepper-wrapper {
    padding-bottom: var(--space-5) !important;
  }

  .analysis-walkthrough .stepper-container {
    gap: 0 !important;
  }

  .analysis-walkthrough .step-label.visible {
    display: none !important;
  }

  .upload-choice-grid,
  .calibration-grid,
  .parameter-grid {
    flex-direction: column !important;
  }

  .upload-choice-grid > *,
  .calibration-grid > *,
  .parameter-grid > * {
    width: 100% !important;
    min-width: 0 !important;
  }

  .method-grid,
  .status-groups,
  .status-list {
    grid-template-columns: 1fr;
  }

  .status-detail {
    grid-template-columns: 1fr;
    gap: 4px;
  }
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}
"""


CLINICAL_THEME = gr.themes.Base(primary_hue="teal", neutral_hue="slate").set(
    body_background_fill="oklch(97% 0.006 205)",
    body_background_fill_dark="oklch(97% 0.006 205)",
    body_text_color="oklch(22% 0.026 215)",
    body_text_color_dark="oklch(22% 0.026 215)",
    body_text_color_subdued="oklch(47% 0.024 215)",
    body_text_color_subdued_dark="oklch(47% 0.024 215)",
    background_fill_primary="oklch(99% 0.004 205)",
    background_fill_primary_dark="oklch(99% 0.004 205)",
    background_fill_secondary="oklch(96% 0.007 205)",
    background_fill_secondary_dark="oklch(96% 0.007 205)",
    block_background_fill="oklch(99% 0.004 205)",
    block_background_fill_dark="oklch(99% 0.004 205)",
    block_border_color="oklch(83% 0.015 205)",
    block_border_color_dark="oklch(83% 0.015 205)",
    block_info_text_color="oklch(58% 0.018 215)",
    block_info_text_color_dark="oklch(58% 0.018 215)",
    block_label_background_fill="oklch(94.5% 0.009 205)",
    block_label_background_fill_dark="oklch(94.5% 0.009 205)",
    block_label_text_color="oklch(34% 0.028 215)",
    block_label_text_color_dark="oklch(34% 0.028 215)",
    block_radius="12px",
    block_shadow="none",
    block_shadow_dark="none",
    panel_background_fill="oklch(99% 0.004 205)",
    panel_background_fill_dark="oklch(99% 0.004 205)",
    panel_border_color="oklch(83% 0.015 205)",
    panel_border_color_dark="oklch(83% 0.015 205)",
    input_background_fill="oklch(99% 0.004 205)",
    input_background_fill_dark="oklch(99% 0.004 205)",
    input_background_fill_focus="oklch(99% 0.004 205)",
    input_background_fill_focus_dark="oklch(99% 0.004 205)",
    input_border_color="oklch(83% 0.015 205)",
    input_border_color_dark="oklch(83% 0.015 205)",
    input_border_color_focus="oklch(39% 0.082 187)",
    input_border_color_focus_dark="oklch(39% 0.082 187)",
    input_placeholder_color="oklch(58% 0.018 215)",
    input_placeholder_color_dark="oklch(58% 0.018 215)",
    input_radius="8px",
    checkbox_label_background_fill="oklch(99% 0.004 205)",
    checkbox_label_background_fill_dark="oklch(99% 0.004 205)",
    checkbox_label_background_fill_selected="oklch(94% 0.02 187)",
    checkbox_label_background_fill_selected_dark="oklch(94% 0.02 187)",
    checkbox_label_text_color="oklch(34% 0.028 215)",
    checkbox_label_text_color_dark="oklch(34% 0.028 215)",
    checkbox_label_text_color_selected="oklch(31% 0.086 187)",
    checkbox_label_text_color_selected_dark="oklch(31% 0.086 187)",
    checkbox_border_color="oklch(72% 0.02 205)",
    checkbox_border_color_dark="oklch(72% 0.02 205)",
    checkbox_border_color_selected="oklch(39% 0.082 187)",
    checkbox_border_color_selected_dark="oklch(39% 0.082 187)",
    button_primary_background_fill="oklch(39% 0.082 187)",
    button_primary_background_fill_dark="oklch(39% 0.082 187)",
    button_primary_background_fill_hover="oklch(31% 0.086 187)",
    button_primary_background_fill_hover_dark="oklch(31% 0.086 187)",
    button_primary_border_color="oklch(31% 0.086 187)",
    button_primary_border_color_dark="oklch(31% 0.086 187)",
    button_primary_text_color="oklch(99% 0.004 205)",
    button_primary_text_color_dark="oklch(99% 0.004 205)",
    button_secondary_background_fill="oklch(96% 0.007 205)",
    button_secondary_background_fill_dark="oklch(96% 0.007 205)",
    button_secondary_text_color="oklch(34% 0.028 215)",
    button_secondary_text_color_dark="oklch(34% 0.028 215)",
    table_border_color="oklch(83% 0.015 205)",
    table_border_color_dark="oklch(83% 0.015 205)",
    table_even_background_fill="oklch(99% 0.004 205)",
    table_even_background_fill_dark="oklch(99% 0.004 205)",
    table_odd_background_fill="oklch(96% 0.007 205)",
    table_odd_background_fill_dark="oklch(96% 0.007 205)",
)




def render_grid_strategy_inputs(grid_strategy_choice: str):
    strategy = ui_validation.strategy_value(grid_strategy_choice)
    return gr.update(visible=strategy == "count"), gr.update(visible=strategy == "spacing")



def _status_detail_class(label: str) -> str:
    lower_label = label.lower()
    if "error" in lower_label:
        return "status-detail is-error"
    if "warning" in lower_label or "default calibration" in lower_label:
        return "status-detail is-warning"
    return "status-detail"


def _status_groups_html(groups: list[tuple[str, list[tuple[str, object]]]] | None) -> str:
    group_sections = []
    for heading, rows in groups or []:
        details = []
        for label, value in rows:
            label_text = str(label)
            details.append(
                f'<div class="{_status_detail_class(label_text)}">'
                f"<dt>{escape(label_text)}</dt>"
                f"<dd>{escape(str(value))}</dd>"
                "</div>"
            )
        if not details:
            continue
        group_sections.append(
            '<section class="status-group">'
            f"<h4>{escape(str(heading))}</h4>"
            f'<dl class="status-group-list">{"".join(details)}</dl>'
            "</section>"
        )
    if not group_sections:
        return ""
    return f'<div class="status-groups">{"".join(group_sections)}</div>'


def _run_status_groups(
    ok_count: int,
    uploaded_count: int,
    warning_count: int,
    error_count: int,
    calibration_warnings: list[str],
    default_calibration_count: int,
    qc_warnings: list[str] | None = None,
) -> list[tuple[str, list[tuple[str, object]]]]:
    processing_rows: list[tuple[str, object]] = [("Fields processed", f"{ok_count} of {uploaded_count}")]
    if warning_count:
        processing_rows.append(("Warnings", warning_count))
    if error_count:
        processing_rows.append(("Errors", error_count))

    groups: list[tuple[str, list[tuple[str, object]]]] = [("Processing", processing_rows)]

    calibration_rows: list[tuple[str, object]] = []
    if calibration_warnings:
        calibration_rows.append(("Calibration warning", calibration_warnings[0]))
    elif default_calibration_count:
        calibration_rows.append(("Default calibration fields", f"{default_calibration_count} of {uploaded_count}"))
    if calibration_rows:
        groups.append(("Calibration", calibration_rows))

    qc_warning_rows = [("Warning", qc_warnings[0])] if qc_warnings else []
    if qc_warning_rows:
        groups.append(("QC warnings", qc_warning_rows))

    return groups


def _status_panel(
    kind: str,
    token: str,
    title: str,
    body: str,
    items: list[str] | None = None,
    groups: list[tuple[str, list[tuple[str, object]]]] | None = None,
) -> str:
    list_items = "".join(f"<li>{escape(item)}</li>" for item in (items or []))
    item_list = f'<ul class="status-list">{list_items}</ul>' if list_items else ""
    detail_html = _status_groups_html(groups) or item_list
    role = "alert" if kind == "error" else "status"
    aria_live = "assertive" if kind in {"error", "warning"} else "polite"
    status_icon = {
        "ready": "info",
        "staged": "upload",
        "complete": "check",
        "warning": "warning",
        "error": "error",
    }.get(kind, "info")
    return f"""
    <section class="status-card {escape(kind)}" role="{role}" aria-live="{aria_live}">
      <div class="status-header">
        <div>
          <p class="status-eyebrow">Run status</p>
          <h3>{escape(title)}</h3>
          <p>{escape(body)}</p>
        </div>
        <span class="status-token">{svg_icon(status_icon)}{escape(token)}</span>
      </div>
      {detail_html}
    </section>
    """



def _render_file_state(uploaded: list[str], validation_errors: list[str] | None = None) -> str:
    if not uploaded:
        return _status_panel(
            "ready",
            "Ready",
            "Awaiting images",
            "Upload cropped ROI fields or folders, confirm calibration, then run analysis.",
        )
    if validation_errors:
        return _status_panel(
            "warning",
            "Review upload",
            "Upload limits need attention",
            "Remove oversized or unreadable images, then run the batch again.",
            ui_validation.display_upload_validation_errors(validation_errors),
        )

    file_word = "item" if len(uploaded) == 1 else "items"
    return _status_panel(
        "staged",
        "Images staged",
        f"{len(uploaded)} uploaded {file_word} ready",
        "Review the method strip and run the batch.",
    )


def _validated_upload_state(files, folders=None) -> tuple[list[str], list[str]]:
    uploaded = ui_validation.combined_uploads(files, folders)
    validation_errors = ui_validation.upload_validation_errors(uploaded) if uploaded else []
    return uploaded, validation_errors


def render_file_state(files, folders=None) -> str:
    uploaded, validation_errors = _validated_upload_state(files, folders)
    return _render_file_state(uploaded, validation_errors)


def render_upload_state(files, folders):
    uploaded, validation_errors = _validated_upload_state(files, folders)
    enabled = bool(uploaded) and not validation_errors
    return (
        _render_file_state(uploaded, validation_errors),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(visible=True, interactive=False),
    )


def show_wizard_step(active_step: int):
    return gr.Walkthrough(selected=active_step)


def _wire_live_update(components, *, fn, inputs, outputs) -> None:
    for component in components:
        component.change(
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            queue=False,
            show_progress="hidden",
        )


def _wire_wizard_navigation(wizard, step_targets) -> None:
    for button, step in step_targets:
        button.click(
            fn=lambda step=step: show_wizard_step(step),
            outputs=wizard,
            queue=False,
            show_progress="hidden",
        )


def render_example_panel() -> str:
    dimensions = (
        "unknown dimensions"
        if ui_config.EXAMPLE_IMAGE_DIMS is None
        else f"{ui_config.EXAMPLE_IMAGE_DIMS[0]} × {ui_config.EXAMPLE_IMAGE_DIMS[1]} px"
    )
    return f"""
    <section class="example-inline" aria-label="Example input reference">
      <h4>{icon_label("image", "Example reference")}</h4>
      <ul>
        <li>Image: <strong>{escape(ui_config.EXAMPLE_FILENAME)}</strong> ({escape(dimensions)})</li>
        <li>Calibration: {ui_config.EXAMPLE_PIXEL_WIDTH_UM:g} × {ui_config.EXAMPLE_PIXEL_HEIGHT_UM:g} µm/px</li>
      </ul>
    </section>
    """


def _display_separator(separator: str) -> str:
    labels = {
        "_": "underscore (_)",
        "-": "hyphen (-)",
        ".": "period (.)",
        " ": "space",
        "\t": "tab",
    }
    return labels.get(separator, separator)


def _separator_code(separator: str) -> str:
    normalized = separator or "_"
    visible = _display_separator(normalized)
    return f"<code>{escape(visible)}</code>"


def render_slide_separator_examples(slide_roi_separator: str = "_") -> str:
    separator = slide_roi_separator or "_"
    examples = [
        f"MouseA{separator}0001.tif",
        f"MouseA{separator}0002.tif",
        f"MouseA_left_lung{separator}0003.tif",
    ]
    rows = []
    for filename in examples:
        slide_id = ui_validation.infer_slide_id(filename, separator)
        field_id = ui_validation.infer_field_id(filename, separator)
        rows.append(
            "<tr>"
            f"<td><code>{escape(filename)}</code></td>"
            f"<td><code>{escape(slide_id)}</code></td>"
            f"<td><code>{escape(field_id)}</code></td>"
            "</tr>"
        )
    return f"""
    <section class="example-inline slide-separator-help" aria-label="Slide separator examples">
      <h4>{icon_label("info", "Slide separator examples")}</h4>
      <p>The separator is the character or text between the slide/specimen ID and the field/ROI ID. The app splits the filename stem at the final occurrence, so earlier separator characters can stay inside the slide ID.</p>
      <div class="separator-rule"><span>Current separator:</span> {_separator_code(separator)}</div>
      <table class="separator-examples" aria-label="Filename parsing examples">
        <thead><tr><th scope="col">Filename</th><th scope="col">Slide used for grouping</th><th scope="col">Field ID</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="separator-note">Files with the same Slide value are combined in one slide summary. For names like <code>MouseA-field03.tif</code>, enter <code>-</code>. If the separator is not found, the whole filename stem becomes the Slide value and the Field is recorded as <code>field</code>.</p>
    </section>
    """


def render_line_protocol_example() -> str:
    return f"""
    <section class="example-inline" aria-label="Line protocol example reference">
      <h4>{icon_label("grid", "Line protocol example")}</h4>
      <p>Blue lines are one-pixel test lines. Amber segments are airspace chords used for MLI: each segment is one continuous stretch of airspace along a test line. Magenta segments are non-airspace chords; they are line-sampling measurements, not wall-thickness measurements.</p>
    </section>
    """


def empty_output_panel(title: str, body: str) -> str:
    icon = "image" if "preview" in title.lower() else "table"
    return f"""
    <section class="empty-output">
      <h3>{icon_label(icon, title, "empty-title-line")}</h3>
      <p>{escape(body)}</p>
    </section>
    """


def _format_table_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):,.4g}"
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return ""
    return str(value)



def _column_css_class(column: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(column).lower()).strip("-")
    return f"col-{slug or 'value'}"


DISPLAY_METADATA_LABELS = {
    "not_recorded": "Not recorded",
    DEFAULT_CALIBRATION_SOURCE: "Built-in default: 0.57 × 0.57 µm/px",
    UNRECORDED_CALIBRATION_SOURCE: "Manual pixel size; source not recorded",
}


def _display_metadata_text(value, fallback: str = "Not recorded") -> str:
    text = ui_validation.metadata_text(value, fallback)
    return DISPLAY_METADATA_LABELS.get(text, text)


DISPLAY_COLUMN_LABELS = {
    "slide_id": "Slide",
    "field_id": "Field",
    "status": "Status",
    "message": "Message",
    "image_load_warnings": "Image warning",
    "mli_orientation_balanced_mean_um": "MLI (µm)",
    "mli_chord_count": "MLI chords",
    "mean_non_airspace_chord_um": "Non-airspace chord (µm)",
    "non_airspace_chord_count": "Non-airspace chords",
    "airspace_fraction": "All airspace fraction",
    "non_edge_airspace_fraction": "Non-edge airspace fraction",
    "non_edge_airspace_component_area_um2_mask": "Non-edge airspace area (µm²)",
    "field_count": "Fields",
    "mli_measured_field_count": "MLI measured fields",
    "field_balanced_mean_mli_um": "Mean MLI (µm)",
    "field_sem_mli_um": "MLI SEM (µm)",
    "chord_pooled_mean_mli_um": "Chord-pooled MLI (µm)",
    "total_mli_chords": "MLI chords",
    "non_airspace_measured_field_count": "Non-airspace measured fields",
    "field_balanced_mean_non_airspace_chord_um": "Mean non-airspace chord (µm)",
    "field_sd_non_airspace_chord_um": "Non-airspace SD (µm)",
    "field_sem_non_airspace_chord_um": "Non-airspace SEM (µm)",
    "total_non_airspace_chords": "Non-airspace chords",
    "mean_airspace_fraction": "All airspace fraction",
    "mean_non_edge_airspace_fraction": "Non-edge airspace fraction",
    "mean_non_edge_airspace_component_area_um2_mask": "Non-edge airspace area (µm²)",
}


def _friendly_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename selected result columns for the compact on-screen preview."""
    return frame.rename(columns={column: DISPLAY_COLUMN_LABELS.get(column, column) for column in frame.columns})


def render_dataframe_table(
    frame: pd.DataFrame,
    empty_title: str,
    empty_body: str,
    extra_class: str = "",
    total_rows: int | None = None,
) -> str:
    """Render result tables as static HTML to avoid the heavy Gradio Dataframe UI."""
    if frame.empty:
        return empty_output_panel(empty_title, empty_body)

    classes = "results-table-wrap"
    if extra_class:
        classes = f"{classes} {extra_class}"
    shown_rows = len(frame)
    full_row_count = shown_rows if total_rows is None else max(int(total_rows), shown_rows)
    truncation_note = (
        ""
        if full_row_count <= shown_rows
        else f'<p class="table-truncation-note">Showing first {shown_rows:,} of {full_row_count:,} rows. '
        "Download the results ZIP for the complete table.</p>"
    )
    headers = "".join(
        f'<th class="{escape(_column_css_class(column))}">{escape(str(column))}</th>' for column in frame.columns
    )
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(
            f'<td class="{escape(_column_css_class(column))}">{escape(_format_table_value(row[column]))}</td>'
            for column in frame.columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"""
    {truncation_note}
    <div class="{escape(classes)}" role="region" aria-label="{escape(empty_title)}" tabindex="0">
      <table class="results-table">
        <thead><tr>{headers}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """




def _export_root_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    export_root = ui_config.EXPORT_ROOT.resolve(strict=False)
    if not ui_config.path_is_relative_to(resolved, export_root):
        raise ValueError(f"Refusing to read file outside export root: {resolved}")
    return resolved


def _qc_preview_data_uri(path: str | Path) -> str:
    resolved = _export_root_file(path)
    suffix = resolved.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_qc_previews(paths: list[Path]) -> str:
    items = []
    for index, path in enumerate(paths):
        uri = _qc_preview_data_uri(path)
        label = Path(path).parent.name
        loading = "eager" if index == 0 else "lazy"
        items.append(
            f"""
            <li>
              <details class="qc-preview-item">
                <summary class="qc-preview-summary" aria-label="Show larger QC preview for {escape(label)}">
                  <img class="qc-preview-thumb" src="{escape(uri)}" alt="QC preview thumbnail for {escape(label)}" loading="{loading}" decoding="async">
                  <strong class="qc-preview-name">{escape(label)}</strong>
                  <span class="qc-preview-action">Click to expand inline</span>
                </summary>
                <div class="qc-preview-expanded">
                  <img class="qc-preview-large" src="{escape(uri)}" alt="Larger QC preview for {escape(label)}" loading="lazy" decoding="async">
                  <span class="qc-preview-note">Full-resolution QC files are included in the results ZIP under <code>{escape(label)}/qc_panel.png</code>.</span>
                </div>
              </details>
            </li>
            """
        )
    return f'<ul class="qc-preview-list">{"".join(items)}</ul>'


def _request_session_hash(request: gr.Request | None) -> str | None:
    if request is None:
        return None
    session_hash = getattr(request, "session_hash", None)
    return str(session_hash) if session_hash else None


def _register_session_export(run_dir: str | Path, request: gr.Request | None) -> None:
    session_hash = _request_session_hash(request)
    if session_hash is None:
        return
    with _SESSION_EXPORT_LOCK:
        _SESSION_EXPORT_RUNS.setdefault(session_hash, set()).add(str(run_dir))


def _unregister_session_export(run_dir: str | Path | None, request: gr.Request | None) -> None:
    session_hash = _request_session_hash(request)
    if session_hash is None or not run_dir:
        return
    with _SESSION_EXPORT_LOCK:
        run_dirs = _SESSION_EXPORT_RUNS.get(session_hash)
        if run_dirs is None:
            return
        run_dirs.discard(str(run_dir))
        if not run_dirs:
            _SESSION_EXPORT_RUNS.pop(session_hash, None)


def cleanup_session_exports(request: gr.Request) -> None:
    session_hash = _request_session_hash(request)
    if session_hash is None:
        return
    with _SESSION_EXPORT_LOCK:
        run_dirs = _SESSION_EXPORT_RUNS.pop(session_hash, set())
    for run_dir in run_dirs:
        ui_config.remove_export_run(run_dir)


def _empty_run_outputs(status_html: str):
    return (
        status_html,
        gr.update(visible=True),
        gr.update(value="", visible=False),
        gr.update(visible=True),
        gr.update(value="", visible=False),
        gr.update(visible=True),
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
        gr.update(visible=True, interactive=False, value="Analyze new dataset"),
        None,
    )


def reset_analysis_workflow(current_run_dir=None, request: gr.Request | None = None):
    ui_config.remove_export_run(current_run_dir)
    _unregister_session_export(current_run_dir, request)
    empty_outputs = _empty_run_outputs(render_file_state(None, None))
    return (
        show_wizard_step(1),
        gr.update(value=None),
        gr.update(value=None),
        empty_outputs[0],
        gr.update(interactive=False),
        gr.update(interactive=False),
        *empty_outputs[1:],
    )


def render_method_summary(
    pixel_width_um: float,
    pixel_height_um: float,
    grid_strategy_choice: str,
    num_lines: int,
    line_spacing_um: float,
    orientation: str,
    threshold_method: str,
    airspace_bright: bool,
    airspace_component_connectivity,
    min_chord_um: float,
    slide_roi_separator: str,
    field_selection_method: str,
    field_exclusion_criteria: str,
    grid_random_offset: bool = False,
    grid_random_seed=None,
    calibration_source: str = DEFAULT_CALIBRATION_SOURCE,
) -> str:
    pixel_width = ui_validation.safe_float(pixel_width_um, 0.57)
    pixel_height = ui_validation.safe_float(pixel_height_um, 0.57)
    line_count = max(ui_validation.safe_int(num_lines, 15), 1)
    spacing = max(ui_validation.safe_float(line_spacing_um, 35.4), 0.01)
    min_chord = max(ui_validation.safe_float(min_chord_um, 0.0), 0.0)

    strategy = ui_validation.strategy_value(grid_strategy_choice)
    threshold = (threshold_method or "huang").replace("_", " ").title()
    orientation_label = {
        "both": "Both orientations",
        "horizontal": "Horizontal",
        "vertical": "Vertical",
    }.get((orientation or "both").lower(), escape(str(orientation)))
    grid_label = f"{spacing:.2f} µm spacing" if strategy == "spacing" else f"{line_count} lines"
    seed = ui_validation.optional_int(grid_random_seed)
    if bool(grid_random_offset):
        grid_phase_label = (
            f"random grid phase seed {seed}" if seed is not None else "random grid phase auto-seed"
        )
    else:
        grid_phase_label = "centered grid phase"
    phase_label = "bright airspaces" if ui_validation.airspace_bright_value(airspace_bright) else "dark airspaces"
    connectivity = ui_validation.component_connectivity_value(airspace_component_connectivity)
    connectivity_label = f"{connectivity}-connected components"
    edge_label = "edge excluded"
    min_label = "no min" if min_chord == 0 else f"min {min_chord:.2f} µm"
    non_airspace_label = "MLI + non-airspace chords"
    separator = slide_roi_separator or "_"
    selection_label = _display_metadata_text(field_selection_method)
    field_exclusion_label = _display_metadata_text(field_exclusion_criteria)
    calibration_source_label = ui_validation.metadata_text(calibration_source, DEFAULT_CALIBRATION_SOURCE)
    if (
        calibration_source_label == DEFAULT_CALIBRATION_SOURCE
        and (
            abs(pixel_width - DEFAULT_PIXEL_WIDTH_UM) > 1e-12
            or abs(pixel_height - DEFAULT_PIXEL_HEIGHT_UM) > 1e-12
        )
    ):
        calibration_source_label = UNRECORDED_CALIBRATION_SOURCE
    calibration_source_label = _display_metadata_text(calibration_source_label)

    chips = [
        "input: pre-cropped ROI",
        f"selection: {selection_label}",
        f"field excl: {field_exclusion_label}",
        f"{pixel_width:.4g} × {pixel_height:.4g} µm/px",
        f"cal: {calibration_source_label}",
        grid_label,
        grid_phase_label,
        orientation_label,
        f"{threshold}, {phase_label}",
        connectivity_label,
        f"{edge_label}, {min_label}",
        non_airspace_label,
        "slide summaries: field-balanced + chord-pooled",
        f"filename separator: {separator!r}",
    ]
    chip_markup = "".join(f"<span>{escape(chip)}</span>" for chip in chips)
    return f"""
    <section class="method-strip" aria-label="Current analysis setup">
      <strong>{icon_label("check", "Current setup")}</strong>
      {chip_markup}
    </section>
    """


def _validated_analysis_params(
    pixel_width_um: float,
    pixel_height_um: float,
    num_lines: int,
    line_spacing_um: float,
    grid_random_seed,
    min_chord_um: float,
    airspace_component_connectivity,
    grid_strategy_choice: str,
    grid_random_offset: bool,
    orientation: str,
    threshold_method: str,
    airspace_bright: bool,
    slide_roi_separator: str,
    field_selection_method: str,
    field_selection_notes: str,
    field_exclusion_criteria: str,
    calibration_source: str,
) -> tuple[AnalysisParams | None, list[str]]:
    pixel_width = ui_validation.safe_float(pixel_width_um, 0.0)
    pixel_height = ui_validation.safe_float(pixel_height_um, 0.0)
    line_count = ui_validation.safe_int(num_lines, 0)
    spacing = ui_validation.safe_float(line_spacing_um, 0.0)
    random_seed = ui_validation.optional_int(grid_random_seed)
    min_chord = ui_validation.safe_float(min_chord_um, 0.0)
    validation_errors: list[str] = []
    connectivity: int | None = None
    try:
        connectivity = ui_validation.component_connectivity_value(airspace_component_connectivity)
    except ValueError as exc:
        validation_errors.append(str(exc))
    if pixel_width <= 0:
        validation_errors.append("Pixel width must be greater than 0 µm/px.")
    if pixel_height <= 0:
        validation_errors.append("Pixel height must be greater than 0 µm/px.")
    if line_count < 1:
        validation_errors.append("Lines per orientation must be at least 1.")
    if spacing <= 0:
        validation_errors.append("Line spacing must be greater than 0 µm.")
    if random_seed is not None and random_seed < 0:
        validation_errors.append("Grid random seed must be a non-negative integer.")
    if min_chord < 0:
        validation_errors.append("Minimum chord length cannot be negative.")
    if validation_errors:
        return None, validation_errors

    assert connectivity is not None
    return (
        AnalysisParams(
            pixel_width_um=pixel_width,
            pixel_height_um=pixel_height,
            calibration_source=ui_validation.metadata_text(calibration_source, DEFAULT_CALIBRATION_SOURCE),
            grid_strategy=ui_validation.strategy_value(grid_strategy_choice),
            num_lines=line_count,
            line_spacing_um=spacing,
            grid_random_offset=bool(grid_random_offset),
            grid_random_seed=random_seed,
            orientation=(orientation or "both").lower(),
            threshold_method=(threshold_method or "huang").lower(),
            airspace_bright=ui_validation.airspace_bright_value(airspace_bright),
            airspace_component_connectivity=connectivity,
            exclude_edge_touching=True,
            min_chord_um=min_chord,
            measure_non_airspace=True,
            slide_roi_separator=slide_roi_separator or "_",
            field_selection_method=field_selection_method,
            field_selection_notes=field_selection_notes,
            field_exclusion_criteria=field_exclusion_criteria,
        ),
        [],
    )


def _build_run_status(
    log_df: pd.DataFrame,
    field_summary: pd.DataFrame,
    uploaded_count: int,
) -> str:
    ok_count = int((log_df["status"] == "ok").sum()) if not log_df.empty else 0
    warning_count = int((log_df["status"] == "warning").sum()) if not log_df.empty else 0
    error_count = int((log_df["status"] == "error").sum()) if not log_df.empty else 0
    default_calibration_count = (
        int(field_summary["calibration_is_default"].fillna(False).astype(bool).sum())
        if not field_summary.empty and "calibration_is_default" in field_summary
        else 0
    )
    calibration_warnings = (
        [
            str(value)
            for value in field_summary.get("calibration_warning", pd.Series(dtype=str)).dropna().unique()
            if str(value).strip()
        ]
        if not field_summary.empty
        else []
    )
    qc_warnings = (
        [
            str(value)
            for value in log_df.loc[
                (log_df["status"].astype(str).str.lower() == "warning")
                & (log_df["filename"].astype(str) != "calibration"),
                "message",
            ]
            .dropna()
            .unique()
            if str(value).strip()
        ]
        if not log_df.empty and {"filename", "status", "message"}.issubset(log_df.columns)
        else []
    )

    status_groups = _run_status_groups(
        ok_count=ok_count,
        uploaded_count=uploaded_count,
        warning_count=warning_count,
        error_count=error_count,
        calibration_warnings=calibration_warnings,
        default_calibration_count=default_calibration_count,
        qc_warnings=qc_warnings,
    )

    if error_count:
        kind, token, title, body = (
            "warning",
            "Review log",
            "Analysis completed with processing warnings",
            "Review the status/message columns in the field summary before using the measurements.",
        )
    elif default_calibration_count:
        kind, token, title, body = (
            "warning",
            "Review calibration",
            "Analysis complete with default calibration warning",
            "Verify the pixel-size calibration source before using physical measurements.",
        )
    elif warning_count:
        kind, token, title, body = (
            "warning",
            "Review log",
            "Analysis completed with processing warnings",
            "Review image-load and status/message columns before using the measurements.",
        )
    else:
        kind, token, title, body = (
            "complete",
            "Complete",
            "Analysis complete and packaged",
            "Review the summaries and QC previews, then download the ZIP archive for the full audit trail.",
        )
    return _status_panel(kind, token, title, body, groups=status_groups)


def _with_internal_input_index(frame: pd.DataFrame) -> pd.DataFrame:
    input_indexes = frame.attrs.get("_input_index")
    if "_input_index" in frame.columns or not isinstance(input_indexes, (list, tuple)):
        return frame
    if len(input_indexes) != len(frame):
        return frame
    out = frame.copy()
    out["_input_index"] = list(input_indexes)
    return out


def _field_display_source(
    field_summary: pd.DataFrame,
    log_df: pd.DataFrame,
    uploaded: list[str],
    slide_roi_separator: str,
) -> pd.DataFrame:
    field_summary_for_join = _with_internal_input_index(field_summary)
    field_log_df = _with_internal_input_index(log_df).copy()
    if not field_log_df.empty and "filename" in field_log_df.columns:
        uploaded_filenames = {Path(path).name for path in uploaded}
        field_log_df = field_log_df[field_log_df["filename"].isin(uploaded_filenames)].copy()
    status_values = (
        field_log_df.get("status", pd.Series(dtype=str)).astype(str).str.lower()
        if not field_log_df.empty
        else pd.Series(dtype=str)
    )
    message_values = (
        field_log_df.get("message", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
        if not field_log_df.empty
        else pd.Series(dtype=str)
    )
    show_status_columns = bool((status_values != "ok").any() or message_values.astype(bool).any())

    field_display_columns = ["slide_id", "field_id"]
    show_image_load_columns = False
    if not field_summary.empty:
        has_image_warnings = (
            "image_load_warnings" in field_summary
            and field_summary["image_load_warnings"].fillna("").astype(str).str.strip().astype(bool).any()
        )
        has_non_uint8_scaling = (
            "image_scaling_applied" in field_summary
            and field_summary["image_scaling_applied"].fillna(False).astype(bool).any()
        )
        has_multiple_pages = (
            "image_page_count" in field_summary
            and (pd.to_numeric(field_summary["image_page_count"], errors="coerce") > 1).any()
        )
        show_image_load_columns = bool(has_image_warnings or has_non_uint8_scaling or has_multiple_pages)
    if show_status_columns:
        field_display_columns.extend(["status", "message"])
    if show_image_load_columns:
        field_display_columns.append("image_load_warnings")
    field_display_columns.extend(
        [
            "mli_orientation_balanced_mean_um",
            "mli_chord_count",
            "mean_non_airspace_chord_um",
            "non_airspace_chord_count",
            "airspace_fraction",
            "non_edge_airspace_fraction",
            "non_edge_airspace_component_area_um2_mask",
        ]
    )

    if show_status_columns:
        log_status = field_log_df[
            [
                c
                for c in [
                    "filename",
                    "status",
                    "message",
                    "field_selection_method",
                    "field_exclusion_flag",
                    "field_exclusion_reason",
                    "_input_index",
                ]
                if c in field_log_df.columns
            ]
        ].copy()
        if "filename" in log_status:
            log_status["slide_id"] = log_status["filename"].map(
                lambda value: ui_validation.infer_slide_id(str(value), slide_roi_separator or "_")
            )
            log_status["field_id"] = log_status["filename"].map(
                lambda value: ui_validation.infer_field_id(str(value), slide_roi_separator or "_")
            )
        merge_key = None
        has_internal_key = (
            "_input_index" in log_status.columns and "_input_index" in field_summary_for_join.columns
        )
        if not field_summary_for_join.empty and has_internal_key:
            merge_key = "_input_index"
        elif (
            not field_summary_for_join.empty
            and "filename" in field_summary_for_join.columns
            and "filename" in log_status.columns
        ):
            # Filename-only joins are safe for normal unique basenames. When
            # duplicate basenames are present and no stable internal row key is
            # available, avoid a Cartesian merge that could associate a log row
            # with the wrong field measurements.
            field_filenames_unique = not field_summary_for_join["filename"].duplicated().any()
            log_filenames_unique = not log_status["filename"].duplicated().any()
            if field_filenames_unique and log_filenames_unique:
                merge_key = "filename"
        if merge_key is not None:
            field_display_source = log_status.merge(
                field_summary_for_join,
                on=merge_key,
                how="left",
                suffixes=("", "_summary"),
            )
            for column in ("filename", "slide_id", "field_id"):
                summary_column = f"{column}_summary"
                if summary_column in field_display_source:
                    field_display_source[column] = field_display_source[summary_column].combine_first(
                        field_display_source[column]
                    )
                    field_display_source = field_display_source.drop(columns=[summary_column])
        else:
            field_display_source = log_status
    else:
        field_display_source = field_summary.copy()
    return field_display_source[[c for c in field_display_columns if c in field_display_source.columns]]


def _slide_display_source(slide_summary: pd.DataFrame) -> pd.DataFrame:
    slide_display_columns = [
        "slide_id",
        "field_count",
        "mli_measured_field_count",
        "field_balanced_mean_mli_um",
        "field_sem_mli_um",
        "chord_pooled_mean_mli_um",
        "total_mli_chords",
        "non_airspace_measured_field_count",
        "field_balanced_mean_non_airspace_chord_um",
        "field_sd_non_airspace_chord_um",
        "field_sem_non_airspace_chord_um",
        "total_non_airspace_chords",
        "mean_airspace_fraction",
        "mean_non_edge_airspace_fraction",
        "mean_non_edge_airspace_component_area_um2_mask",
    ]
    return slide_summary[[c for c in slide_display_columns if c in slide_summary.columns]]


def run_analysis(
    uploads,
    folder_uploads,
    pixel_width_um: float,
    pixel_height_um: float,
    grid_strategy_choice: str,
    num_lines: int,
    line_spacing_um: float,
    grid_random_offset: bool,
    grid_random_seed,
    orientation: str,
    threshold_method: str,
    airspace_bright: bool,
    airspace_component_connectivity,
    min_chord_um: float,
    slide_roi_separator: str,
    field_selection_method: str,
    field_selection_notes: str,
    field_exclusion_criteria: str,
    calibration_source: str = DEFAULT_CALIBRATION_SOURCE,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
    request: gr.Request | None = None,
):
    uploaded = ui_validation.combined_uploads(uploads, folder_uploads)
    if not uploaded:
        return _empty_run_outputs(
            _status_panel(
                "warning",
                "No images",
                "No field images selected",
                "Upload cropped ROI images before running analysis.",
            )
        )
    upload_errors = ui_validation.upload_validation_errors(uploaded)
    if upload_errors:
        return _empty_run_outputs(
            _status_panel(
                "warning",
                "Review upload",
                "Upload limits need attention",
                "Remove oversized or unreadable images, then run the batch again.",
                ui_validation.display_upload_validation_errors(upload_errors),
            )
        )

    params, validation_errors = _validated_analysis_params(
        pixel_width_um,
        pixel_height_um,
        num_lines,
        line_spacing_um,
        grid_random_seed,
        min_chord_um,
        airspace_component_connectivity,
        grid_strategy_choice,
        grid_random_offset,
        orientation,
        threshold_method,
        airspace_bright,
        slide_roi_separator,
        field_selection_method,
        field_selection_notes,
        field_exclusion_criteria,
        calibration_source,
    )
    if validation_errors:
        return _empty_run_outputs(
            _status_panel(
                "warning",
                "Check settings",
                "Analysis settings need correction",
                "Fix the highlighted method values, then run the batch again.",
                validation_errors,
            )
        )

    assert params is not None
    progress(0.05, desc="Preparing analysis")
    ui_config.EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    ui_config.cleanup_stale_export_runs(ui_config.EXPORT_ROOT)
    work_dir = Path(tempfile.mkdtemp(prefix=ui_config.EXPORT_RUN_PREFIX, dir=ui_config.EXPORT_ROOT))
    results_dir = work_dir / "results"

    try:
        progress(0.10, desc="Preparing image list")

        def analysis_progress(fraction: float, description: str) -> None:
            progress(0.12 + (0.73 * fraction), desc=description)

        result = process_files(
            uploaded,
            results_dir,
            params,
            progress_callback=analysis_progress,
        )
        field_summary = result["field_summary"]
        slide_summary = result["slide_summary"]
        log_df = result["processing_log"]
        preview_paths = result["preview_paths"]

        progress(0.85, desc="Packaging results")
        archive_stem = ui_validation.archive_stem(uploaded, slide_roi_separator or "_")
        archive_path = shutil.make_archive(str(work_dir / archive_stem), "zip", root_dir=results_dir)

        status = _build_run_status(log_df, field_summary, len(uploaded))

        field_display_source = _field_display_source(field_summary, log_df, uploaded, slide_roi_separator or "_")
        slide_display_source = _slide_display_source(slide_summary)
        field_table = render_dataframe_table(
            _friendly_display_frame(field_display_source.head(100)),
            "Field summary",
            "The run completed, but no field summary rows were returned.",
            extra_class="field-summary-table",
            total_rows=len(field_display_source),
        )
        slide_table = render_dataframe_table(
            _friendly_display_frame(slide_display_source.head(100)),
            "Slide summary",
            "The run completed, but no slide summary rows were returned.",
            total_rows=len(slide_display_source),
        )
        _register_session_export(work_dir, request)
        progress(1.0, desc="Done")
        return (
            status,
            gr.update(visible=False),
            gr.update(value=field_table, visible=True),
            gr.update(visible=False),
            gr.update(value=slide_table, visible=True),
            gr.update(
                value=empty_output_panel("No QC previews", "The run completed, but no preview panels were returned."),
                visible=not bool(preview_paths),
            ),
            gr.update(value=render_qc_previews(preview_paths), visible=bool(preview_paths)),
            gr.update(value=archive_path, visible=True, label="Download results ZIP"),
            gr.update(visible=True, interactive=True, value="Analyze new dataset"),
            str(work_dir),
        )
    except Exception as exc:
        ui_config.remove_export_run(work_dir)
        _unregister_session_export(work_dir, request)
        logger.exception("Unexpected error during Gradio analysis run")
        return _empty_run_outputs(
            _status_panel(
                "error",
                "Failed",
                "Analysis failed before outputs were packaged",
                f"{type(exc).__name__}: {exc}",
                ["Check image files and method settings, then run again."],
            )
        )


LAUNCH_APPEARANCE = {
    "theme": CLINICAL_THEME,
    "css": CUSTOM_CSS,
    "js": CUSTOM_JS,
}
LAUNCH_KWARGS = LAUNCH_APPEARANCE


def create_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Lingappan MLI Analyzer",
        analytics_enabled=False,
        delete_cache=(ui_config.GRADIO_CACHE_CLEANUP_SECONDS, ui_config.GRADIO_CACHE_TTL_SECONDS),
    ) as demo:
        gr.HTML(
            f"""
            <header class="app-header">
              <div class="brand-row">
                <div class="brand-mark" aria-hidden="true">{brand_logo_svg()}</div>
                <div>
                  <h1>Lingappan MLI Analyzer</h1>
                  <p class="header-copy">MLI analysis for cropped lung histology fields, with QC overlays, slide summaries, and downloadable audit outputs.</p>
                  <p class="header-links"><a href="{GITHUB_README_URL}" target="_blank" rel="noopener noreferrer">GitHub README</a></p>
                </div>
              </div>
            </header>
            """,
            padding=False,
        )

        with gr.Column(elem_classes=["workflow-card"], min_width=0):
            gr.HTML(
                f"""
                <div class="workflow-intro">
                  <p class="panel-kicker">Setup protocol</p>
                  <h2>{icon_label("check", "Prepare the analysis run")}</h2>
                  <p>Move through each step, confirm the method settings, then download the complete ZIP package from the final step.</p>
                </div>
                """,
                padding=False,
            )
            with gr.Walkthrough(selected=1, elem_id="analysis-walkthrough", elem_classes=["analysis-walkthrough"]) as wizard_walkthrough:
                with gr.Step("Intake", id=1):
                    with gr.Column(elem_classes=["wizard-panel"], min_width=0):
                        gr.HTML(
                            section_heading(
                                "01",
                                "upload",
                                "Specimen field intake",
                                "Upload only pre-cropped lung ROI fields; whole-slide images are out of scope.",
                            ),
                            padding=False,
                        )
                        with gr.Row(elem_classes=["upload-choice-grid"], equal_height=True):
                            uploads = gr.File(
                                label="Individual image files",
                                file_count="multiple",
                                file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
                                type="filepath",
                                elem_classes=["upload-control", "file-upload-control"],
                            )
                            folder_uploads = gr.File(
                                label="Folder(s) of cropped images",
                                file_count="directory",
                                file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
                                type="filepath",
                                elem_classes=["upload-control", "folder-upload-control"],
                            )
                        slide_roi_separator = gr.Textbox(
                            label="Slide/field separator",
                            value="_",
                            max_lines=1,
                            info=(
                                "Enter the exact character or text before the field ID. "
                                "The final match in the filename stem is used for grouping fields by slide."
                            ),
                        )
                        slide_separator_help = gr.HTML(render_slide_separator_examples("_"), padding=False)
                        gr.HTML(
                            '<p class="control-note">Select individual pre-cropped ROI image files, upload one or more folders of ROI images, or use both. Whole-slide field selection and protocol-driven field exclusions must happen before this app; record that upstream method below.</p>',
                            padding=False,
                        )
                        with gr.Row(elem_classes=["parameter-grid"]):
                            field_selection_method = gr.Textbox(
                                label="Upstream field-selection method",
                                value="",
                                placeholder="Not recorded",
                                max_lines=1,
                                info="Recorded in parameters and field_summary; the app does not select fields from whole-slide images.",
                            )
                            field_exclusion_criteria = gr.Textbox(
                                label="Upstream field-exclusion criteria",
                                value="",
                                placeholder="Not recorded",
                                max_lines=1,
                                info="Criteria applied before upload, if any. The app records processing failures separately.",
                            )
                        field_selection_notes = gr.Textbox(
                            label="Field-selection notes (optional)",
                            value="",
                            lines=2,
                            max_lines=3,
                            info="Optional free-text notes copied to parameters, field_summary, and processing_log.",
                        )
                        gr.Image(
                            value=ui_config.EXAMPLE_IMAGE_PREVIEW,
                            label="Example cropped ROI image",
                            show_label=True,
                            buttons=[],
                            interactive=False,
                            height=260,
                            visible=ui_config.EXAMPLE_IMAGE_PREVIEW is not None,
                            elem_classes=["example-preview"],
                        )
                        with gr.Row(elem_classes=["wizard-nav"]):
                            gr.HTML("", padding=False)
                            intake_next = gr.Button("Continue to calibration", variant="primary", interactive=False)

                with gr.Step("Calibration", id=2):
                    with gr.Column(elem_classes=["wizard-panel"], min_width=0):
                        gr.HTML(
                            section_heading(
                                "02",
                                "ruler",
                                "Calibration",
                                "Set pixel size and calibration source before measurement.",
                            ),
                            padding=False,
                        )
                        with gr.Column(elem_classes=["example-preview-group"], min_width=0):
                            gr.HTML(render_example_panel(), padding=False)
                            calibration_preview = gr.Image(
                                value=ui_previews.render_calibration_preview(
                                    ui_config.EXAMPLE_PIXEL_WIDTH_UM,
                                    ui_config.EXAMPLE_PIXEL_HEIGHT_UM,
                                ),
                                label="Calibration example with measured dimensions",
                                show_label=True,
                                buttons=[],
                                interactive=False,
                                height=500,
                                visible=ui_config.EXAMPLE_IMAGE_PREVIEW is not None,
                                elem_classes=["example-preview"],
                            )
                        with gr.Row(elem_classes=["calibration-grid"]):
                            pixel_width_um = gr.Number(
                                label="Pixel width (µm/px)",
                                value=ui_config.EXAMPLE_PIXEL_WIDTH_UM,
                                precision=4,
                                minimum=0.0001,
                                info=f"Example {ui_config.EXAMPLE_FILENAME}: {ui_config.EXAMPLE_PIXEL_WIDTH_UM:g} µm/px.",
                            )
                            pixel_height_um = gr.Number(
                                label="Pixel height (µm/px)",
                                value=ui_config.EXAMPLE_PIXEL_HEIGHT_UM,
                                precision=4,
                                minimum=0.0001,
                                info=f"Example {ui_config.EXAMPLE_FILENAME}: {ui_config.EXAMPLE_PIXEL_HEIGHT_UM:g} µm/px. Use the same value for square pixels.",
                            )
                        calibration_source = gr.Textbox(
                            label="Calibration source",
                            value="",
                            placeholder="Built-in default: 0.57 × 0.57 µm/px",
                            max_lines=1,
                            info="Record where the pixel size came from (microscope metadata, stage micrometer, scale bar). Leaving this blank uses the built-in default and logs a warning.",
                        )
                        with gr.Row(elem_classes=["wizard-nav"]):
                            calibration_back = gr.Button("Back to intake", elem_classes=["secondary-action"])
                            calibration_next = gr.Button("Continue to filter setup", variant="primary")

                with gr.Step("Filter setup", id=3):
                    with gr.Column(elem_classes=["wizard-panel"], min_width=0):
                        gr.HTML(
                            section_heading(
                                "03",
                                "filter",
                                "Filter and segmentation setup",
                                "Confirm how airspace is segmented before chord measurement.",
                            ),
                            padding=False,
                        )
                        with gr.Column(elem_classes=["example-preview-group"], min_width=0):
                            gr.HTML(
                                f"""
                                <section class="example-inline" aria-label="Filter setup example reference">
                                  <h4>{icon_label("filter", "Filter preview")}</h4>
                                  <p>The blue overlay marks pixels classified as airspace by the selected threshold method. Edge-touching chords are excluded because the image boundary cuts them off. Non-airspace chord measurements, when reviewed, are line-sampling statistics and not septal wall thickness.</p>
                                </section>
                                """,
                                padding=False,
                            )
                            filter_preview = gr.Image(
                                value=ui_previews.render_filter_preview("huang", True),
                                label="Example filter preview",
                                buttons=[],
                                interactive=False,
                                height=520,
                                visible=ui_config.EXAMPLE_IMAGE_PREVIEW is not None,
                                elem_classes=["example-preview"],
                            )
                        threshold_method = gr.Radio(
                            label="Threshold method",
                            choices=[("Huang thresholding", "huang"), ("Otsu thresholding", "otsu")],
                            value="huang",
                            info="Huang is the default protocol setting; Otsu is available for comparison runs.",
                        )
                        airspace_bright = gr.Radio(
                            label="Airspace intensity",
                            choices=[("Airspaces are brighter than tissue", "bright"), ("Airspaces are darker than tissue", "dark")],
                            value="bright",
                            info="Choose the polarity that matches the stain or image inversion.",
                        )
                        airspace_component_connectivity = gr.Radio(
                            label="Airspace component connectivity",
                            choices=[("8-connected components (default)", "8"), ("4-connected components", "4")],
                            value="8",
                            info="Connectivity for filled connected-component area metrics and edge-touching component exclusion.",
                        )
                        with gr.Row(elem_classes=["wizard-nav"]):
                            filter_back = gr.Button("Back to calibration", elem_classes=["secondary-action"])
                            filter_next = gr.Button("Continue to line protocol", variant="primary")

                with gr.Step("Line protocol", id=4):
                    with gr.Column(elem_classes=["wizard-panel"], min_width=0):
                        gr.HTML(
                            section_heading(
                                "04",
                                "grid",
                                "Test-line protocol",
                                "Choose test-line density and orientation while checking a representative field."
                            ),
                            padding=False,
                        )
                        with gr.Column(elem_classes=["example-preview-group"], min_width=0):
                            gr.HTML(render_line_protocol_example(), padding=False)
                            line_protocol_preview = gr.Image(
                                value=ui_previews.render_line_protocol_preview(
                                    ui_config.EXAMPLE_PIXEL_WIDTH_UM,
                                    ui_config.EXAMPLE_PIXEL_HEIGHT_UM,
                                    "count",
                                    15,
                                    ui_config.EXAMPLE_LINE_SPACING_UM,
                                    "both",
                                    0.0,
                                    grid_random_offset=True,
                                ),
                                label="Example line preview",
                                buttons=[],
                                interactive=False,
                                height=520,
                                visible=ui_config.EXAMPLE_IMAGE_PREVIEW is not None,
                                elem_classes=["example-preview"],
                            )
                        grid_strategy = gr.Radio(
                            label="Grid strategy",
                            choices=[("Fixed number of lines", "count"), ("Physical line spacing", "spacing")],
                            value="count",
                            info="Use fixed line counts to match a protocol, or physical spacing to standardize density across fields.",
                        )
                        with gr.Row(elem_classes=["parameter-grid"]):
                            num_lines = gr.Number(
                                label="Lines per orientation",
                                value=15,
                                precision=0,
                                minimum=1,
                                info=f"Example {ui_config.EXAMPLE_FILENAME}: 15 lines per orientation.",
                            )
                            line_spacing_um = gr.Number(
                                label="Line spacing (µm)",
                                value=ui_config.EXAMPLE_LINE_SPACING_UM,
                                precision=2,
                                minimum=0.01,
                                info=f"Example physical spacing: {ui_config.EXAMPLE_LINE_SPACING_UM:g} µm.",
                                visible=False,
                            )
                        with gr.Row(elem_classes=["parameter-grid"]):
                            grid_random_offset = gr.Checkbox(
                                label="Systematic-random grid phase",
                                value=True,
                                info="Enabled by default: shift each count or spacing grid by a seed-recorded random phase.",
                                elem_classes=["systematic-grid-phase-control"],
                            )
                            grid_random_seed = gr.Number(
                                label="Grid random seed (optional)",
                                value=None,
                                precision=0,
                                minimum=0,
                                info="Leave blank to generate and record a seed when random phase is enabled.",
                            )
                        orientation = gr.Radio(
                            label="Orientation",
                            choices=[
                                ("Both orientations", "both"),
                                ("Horizontal only", "horizontal"),
                                ("Vertical only", "vertical"),
                            ],
                            value="both",
                            info="Both orientations reports horizontal and vertical results plus a combined MLI summary.",
                        )
                        min_chord_um = gr.Number(
                            label="Minimum chord length (µm)",
                            value=0.0,
                            precision=2,
                            minimum=0.0,
                            info="Optional lower cutoff for included chords. The preview updates to show which example chords remain.",
                        )
                        with gr.Row(elem_classes=["wizard-nav"]):
                            protocol_back = gr.Button("Back to filter setup", elem_classes=["secondary-action"])
                            protocol_next = gr.Button("Continue to run review", variant="primary")

                with gr.Step("Run", id=5):
                    with gr.Column(elem_classes=["wizard-panel"], min_width=0):
                        gr.HTML(
                            section_heading(
                                "05",
                                "download",
                                "Review, run, and export",
                                "Review the setup, run the analysis, then download the ZIP package.",
                            ),
                            padding=False,
                        )
                        method_summary = gr.HTML(
                            render_method_summary(
                                ui_config.EXAMPLE_PIXEL_WIDTH_UM,
                                ui_config.EXAMPLE_PIXEL_HEIGHT_UM,
                                "count",
                                15,
                                ui_config.EXAMPLE_LINE_SPACING_UM,
                                "both",
                                "huang",
                                True,
                                "8",
                                0.0,
                                "_",
                                "",
                                "",
                                grid_random_offset=True,
                            ),
                            padding=False,
                        )
                        status = gr.HTML(render_file_state(None, None), padding=False)

                        with gr.Column(elem_classes=["run-block"], min_width=0):
                            gr.HTML(
                                """
                                <p class="run-guidance">Creates field summaries plus slide summaries with field-balanced mean/SEM and supplementary chord-pooled means, QC previews, logs, and ZIP export. The download button appears here when packaging is complete.</p>
                                """,
                                padding=False,
                            )
                            with gr.Row(elem_classes=["wizard-nav", "run-action-row"]):
                                run_back = gr.Button("Back to line protocol", elem_classes=["secondary-action"])
                                run_button = gr.Button(
                                    "Run MLI analysis",
                                    variant="primary",
                                    size="lg",
                                    interactive=False,
                                    elem_classes=["run-action"],
                                )

                        with gr.Row(elem_classes=["post-run-actions", "run-action-row"]):
                            restart_button = gr.Button(
                                "Analyze new dataset",
                                variant="secondary",
                                size="lg",
                                visible=True,
                                interactive=False,
                                elem_classes=["secondary-action", "restart-action"],
                            )
                            download_button = gr.DownloadButton(
                                "Download results ZIP",
                                variant="primary",
                                size="lg",
                                visible=False,
                                elem_classes=["download-action"],
                            )

                        with gr.Column(elem_classes=["results-stack"], min_width=0):
                            gr.HTML(f'<h3 class="results-section-title">{icon_label("table", "Field summary", "result-title-line")}</h3>', padding=False)
                            field_empty = gr.HTML(
                                empty_output_panel("Field summary", "Run analysis to populate per-field measurements."),
                                padding=False,
                            )
                            field_summary = gr.HTML("", visible=False, padding=False)

                            gr.HTML(f'<h3 class="results-section-title">{icon_label("table", "Slide summary", "result-title-line")}</h3>', padding=False)
                            slide_empty = gr.HTML(
                                empty_output_panel("Slide summary", "Slide-level summaries appear after a completed run."),
                                padding=False,
                            )
                            slide_summary = gr.HTML("", visible=False, padding=False)

                            gr.HTML(f'<h3 class="results-section-title">{icon_label("image", "QC previews", "result-title-line")}</h3>', padding=False)
                            gallery_empty = gr.HTML(
                                empty_output_panel("QC previews", "Preview panels appear when processed images are available."),
                                padding=False,
                            )
                            gallery = gr.HTML("", visible=False, padding=False)

        current_run_dir = gr.State(
            value=None,
            time_to_live=(
                ui_config.EXPORT_RUN_TTL_SECONDS if ui_config.EXPORT_RUN_TTL_SECONDS > 0 else None
            ),
            delete_callback=ui_config.remove_export_run,
        )

        setup_inputs = [
            pixel_width_um,
            pixel_height_um,
            grid_strategy,
            num_lines,
            line_spacing_um,
            orientation,
            threshold_method,
            airspace_bright,
            airspace_component_connectivity,
            min_chord_um,
            slide_roi_separator,
            field_selection_method,
            field_exclusion_criteria,
            grid_random_offset,
            grid_random_seed,
            calibration_source,
        ]

        _wire_live_update(setup_inputs, fn=render_method_summary, inputs=setup_inputs, outputs=method_summary)
        _wire_live_update(
            (pixel_width_um, pixel_height_um),
            fn=ui_previews.render_calibration_preview,
            inputs=[pixel_width_um, pixel_height_um],
            outputs=calibration_preview,
        )
        _wire_live_update(
            (slide_roi_separator,),
            fn=render_slide_separator_examples,
            inputs=[slide_roi_separator],
            outputs=slide_separator_help,
        )

        grid_strategy.change(
            fn=render_grid_strategy_inputs,
            inputs=grid_strategy,
            outputs=[num_lines, line_spacing_um],
            queue=False,
            show_progress="hidden",
        )

        line_preview_inputs = [
            pixel_width_um,
            pixel_height_um,
            grid_strategy,
            num_lines,
            line_spacing_um,
            orientation,
            min_chord_um,
            threshold_method,
            airspace_bright,
            grid_random_offset,
            grid_random_seed,
        ]
        _wire_live_update(
            line_preview_inputs,
            fn=ui_previews.render_line_protocol_preview,
            inputs=line_preview_inputs,
            outputs=line_protocol_preview,
        )

        filter_preview_inputs = [threshold_method, airspace_bright]
        _wire_live_update(
            filter_preview_inputs,
            fn=ui_previews.render_filter_preview,
            inputs=filter_preview_inputs,
            outputs=filter_preview,
        )

        _wire_wizard_navigation(
            wizard_walkthrough,
            [
                (intake_next, 2),
                (calibration_back, 1),
                (calibration_next, 3),
                (filter_back, 2),
                (filter_next, 4),
                (protocol_back, 3),
                (protocol_next, 5),
                (run_back, 4),
            ],
        )

        upload_inputs = [uploads, folder_uploads]
        uploads.change(
            fn=render_upload_state,
            inputs=upload_inputs,
            outputs=[status, run_button, intake_next, restart_button],
            queue=False,
            show_progress="hidden",
        )
        folder_uploads.change(
            fn=render_upload_state,
            inputs=upload_inputs,
            outputs=[status, run_button, intake_next, restart_button],
            queue=False,
            show_progress="hidden",
        )

        run_button.click(
            fn=run_analysis,
            inputs=[
                uploads,
                folder_uploads,
                pixel_width_um,
                pixel_height_um,
                grid_strategy,
                num_lines,
                line_spacing_um,
                grid_random_offset,
                grid_random_seed,
                orientation,
                threshold_method,
                airspace_bright,
                airspace_component_connectivity,
                min_chord_um,
                slide_roi_separator,
                field_selection_method,
                field_selection_notes,
                field_exclusion_criteria,
                calibration_source,
            ],
            outputs=[
                status,
                field_empty,
                field_summary,
                slide_empty,
                slide_summary,
                gallery_empty,
                gallery,
                download_button,
                restart_button,
                current_run_dir,
            ],
            show_progress_on=status,
        )
        restart_button.click(
            fn=reset_analysis_workflow,
            inputs=current_run_dir,
            outputs=[
                wizard_walkthrough,
                uploads,
                folder_uploads,
                status,
                run_button,
                intake_next,
                field_empty,
                field_summary,
                slide_empty,
                slide_summary,
                gallery_empty,
                gallery,
                download_button,
                restart_button,
                current_run_dir,
            ],
            queue=False,
            show_progress="hidden",
        )


    demo.unload(cleanup_session_exports)
    demo.queue(default_concurrency_limit=1, max_size=8)
    return demo


demo = create_demo()


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        **LAUNCH_KWARGS,
    )
