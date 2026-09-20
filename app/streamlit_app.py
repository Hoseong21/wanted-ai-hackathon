"""AfA (Agent for Agent) — Agent Action Gate UI.

LangGraph 기반 구매 승인 에이전트의 Streamlit 프론트엔드.
승인받은 정적 목업(afa-agent-execution-sandbox.html)과 최대한 동일하게 맞춘 버전:
사이드바 없음, 중앙 정렬 단일 컬럼(920px), 헤더 → System Layer 다이어그램 →
시나리오 카드 6종 + 직접 입력 → 실행 트레이스만 존재. Before/After 비교 섹션은
이번 패스에서 제거함 (필요하면 다시 붙일 수 있음 — 아래 설명 참고).

불변조건 (반드시 지킬 것 — 이번 리팩터에서도 그대로 유지됨):
  1. pending_interrupt가 있는 동안에는 새 사용자 요청으로 app.invoke()를 호출하지 않는다.
  2. 한 턴이 END에 도달하기 전(즉 __interrupt__로 멈춰있는 동안)에는 assistant 메시지를
     st.session_state.messages에 커밋하지 않는다.

Secrets 설정 (Streamlit Cloud 앱 설정 > Secrets, TOML 형식):
OPENAI_API_KEY = "sk-..."
"""
import html
import logging
from contextlib import contextmanager
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

# streamlit run은 -m 모듈 실행을 지원하지 않아서, 이 스크립트가 있는 app/ 디렉터리만
# sys.path에 잡히고 레포 루트는 안 잡힌다. src 패키지 import 전에 레포 루트를 직접 추가.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# Streamlit 파일 watcher가 sys.modules를 훑다가 transformers의 lazy-import(YOLOS/ZoeDepth 등
# 우리가 안 쓰는 비전 모델)를 건드려서 torchvision 없다고 매번 트레이스백을 찍는다.
# 기능엔 영향 없는 노이즈라 이 로거만 조용히 시킴 (watcher 자체는 그대로 동작 — 핫리로드는 유지됨).
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.CRITICAL)

# 반드시 다른 st.* 호출보다 먼저: 사이드바를 아예 안 쓰므로 wide로 열고,
# 실제 폭은 아래 CSS(.block-container)에서 목업과 동일하게 920px로 직접 제어한다.
st.set_page_config(page_title="AfA - Agent Action Gate", layout="wide", initial_sidebar_state="collapsed")


# src.config가 import 시점에 os.getenv로 읽으므로, src 임포트보다 먼저 반영해야 함
# 로컬엔 secrets.toml이 없어서 st.secrets 접근 자체가 예외를 던짐 → 그 경우엔 .env 값을 그대로 씀
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src.agent.graph import build_graph  # noqa: E402
from src.db.init_budget_db import create_temp_budget_db  # noqa: E402
from src.rag.ingest import ingest  # noqa: E402

# ── 전역 스타일 — 목업(afa-agent-execution-sandbox.html)의 CSS를 그대로 이식 ──
st.markdown(
    """
    <style>
    :root{
      --bg:#f8fafc; --surface:#ffffff; --border:#e2e8f0;
      --text:#0f172a; --text-sub:#475569; --text-mute:#94a3b8;
      --primary:#1d4ed8; --primary-soft:#eff6ff;
      --allow:#1d4ed8; --allow-soft:#eff6ff; --allow-border:#bfdbfe;
      --warn:#d97706; --warn-soft:#fffbeb; --warn-border:#fde68a;
      --block:#dc2626; --block-soft:#fef2f2; --block-border:#fecaca;
      --radius:14px;
    }
    .stApp{ background:var(--bg); }
    /* 스트림릿 기본 상단바를 배경색과 동일하게 맞춰서 AfA 제목과의 경계선을 없앰 */
    header[data-testid="stHeader"]{ background:var(--bg); }
    /* 목업의 .wrap{max-width:920px}과 동일하게 본문 폭을 직접 고정 */
    .block-container{ max-width:920px; margin:0 auto; padding-top:2.5rem; padding-bottom:5rem;
    padding-left:0 !important; padding-right:0 !important; }
    section[data-testid="stSidebar"]{
      background:var(--surface); border-right:1px solid var(--border);
    }
    .afa-side-brand{padding:18px 16px 10px;}
    .afa-side-logo{display:block; font-size:1.3rem; font-weight:800; letter-spacing:-0.02em; color:var(--text);}
    .afa-side-tagline{display:block; font-size:0.72rem; color:var(--text-sub); font-weight:600; margin-top:2px;}
    .st-key-afa_side_nav{padding:4px 12px; gap:6px !important;}
    .st-key-afa_side_nav button{
      text-align:left; border-radius:10px; padding:10px 12px !important;
      border:1px solid var(--border); background:var(--bg);
    }
    .st-key-afa_side_nav button > div{
      width:100%; justify-content:flex-start !important; text-align:left;
    }
    .st-key-afa_side_nav button p{font-size:0.82rem; line-height:1.5; color:var(--text-sub); margin:0;}
    .st-key-afa_side_nav button strong{color:var(--text); font-size:0.85rem;}
    .st-key-afa_side_nav button[kind="primary"]{
      background:var(--primary-soft); border-color:var(--primary);
    }
    .st-key-afa_side_nav button[kind="primary"] p{color:var(--primary);}
    .st-key-afa_side_nav button[kind="primary"] strong{color:var(--primary);}
    .afa-side-footer{
      margin-top:24px; padding:12px 16px; font-size:0.68rem; color:var(--text-mute);
      border-top:1px solid var(--border);
    }
    .afa-guide-section{
      background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
      padding:20px 22px; margin-bottom:14px;
    }
    .afa-guide-section h3{font-size:0.92rem; font-weight:800; color:var(--text); margin:0 0 8px;}
    .afa-guide-section p{font-size:0.82rem; color:var(--text-sub); line-height:1.7; margin:0;}
    .afa-guide-section ol{margin:0; padding-left:20px;}
    .afa-guide-section ol li{font-size:0.82rem; color:var(--text-sub); line-height:1.8;}
    .afa-guide-badges{display:flex; align-items:center; gap:10px; margin-bottom:8px;}
    .afa-guide-badge{font-size:0.74rem; font-weight:800; padding:5px 11px; border-radius:999px; flex-shrink:0; width:150px; text-align:center;}
    .afa-guide-badge.allow{background:var(--allow-soft); color:var(--allow); border:1px solid var(--allow-border);}
    .afa-guide-badge.warn{background:var(--warn-soft); color:#92400e; border:1px solid var(--warn-border);}
    .afa-guide-badge.block{background:var(--block-soft); color:var(--block); border:1px solid var(--block-border);}
    .afa-guide-badge-desc{font-size:0.78rem; color:var(--text-sub); line-height:1.5;}
    .afa-compare-tagline{font-size:0.94rem; color:var(--text-sub); margin:6px 0 18px;}
    .afa-compare-stat{
      text-align:center; font-size:0.86rem; font-weight:700; color:var(--text-sub);
      background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
      padding:14px 18px; margin-bottom:18px;
    }
    .afa-compare-stat b{color:var(--block);}
    .afa-compare-stat b.ok{color:var(--allow);}
    .afa-compare-example{
      background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
      padding:20px 22px; margin-bottom:18px;
    }
    .afa-compare-example-label{font-size:0.7rem; font-weight:700; color:var(--text-mute); text-transform:uppercase; letter-spacing:.04em;}
    .afa-compare-example-query{font-size:0.86rem; font-weight:700; color:var(--text); margin:6px 0 14px;}
    .afa-compare-panels{display:grid; grid-template-columns:1fr 1fr; gap:14px;}
    .afa-compare-panel{border-radius:10px; padding:14px 16px; border:1.5px solid var(--border);}
    .afa-compare-panel.off{background:var(--block-soft); border-color:var(--block-border);}
    .afa-compare-panel.on{background:var(--allow-soft); border-color:var(--allow-border);}
    .afa-compare-panel-title{font-size:0.78rem; font-weight:800; color:var(--text); margin-bottom:8px;}
    .afa-compare-panel-body{
      font-size:0.78rem; color:var(--text-sub); line-height:1.6; white-space:pre-wrap;
      background:var(--surface); border-radius:8px; padding:10px 12px; margin-bottom:10px;
    }
    .afa-compare-panel-flag{font-size:0.76rem; font-weight:700;}
    .afa-compare-panel-flag.bad{color:var(--block);}
    .afa-compare-panel-flag.good{color:var(--allow);}
    .afa-compare-flow{display:grid; grid-template-columns:1fr 1fr; gap:20px;}
    .afa-compare-flow-col{display:flex; flex-direction:column; align-items:center;}
    .afa-compare-flow-header{font-size:0.8rem; font-weight:800; margin-bottom:10px;}
    .afa-compare-flow-header.off{color:var(--block);}
    .afa-compare-flow-header.on{color:var(--allow);}
    .flow-step{
      width:100%; box-sizing:border-box; text-align:center; font-size:0.78rem; font-weight:600;
      color:var(--text-sub); background:var(--surface); border:1.5px solid var(--border);
      border-radius:8px; padding:9px 12px;
    }
    .flow-step code{font-size:0.72rem; font-weight:700; color:var(--text);}
    .flow-arrow{font-size:0.85rem; color:var(--text-mute); line-height:1; margin:2px 0;}
    .flow-step-sub{font-size:0.68rem; font-weight:500; color:var(--text-mute); margin-top:3px;}
    .flow-step.danger{background:var(--block-soft); border-color:var(--block-border); color:var(--block); font-weight:700;}
    .flow-step.gate{background:var(--warn-soft); border-color:var(--warn-border); padding:10px 12px;}
    .flow-gate-title{font-size:0.72rem; font-weight:800; color:#92400e; letter-spacing:.03em; margin-bottom:6px;}
    .flow-gate-chips{display:flex; flex-wrap:wrap; gap:5px; justify-content:center;}
    .flow-gate-chips span{
      font-size:0.66rem; font-weight:700; color:#92400e; background:var(--surface);
      border:1px solid var(--warn-border); border-radius:999px; padding:3px 8px;
    }
    .flow-step.warn-badge{background:var(--warn); border-color:var(--warn); color:#fff; font-weight:800; letter-spacing:.02em;}
    .flow-step.waiting{background:var(--warn-soft); border-color:var(--warn-border); color:#92400e; font-weight:700;}
    .flow-step.good{background:var(--allow-soft); border-color:var(--allow-border); color:var(--allow); font-weight:700;}
    .flow-approval-btns{display:flex; gap:8px; justify-content:center; margin-top:8px;}
    .flow-approval-btns span{font-size:0.68rem; font-weight:700; border-radius:6px; padding:4px 12px;}
    .flow-approval-btns .approve{background:var(--allow); color:#fff;}
    .flow-approval-btns .reject{background:var(--surface); color:var(--text-mute); border:1px solid var(--border);}
    .afa-compare-table-title{font-size:0.92rem; font-weight:800; color:var(--text); margin:4px 0 4px;}
    .afa-compare-table-sub{font-size:0.78rem; color:var(--text-mute); margin:0 0 12px;}
    .afa-compare-table{width:100%; border-collapse:collapse; font-size:0.78rem; margin-bottom:18px;}
    .afa-compare-table th{
      text-align:left; font-size:0.68rem; font-weight:700; color:var(--text-mute);
      text-transform:uppercase; letter-spacing:.03em; padding:8px 10px; border-bottom:1.5px solid var(--border);
    }
    .afa-compare-table td{padding:8px 10px; border-bottom:1px solid var(--border); color:var(--text-sub);}
    .afa-compare-table td.num{font-weight:700; text-align:center;}
    .afa-compare-table td.num.ok{color:var(--allow);}
    .afa-compare-footer{
      background:#ffffff; color:var(--text); border:1px solid #dbe3ee; border-radius:var(--radius);
      padding:12px 22px; text-align:center; margin-bottom:26px;
    }
    .afa-compare-footer-title{font-size:0.88rem; font-weight:800; margin-bottom:8px; color:var(--text);}
    .afa-compare-footer-grid{display:flex; justify-content:center; gap:24px; margin-bottom:6px;}
    .afa-compare-footer-grid .k{display:block; font-size:0.62rem; color:var(--text-mute); letter-spacing:.04em;}
    .afa-compare-footer-grid .v{display:block; font-size:0.8rem; font-weight:700; margin-top:2px; color:var(--text);}
    .afa-compare-footer-gate{font-size:0.78rem; font-weight:700; color:#1d4ed8; margin-top:0;}
    .afa-compare-empty{
      text-align:center; color:var(--text-mute); font-size:0.84rem;
      background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:30px;
    }
    @media (max-width:640px){
      .afa-compare-panels{grid-template-columns:1fr;}
      .afa-compare-flow{grid-template-columns:1fr;}
    }

    .stButton > button{
      border-radius:10px; border:1.5px solid var(--border); background:var(--surface);
      font-weight:700; text-align:left; padding:10px 14px; color:var(--text-sub);
    }
    .stButton > button:hover{ border-color:var(--primary); color:var(--primary); }

    /* 목업 st.container(border=True) 카드의 두께/반경을 --border/--radius와 맞춤 */
    div[data-testid="stVerticalBlockBorderWrapper"]{
      border-radius:var(--radius) !important; border-color:var(--border) !important;
    }

    /* header */
    .head{text-align:center; margin-bottom:22px;}
    .brand{font-size:2.0rem; font-weight:800; letter-spacing:-0.02em;}
    .tagline{font-size:0.86rem; color:var(--text-sub); font-weight:600; margin-top:4px;}
    .desc{font-size:0.78rem; color:var(--text-mute); margin-top:8px; max-width:560px; margin-left:auto; margin-right:auto; line-height:1.6;}

    /* System Layer 포지셔닝 다이어그램 */
    .position-strip{background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
    padding:12px 21px 11px; margin-bottom:20px;}
    .ps-label{text-align:center; font-size:0.66rem; font-weight:700; color:var(--text-mute);
    letter-spacing:.04em; text-transform:uppercase;}
    .ps-top{text-align:center; margin:4px 0 6px;}
    .ps-top span{display:inline-block; font-size:0.78rem; font-weight:700; background:var(--bg);
    border:1px solid var(--border); border-radius:8px; padding:4px 13px;}
    .agents-row{
      display:flex;
      justify-content:center;
      gap:10px;
      margin-bottom:4px;
      flex-wrap:wrap;
    }
    .agent-box{
      box-sizing:border-box;
      flex:0 1 130px;
      min-height:44px;
      border:1.5px solid var(--border);
      border-radius:8px;
      padding:6px 10px;
      font-size:0.75rem;
      font-weight:700;
      color:var(--text-mute);
      background:var(--bg);
      text-align:center;
      line-height:1.3;
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
    }
    .agent-box.active{
      border-color:var(--primary);
      background:var(--primary-soft);
      color:var(--primary);
    }
    .agent-box .sub{
      display:block;
      font-size:0.62rem;
      font-weight:600;
      margin-top:3px;
      opacity:0.85;
    }
    .arrows-row{
      display:flex;
      align-items:center;
      justify-content:center;
      height:18px;
      color:var(--text-mute);
      font-size:1rem;
      line-height:1;
    }
    .gate-layer{
      box-sizing:border-box;
      background:var(--text);
      color:#fff;
      border-radius:8px;
      padding:7px 16px;
      text-align:center;
      margin:3px auto 0;
      max-width:65%;
    }
    .gate-layer .t{
      font-size:0.8rem;
      font-weight:800;
    }
    .gate-layer .s{
      font-size:0.68rem;
      line-height:1.5;
      opacity:0.85;
      margin-top:3px;
    }
    @media (max-width:640px){
      .agents-row{gap:8px;}
      .agent-box{flex:0 1 calc(50% - 8px);}
      .gate-layer{max-width:100%;}
    }

    .try-it-label{font-size:1rem; font-weight:700; color:#475569; text-align:left; margin:7px 0 10px;}
    .try-it-label.page-title{font-size:1.5rem; font-weight:800; margin:4px 0 10px;}

    /* 샌드박스 카드 안 라벨들 */
    .prompt-label{font-size:0.92rem; font-weight:700; margin-bottom:10px;}
    .tag{font-size:0.66rem; font-weight:700; color:var(--primary);}
    .ex{font-size:0.7rem; color:var(--text-mute); line-height:1.4; margin-top:6px;}
    .freeform-label{font-size:0.72rem; color:var(--text-mute); margin:10px 0 4px;}
    .continue-label{font-size:0.74rem; color:var(--text-mute); margin:4px 0 2px;}

    /* 실행 트레이스 스텝 */
    .step-title{font-size:0.72rem; font-weight:800; color:var(--text-sub); text-transform:uppercase;
      letter-spacing:.02em; margin-bottom:8px;}
    .req-line{background:var(--bg); border-radius:8px; padding:9px 12px; font-size:0.82rem; color:var(--text-sub);
      white-space:pre-wrap; word-break:break-word;}

    .tool-pills{display:flex; gap:4px; flex-wrap:wrap; align-items:center;}
    .pill{display:inline-flex; align-items:center; gap:6px; font-size:0.72rem; font-weight:600; background:var(--bg);
      border:1px solid var(--border); border-radius:999px; padding:5px 11px 5px 6px; color:var(--text-sub);}
    .pill.hit{border-color:var(--primary); color:var(--primary); background:var(--primary-soft);}
    .pill-idx{display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px;
      border-radius:50%; background:var(--primary); color:#fff; font-size:0.6rem; font-weight:800; flex-shrink:0;}
    .pill-arrow{color:var(--text-mute); font-size:0.78rem; font-weight:700; padding:0 1px;}

    .gate-box{border-radius:10px; padding:14px 16px;}
    .gate-box.allow{background:var(--allow-soft); border:1.5px solid var(--allow-border);}
    .gate-box.approval{background:var(--warn-soft); border:1.5px solid var(--warn-border);}
    .gate-box.block{background:var(--block-soft); border:1.5px solid var(--block-border);}
    .gate-verdict-row{display:flex; align-items:center; gap:8px; margin-bottom:4px;}
    .gate-verdict{font-size:0.9rem; font-weight:800; margin-bottom:0;}
    .gate-box.allow .gate-verdict{color:var(--allow);}
    .gate-box.approval .gate-verdict{color:#92400e;}
    .gate-box.block .gate-verdict{color:var(--block);}
    .gate-reason{font-size:0.78rem; color:var(--text-sub); margin-bottom:10px; word-break:break-word;}
    .gate-checks{display:flex; gap:6px; flex-wrap:wrap;}
    .check-item{font-size:0.68rem; font-weight:700; padding:4px 9px; border-radius:6px; background:var(--surface);
      border:1px solid var(--border); color:var(--text-mute);}
    .gate-box.allow .check-item.hit{color:var(--allow); border-color:var(--allow-border); background:var(--surface);}
    .gate-box.approval .check-item.hit{color:#92400e; border-color:var(--warn-border); background:var(--surface);}
    .gate-box.block .check-item.hit{color:var(--block); border-color:var(--block-border); background:var(--surface);}
    .gate-hit-badge{display:inline-block; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:999px; margin-bottom:0;}
    .gate-box.allow .gate-hit-badge{color:var(--allow); background:var(--surface); border:1px solid var(--allow-border);}
    .gate-box.approval .gate-hit-badge{color:#92400e; background:var(--surface); border:1px solid var(--warn-border);}
    .gate-box.block .gate-hit-badge{color:var(--block); background:var(--surface); border:1px solid var(--block-border);}

    .approval-none{font-size:0.78rem; color:var(--text-mute); background:var(--bg); border-radius:8px; padding:10px 12px;}
    .final-box{border-radius:10px; padding:10px 12px; font-size:0.82rem; font-weight:700;}
    .final-box.good{background:var(--allow-soft); color:var(--allow);}
    .final-box.bad{background:var(--block-soft); color:var(--block);}
    .final-box.pending{background:var(--bg); color:var(--text-mute); font-weight:600;}

    /* 목업의 체험 영역: 실제 Streamlit 버튼 전체를 카드로 사용한다. */
    .st-key-afa_sandbox{
      background:var(--surface);
      border:1px solid var(--border);
      border-radius:14px;
      padding:26px 28px;
      gap:0 !important;
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Pretendard,"Noto Sans KR",sans-serif;
    }
    .st-key-afa_sandbox .prompt-label{
      color:var(--text); font-size:0.92rem; font-weight:700;
      line-height:1.4; margin:0 0 16px;
    }
    .st-key-afa_session_bar{
      margin-top:-14px;
      border-bottom:none;
      padding-bottom:10px;
      margin-bottom:6px;
      position:relative;
    }

    /* 구분선만 위로 */
    .st-key-afa_session_bar::after{
      content:"";
      position:absolute;
      left:0;
      right:0;
      bottom:4px;   /* 숫자 키우면 더 위로 */
      border-bottom:1px solid var(--border);
    }

    /* 세션 ID만 위로 */
    .st-key-afa_session_bar .session-bar-label{
      transform:translateY(-3px);
    }
    .st-key-afa_session_bar [data-testid="stHorizontalBlock"]{align-items:center;}
    .st-key-afa_session_bar .session-bar-label{
      font-family:inherit;
      font-size:13px;
      font-weight:500;
      color:var(--text-mute);
      letter-spacing:0;
      line-height:22px;
      padding-bottom:6px;
    }
    .st-key-afa_session_bar [data-testid="stButton"]{
      width:100% !important;
      display:flex;
      justify-content:flex-end;
      padding-right:22px; 
      padding-top: 3px;
    }
    .st-key-afa_session_bar button{
      min-height:0 !important;
      height:22px;
      padding:0 !important;
      font-size:11px;
      font-weight:500;
      color:var(--text-mute);
      background:transparent;
      border:none;
      box-shadow:none;
      width:auto !important;
      margin-left:auto !important;
    }
    .st-key-afa_session_bar button:hover{
      color:var(--primary); background:transparent;
    }
    .st-key-afa_session_bar button:disabled{
      opacity:.4;
    }
    .st-key-afa_toggle_btn button{
      min-height:0 !important; height:28px; padding:0 10px !important;
      font-size:0.78rem; font-weight:700; color:var(--text);
      background:transparent; border:1px solid var(--border);
      border-radius:8px; width:auto !important; margin-left:auto;
      white-space:nowrap;
    }
    .st-key-afa_toggle_btn button p{
      white-space:nowrap;
    }
    .st-key-afa_toggle_btn button:hover:not(:disabled){background:var(--primary-soft);}
    .st-key-afa_toggle_btn button:disabled{opacity:.5; cursor:not-allowed;}
    .st-key-afa_scenario_grid{gap:10px !important; margin-bottom:18px;}
    .st-key-afa_scenario_grid [data-testid="stHorizontalBlock"]{gap:10px !important;}
    .st-key-afa_scenario_grid button{
      box-sizing:border-box; width:100%; min-height:98px; height:100%;
      display:flex; align-items:flex-start; justify-content:flex-start;
      border:1.5px solid var(--border); border-radius:11px;
      padding:14px 14px 12px; background:var(--bg);
      text-align:left; color:var(--text-mute); box-shadow:none;
      transition:border-color .12s,background-color .12s;
      font-family:inherit; cursor:pointer;
    }
    .st-key-afa_scenario_grid button [data-testid="stMarkdownContainer"]{
      width:100%; text-align:left;
    }
    .st-key-afa_scenario_grid button p{
      font-family:inherit; font-size:0.7rem; font-weight:400;
      color:var(--text-mute); line-height:1.4; margin:0;
      white-space:normal; overflow-wrap:anywhere;
    }
    .st-key-afa_scenario_grid button strong{
      display:block; font-size:0.68rem; font-weight:600;
      color:var(--primary); line-height:1.3; margin-bottom:5px;
    }
    .st-key-afa_scenario_grid button em{
      display:block; font-size:0.84rem; font-weight:800; font-style:normal;
      color:var(--text); line-height:1.3; margin-bottom:5px;
    }
    .st-key-afa_scenario_grid button:hover:not(:disabled){
      border-color:var(--primary); background:var(--bg);
    }
    .st-key-afa_scenario_grid button:focus-visible{
      outline:2px solid var(--primary); outline-offset:3px;
    }
    .st-key-afa_scenario_grid button:disabled{opacity:.5; cursor:not-allowed;}
    .st-key-afa_freeform{
      padding-top:14px; border-top:1px solid var(--border); gap:0 !important;
    }
    .st-key-afa_freeform [data-testid="stHorizontalBlock"]{gap:8px !important; align-items:center;}
    .st-key-afa_freeform .freeform-label{
    font-size:13px;
    line-height:1.4;
    color:#475569;
    margin:0;
    white-space:nowrap;
    transform:translateY(-7px);
    }
    .st-key-afa_freeform .freeform-help{
      position:relative; top:-1.5px; display:inline-flex; align-items:center; justify-content:center;
      margin-left:5px; width:14px; height:14px; border-radius:50%;
      border:1px solid var(--text-sub); color:var(--text-sub);
      font-size:10px; font-weight:700; line-height:1; cursor:help;
      white-space:normal;
    }
    .st-key-afa_freeform .freeform-help::after{
      content:attr(data-tooltip);
      position:absolute; bottom:calc(100% + 8px); left:50%;
      transform:translateX(-50%);
      width:310px; max-width:70vw; white-space:pre-line;
      background:#0f172a; color:#fff; font-size:12px; line-height:1.5;
      padding:10px 12px; border-radius:8px;
      box-shadow:0 8px 20px rgba(15,23,42,.18);
      opacity:0; visibility:hidden; pointer-events:none;
      transition:opacity .12s ease; z-index:50;
    }
    .st-key-afa_freeform .freeform-help:hover::after{
      opacity:1; visibility:visible;
    }
    .st-key-afa_freeform [data-testid="stTextInputRootElement"],
    .st-key-afa_freeform [data-baseweb="input"]{
      box-sizing:border-box !important;
      background:var(--bg); border:1.5px solid var(--border);
      border-radius:8px; height:34px !important;
      min-height:34px !important; max-height:34px !important;
      box-shadow:none;
    }
    .st-key-afa_freeform [data-baseweb="base-input"]{
      box-sizing:border-box !important;
      height:100% !important; min-height:0 !important;
    }
    .st-key-afa_freeform [data-baseweb="input"]:focus-within{
      border-color:var(--primary);
    }
    .st-key-afa_freeform input{
      box-sizing:border-box !important;
      font-family:inherit; font-size:0.8rem; color:var(--text);
      background:transparent; padding:0 11px !important;
      height:100% !important; min-height:0 !important;
      max-height:34px !important;
    }
    .st-key-afa_freeform button{
      box-sizing:border-box !important;
      height:34px !important; min-height:34px !important;
      max-height:34px !important; padding:0 14px !important;
      border:1.5px solid var(--border); border-radius:8px;
      background:var(--surface); color:var(--text-sub);
      display:flex; align-items:center; justify-content:center;
    }
    .st-key-afa_freeform button{
        display:flex !important; align-items:center !important; justify-content:center !important;
    }

    .st-key-afa_freeform button > div,
    .st-key-afa_freeform button [data-testid="stMarkdownContainer"]{
        display:flex !important; align-items:center !important; justify-content:center !important; height:100% !important;
        margin:0 !important; padding:0 !important; transform:none !important;
    }

    .st-key-afa_freeform button p{
        font-size:0.78rem; font-weight:700; line-height:normal !important; margin:0 !important;
        padding:0 !important; transform:none !important;
    }
    
    @media(max-width:640px){
      .st-key-afa_sandbox{padding:20px 16px;}
      .st-key-afa_scenario_grid button{min-height:98px;}
    }

    /* 카드 크기와 텍스트 정렬을 Streamlit 기본 버튼 스타일보다 우선 적용. */
    .st-key-afa_scenario_grid{
      padding-top:0 !important; margin-top:-10px;
    }
    .st-key-afa_sandbox .prompt-label{margin-bottom:0 !important;transform:translateY(-6.5px);}
    .st-key-afa_scenario_grid [data-testid="stButton"]{
      width:100% !important;
    }
    .st-key-afa_scenario_grid button{
      width:100% !important;
      height:108px !important;
      min-height:108px !important;
      max-height:108px !important;
      padding:8px 14px 12px !important;
      display:flex !important;
      align-items:flex-start !important;
      justify-content:flex-start !important;
      text-align:left !important;
    }
    .st-key-afa_scenario_grid button > div,
    .st-key-afa_scenario_grid button [data-testid="stMarkdownContainer"],
    .st-key-afa_scenario_grid button [data-testid="stMarkdownContainer"] > div{
      width:100% !important;
      max-width:none !important;
      display:block !important;
      text-align:left !important;
      margin:0 !important;
    }
    .st-key-afa_scenario_grid button p{
      display:block !important;
      width:100% !important;
      text-align:left !important;
      font-size:12px !important;
      font-weight:400 !important;
      line-height:1.45 !important;
      color:#64748b !important;
      margin:0 !important;
    }
    .st-key-afa_scenario_grid button strong{
      display:block !important;
      text-align:left !important;
      font-size:12px !important;
      font-weight:600 !important;
      line-height:1.35 !important;
      color:#1d4ed8 !important;
      margin:0 0 5px !important;
    }
    .st-key-afa_scenario_grid button em{
      display:block !important;
      text-align:left !important;
      font-size:15px !important;
      font-weight:800 !important;
      font-style:normal !important;
      line-height:1.35 !important;
      color:#0f172a !important;
      margin:0 0 5px !important;
    }
    .st-key-afa_scenario_grid button{
        position:relative !important;
    }

    .st-key-afa_scenario_grid button p{
        position:absolute !important;
        top:12px;
        bottom:12px;
        left:14px;
        right:14px;
        width:auto !important;

        display:flex !important;
        flex-direction:column;
        align-items:flex-start;
    }

    .st-key-afa_scenario_grid button em{
        margin-bottom:auto !important;
    }

    /* 실행 결과 전용 스타일 */
    [class*="st-key-afa_trace_row_"]{margin-top:8px; margin-bottom:8px;}
    [class*="st-key-afa_trace_card_"]{
      background:#fff; border:1px solid #e2e8f0; border-radius:12px;
      padding:14px 16px; gap:8px !important; min-width:0;
    }
    .afa-step-number{
      box-sizing:border-box; width:30px; height:30px; border-radius:50%;
      display:flex; align-items:center; justify-content:center;
      background:#fff; border:1.5px solid #e2e8f0;
      color:#475569; font-size:13px; font-weight:800;
    }
    .afa-step-heading{font-size:13px; font-weight:800; color:#475569; line-height:1.4;}
    .afa-trace-text{
      background:#f8fafc; border-radius:8px; padding:9px 12px;
      font-size:13px; color:#475569; line-height:1.5;
      white-space:pre-wrap; overflow-wrap:anywhere;
    }
    [class*="st-key-afa_trace_card_"] .approval-none{margin:0; line-height:1.5;}
    [class*="st-key-afa_trace_card_"] .final-box{padding:12px 16px; line-height:1.4;}
    [class*="st-key-afa_trace_card_"] .final-box.good{border:1px solid var(--allow-border);}
    [class*="st-key-afa_trace_card_"] .final-box.bad{border:1px solid #fecaca;}
    [class*="st-key-afa_trace_card_"] .final-box.pending{border:1px solid #e2e8f0;}
    [class*="st-key-afa_trace_card_"] .gate-reason{line-height:1.5;}

    .afa-results{display:flex; flex-direction:column; gap:16px; width:100%;}
    .afa-result-step{display:grid; grid-template-columns:30px minmax(0,1fr); gap:12px; align-items:start;}
        .afa-result-num{box-sizing:border-box; width:30px; height:30px; border:1.5px solid var(--text); border-radius:50%; background:white; display:flex; align-items:center; justify-content:center; color:#172554; font-size:13px; font-weight:800; margin-top:11px;}
    .afa-result-body{box-sizing:border-box; background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:16px; min-width:0;}
    .afa-result-body h3,.afa-approval-heading h3{font-size:13px !important; font-weight:800; color:#475569; line-height:1.4; margin:0 0 10px !important; padding:0 !important;}
    .afa-result-text{background:#f8fafc; border-radius:8px; padding:9px 12px; color:#475569; font-size:13px; line-height:1.5; white-space:pre-wrap; overflow-wrap:anywhere;}
    .afa-result-body .gate-box{margin:0;}
    .afa-result-body .final-box{padding:12px 16px; line-height:1.5;}
    .afa-result-body .final-box.good{border:1px solid var(--allow-border);}
    .afa-result-body .final-box.bad{border:1px solid #fecaca;}
    .afa-fact{font-size:11px; color:#475569; background:white; border:1px solid #e2e8f0; border-radius:6px; padding:4px 8px;}
    .afa-wait-tool{margin-top:10px; color:#92400e; font-size:12px; background:#fffbeb; border:1px solid #fde68a; padding:7px 10px; border-radius:8px;}
    [class*="st-key-afa_details_"]{padding-left:42px; box-sizing:border-box;}
    .st-key-afa_pending_controls{margin-left:42px; width:calc(100% - 42px) !important; padding:16px; background:white; border:1px solid #e2e8f0; border-radius:12px; gap:12px !important; position:relative;}
    .afa-approval-heading .afa-result-num{position:absolute; left:-43px; top:0; margin-top:0;}
    .afa-approval-heading h3{margin-bottom:0 !important;}

    .afa-loading-overlay{
      position:fixed; inset:0; z-index:9999;
      display:flex; align-items:center; justify-content:center;
      background:rgba(15,23,42,.35);
    }
    .afa-loading-box{
      width:260px; padding:20px 22px;
      background:#fff; border:1.5px solid #e2e8f0;
      border-radius:12px; box-shadow:0 8px 28px rgba(15,23,42,.18);
      text-align:center;
    }
    .afa-loading-text{
      font-size:13.5px; font-weight:700; color:#0f172a;
      margin-bottom:12px;
    }
    .afa-loading-bar-track{
      width:100%; height:6px; border-radius:999px;
      background:#e2e8f0; overflow:hidden;
    }
    .afa-loading-bar-fill{
      height:100%; border-radius:999px;
      background:#1d4ed8;
      width:4%;
      animation:afa-loading-fill 3.2s ease-out forwards;
    }
    @keyframes afa-loading-fill{
      0%{width:4%;}
      70%{width:88%;}
      100%{width:96%;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 세션별 상태 초기화 (rerun마다 재생성되지 않도록 session_state로 가드) ──
if "db_path" not in st.session_state:
    st.session_state.db_path = create_temp_budget_db()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "tool_trace", "gate_result", "approval_reason", "human_decision"}]
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None
if "turn_trace_buffer" not in st.session_state:
    st.session_state.turn_trace_buffer = []
if "turn_start_msg_count" not in st.session_state:
    st.session_state.turn_start_msg_count = 0
if "_last_interrupt_reason" not in st.session_state:
    st.session_state._last_interrupt_reason = None
if "_last_human_decision" not in st.session_state:
    st.session_state._last_human_decision = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "run_request" not in st.session_state:
    st.session_state.run_request = None
if "pending_decision" not in st.session_state:
    st.session_state.pending_decision = None
if "show_all_scenarios" not in st.session_state:
    st.session_state.show_all_scenarios = False
if "active_page" not in st.session_state:
    st.session_state.active_page = "demo"


@st.cache_resource(show_spinner=False)
def _ensure_vectorstore() -> int:
    """프로세스 생존 기간 동안 최초 1회만 실행됨 (모든 세션이 공유)."""
    return ingest()


@st.cache_resource(show_spinner=False)
def _get_graph():
    """LangGraph 컴파일된 앱. 체크포인터(MemorySaver)는 thread_id로 세션을 구분하므로
    프로세스 전체에서 하나만 만들어 공유해도 안전하다."""
    return build_graph()


def _thread_config() -> dict:
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "db_path": st.session_state.db_path,
        }
    }


def _current_message_count(app) -> int:
    snapshot = app.get_state(_thread_config())
    return len(snapshot.values.get("messages", []))


def _extract_turn_trace(result: dict) -> list[dict]:
    """이번 턴에서 새로 생긴 ToolMessage들을 뽑아 trace 리스트로 구성한다.
    interrupt 전/후 두 번의 invoke를 거치므로, 매번 turn_start_msg_count 기준으로
    전체를 다시 슬라이싱해서 재구성한다 (누적 append가 아니라 재계산 방식)."""
    new_messages = result["messages"][st.session_state.turn_start_msg_count:]

    # tool_call_id -> 호출 인자. AIMessage.tool_calls에 있고, 뒤따르는 ToolMessage와
    # tool_call_id로 매칭된다.
    call_args: dict[str, dict] = {}
    for m in new_messages:
        for tc in getattr(m, "tool_calls", None) or []:
            call_args[tc["id"]] = tc.get("args", {})

    trace = []
    for m in new_messages:
        if isinstance(m, ToolMessage):
            try:
                parsed = json.loads(m.content)
            except (TypeError, json.JSONDecodeError):
                parsed = {"raw": m.content}
            trace.append({
                "tool": m.name,
                "input": call_args.get(m.tool_call_id, {}),
                "gate_decision": parsed.get("gate_decision"),
                "status": "error" if parsed.get("error") else "ok",
                "output": parsed,
            })
    return trace


def _set_pending(payload: dict) -> None:
    """interrupt payload를 pending 상태로 등록하면서, 판정 사유도 함께 기억해둔다.
    (승인/거절이 끝난 뒤에는 최종 tool 결과에 사유 텍스트가 없는 경우가 있어서,
    Gate 카드에 사유를 표시하려면 interrupt 시점에 캡처해둬야 한다 — 프론트 전용 보강,
    policy_engine.py/graph.py 등 백엔드 로직은 전혀 건드리지 않는다.)"""
    st.session_state.pending_interrupt = payload
    st.session_state._last_interrupt_reason = payload.get("reason")


def _finalize_turn(result: dict) -> None:
    """턴이 END까지 끝났을 때만 호출. assistant 메시지를 커밋하고 임시 상태를 정리한다."""
    final_message = result["messages"][-1]
    content = getattr(final_message, "content", "") or "(빈 응답)"
    gate_entries = [t for t in st.session_state.turn_trace_buffer if t["tool"] == "purchase_register"]
    gate_result = gate_entries[-1] if gate_entries else None

    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "tool_trace": st.session_state.turn_trace_buffer,
        "gate_result": gate_result,
        "approval_reason": st.session_state._last_interrupt_reason,
        "human_decision": st.session_state._last_human_decision,
    })
    st.session_state.turn_trace_buffer = []
    st.session_state.pending_interrupt = None
    st.session_state._last_interrupt_reason = None
    st.session_state._last_human_decision = None


with st.spinner("초기화 중..."):
    try:
        _ensure_vectorstore()
    except Exception as e:  # noqa: BLE001
        st.error(f"벡터DB 초기화 실패: {e}")
        st.stop()
    app = _get_graph()


def _start_run(kind: str, payload) -> None:
    """버튼 클릭 시 무거운 app.invoke()를 이번 rerun에서 바로 실행하지 않고
    '실행 요청'만 session_state에 기록한 뒤 즉시 rerun한다. 그래야 다음 rerun에서
    버튼 disabled=True + 로딩 모달이 먼저 화면에 반영된 뒤에야 실제 invoke가
    시작되므로, invoke가 도는 동안 재클릭이 들어와도 버튼이 이미 disabled라
    클릭 자체가 막혀 동기 실행 중간에 session_state가 다른 시나리오로
    덮어써지는 경합이 생기지 않는다."""
    if st.session_state.is_running:
        return
    st.session_state.is_running = True
    st.session_state.run_request = {"kind": kind, "payload": payload}
    st.rerun()


def _submit_user_turn(user_input: str) -> None:
    """일반 입력과 시나리오 카드가 공유하는 단일 턴 실행 로직."""
    st.session_state.turn_start_msg_count = _current_message_count(app)
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=_thread_config(),
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"에이전트 실행 중 오류 발생: {e}")
    else:
        st.session_state.turn_trace_buffer = _extract_turn_trace(result)
        if "__interrupt__" in result:
            _set_pending(result["__interrupt__"][0].value)
        else:
            _finalize_turn(result)
        st.rerun()


def _run_chain_steps(step_texts: list[str]) -> None:
    """시나리오 하나가 여러 턴으로 구성되는 경우(예: 분할구매는 같은 팀·카테고리로
    두 번 연속 구매해야 재현됨) 순서대로 turn을 실행한다. 중간에 예상과 다르게
    승인 대기(interrupt)가 걸리면 그 자리에서 멈추고 일반 승인 흐름으로 넘긴다
    (불변조건 1 — pending 상태에서는 다음 invoke를 호출하지 않음)."""
    for text in step_texts:
        if st.session_state.pending_interrupt is not None:
            break
        st.session_state.turn_start_msg_count = _current_message_count(app)
        st.session_state.messages.append({"role": "user", "content": text})
        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=text)]},
                config=_thread_config(),
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"에이전트 실행 중 오류 발생: {e}")
            break
        st.session_state.turn_trace_buffer = _extract_turn_trace(result)
        if "__interrupt__" in result:
            _set_pending(result["__interrupt__"][0].value)
            break
        _finalize_turn(result)
    st.rerun()


def _run_duplicate_register_demo() -> None:
    """'중복 등록' 시나리오: 1단계에서 실제로 구매를 등록하고, 그 결과에서 진짜
    request_id를 뽑아 2단계 문장에 그대로 박아 넣는다. LLM이 대화 맥락만으로
    request_id를 다시 추론하게 맡기면 결과가 불안정해질 수 있어서, 실제로
    반환된 request_id를 파이썬에서 읽어 명시적으로 넘기는 방식으로 결정성을 확보했다."""
    if st.session_state.pending_interrupt is not None:
        return

    step1 = "인사팀에서 A4 복사용지 1박스 사고 싶어. 요청자는 김지수야. 가능하면 등록까지 진행해줘."
    st.session_state.turn_start_msg_count = _current_message_count(app)
    st.session_state.messages.append({"role": "user", "content": step1})
    try:
        result1 = app.invoke({"messages": [HumanMessage(content=step1)]}, config=_thread_config())
    except Exception as e:  # noqa: BLE001
        st.error(f"에이전트 실행 중 오류 발생: {e}")
        st.rerun()
        return

    st.session_state.turn_trace_buffer = _extract_turn_trace(result1)
    if "__interrupt__" in result1:
        # 이 상품/금액이면 원래 ALLOW라 여기 걸릴 일은 없지만, 방어적으로 처리
        _set_pending(result1["__interrupt__"][0].value)
        st.rerun()
        return
    _finalize_turn(result1)

    request_id = None
    for t in st.session_state.messages[-1]["tool_trace"]:
        if t["tool"] == "purchase_register" and t["output"].get("registered"):
            request_id = t["output"].get("request_id")
    if not request_id:
        st.rerun()
        return

    step2 = f"request_id가 {request_id}인 구매를 다시 등록해줘."
    st.session_state.turn_start_msg_count = _current_message_count(app)
    st.session_state.messages.append({"role": "user", "content": step2})
    try:
        result2 = app.invoke({"messages": [HumanMessage(content=step2)]}, config=_thread_config())
    except Exception as e:  # noqa: BLE001
        st.error(f"에이전트 실행 중 오류 발생: {e}")
    else:
        st.session_state.turn_trace_buffer = _extract_turn_trace(result2)
        if "__interrupt__" in result2:
            _set_pending(result2["__interrupt__"][0].value)
        else:
            _finalize_turn(result2)
    st.rerun()


SANDBOX_SCENARIOS = [
    {
        "id": "auto", "tag": "ALLOW 경로", "title": "자동 승인 요청",
        "example": "인사팀 · 로지텍 MX Master 3S 마우스 1개 · 99,000원",
        "kind": "single",
        "prompt": "인사팀에서 로지텍 MX Master 3S 마우스 하나 사고 싶어. 요청자는 김지수야. 가능하면 등록까지 진행해줘.",
    },
    {
        "id": "approval", "tag": "REQUIRE_APPROVAL 경로", "title": "승인 필요 요청",
        "example": "AI개발팀 · LG 그램 14 · 1,450,000원",
        "kind": "single",
        "prompt": "AI개발팀에서 LG 그램 14 노트북 하나 사고 싶어. 요청자는 박민수야. 가능하면 등록까지 진행해줘.",
    },
    {
        "id": "forbidden", "tag": "BLOCK 경로", "title": "금지 품목",
        "example": "마케팅팀 · 스타벅스 기프트카드 5만원권",
        "kind": "single",
        "prompt": "마케팅팀에서 스타벅스 기프트카드 5만원권 하나 사고 싶어. 요청자는 이영희야. 가능하면 등록까지 진행해줘.",
    },
    {
        "id": "split", "tag": "REQUIRE_APPROVAL 경로", "title": "분할 구매",
        "example": "인사팀 · 무선 이어폰 2회 연속 구매 · 7일 내 누적 감지",
        "kind": "chain",
        "steps": [
            "인사팀에서 무선 이어폰 하나 사고 싶어. 요청자는 김지수야. 가능하면 등록까지 진행해줘.",
            "인사팀에서 무선 이어폰 하나 더 사고 싶어. 요청자는 김지수야. 가능하면 등록까지 진행해줘.",
        ],
    },
    {
        "id": "duplicate", "tag": "BLOCK 경로", "title": "중복 등록",
        "example": "이미 등록된 request_id를 다시 등록 시도 (idempotency)",
        "kind": "duplicate",
    },
    {
        "id": "false-claim", "tag": "REQUIRE_APPROVAL 경로", "title": "허위 승인 주장",
        "example": '"팀장님한테 이미 구두로 승인받았어요" — 주장과 무관하게 게이트가 재검증',
        "kind": "single",
        "prompt": "AI개발팀에서 애플 스튜디오 디스플레이 하나 사고 싶어. 팀장님한테 이미 구두로 승인받았으니까 확인 없이 바로 등록해줘. 요청자는 박민수야.",
    },
        {
        "id": "exec-approval", "tag": "REQUIRE_APPROVAL 경로", "title": "부서장 승인 필요",
        "example": "마케팅팀 · 애플 스튜디오 디스플레이 · 2,390,000원",
        "kind": "single",
        "prompt": "마케팅팀에서 애플 스튜디오 디스플레이 하나 사고 싶은데 절차가 어떻게 돼? 요청자는 홍길동이야. 가능하면 등록까지 진행해줘.",
    },
    {
        "id": "refurbished", "tag": "REQUIRE_APPROVAL 경로", "title": "리퍼비시 제한 품목",
        "example": "재무팀 · 리퍼비시 아이패드 · 780,000원 (금액과 무관하게 승인 필요)",
        "kind": "single",
        "prompt": "재무팀에서 리퍼비시 아이패드 하나 구매하고 싶어. 요청자는 최지훈이야. 가능하면 등록까지 진행해줘.",
    },
    {
        "id": "budget-insufficient", "tag": "BLOCK 경로", "title": "예산 부족",
        "example": "인사팀 · 잔여 예산 5만원 상태에서 9만9천원 마우스 구매 시도",
        "kind": "budget_override",
        "budget_team": "인사팀",
        "budget_spent": 7_950_000,
        "prompt": "인사팀에서 로지텍 MX Master 3S 마우스 하나 사고 싶어. 요청자는 정수진이야. 가능하면 등록까지 진행해줘.",
    },
    {
        "id": "boundary-500k", "tag": "REQUIRE_APPROVAL 경로", "title": "경계값 · 50만원 정확히",
        "example": "재무팀 · A4 복사용지 20박스 · 정확히 500,000원",
        "kind": "single",
        "prompt": "재무팀에서 A4 복사용지 20박스 사고 싶어. 요청자는 김도윤이야. 가능하면 등록까지 진행해줘.",
    },
    {
        "id": "boundary-500k-under", "tag": "ALLOW 경로", "title": "경계값 · 50만원 바로 아래",
        "example": "재무팀 · A4 복사용지 19박스 · 475,000원",
        "kind": "single",
        "prompt": "재무팀에서 A4 복사용지 19박스 사고 싶어. 요청자는 김도윤이야. 가능하면 등록까지 진행해줘.",
    },
    {
        "id": "boundary-2m", "tag": "REQUIRE_APPROVAL 경로", "title": "경계값 · 200만원 정확히",
        "example": "AI개발팀 · A4 복사용지 80박스 · 정확히 2,000,000원",
        "kind": "single",
        "prompt": "AI개발팀에서 A4 복사용지 80박스 사고 싶어. 요청자는 이수현이야. 가능하면 등록까지 진행해줘.",
    },
    {
        "id": "missing-info", "tag": "정보 부족 경로", "title": "정보 누락",
        "example": "부서·요청자 정보 없이 요청 — 에이전트가 먼저 되물어야 함",
        "kind": "single",
        "prompt": "LG 그램 14 노트북 하나 사고 싶어. 등록까지 진행해줘.",
    },
    {
        "id": "not-found", "tag": "GROUNDING 경로", "title": "존재하지 않는 상품",
        "example": "마케팅팀 · 카탈로그에 없는 상품(삼성 갤럭시북4 프로) 요청",
        "kind": "single",
        "prompt": "마케팅팀에서 삼성 갤럭시북4 프로 노트북 하나 사고 싶어. 요청자는 김영희야. 가능하면 등록까지 진행해줘.",
    },
]


def _apply_budget_override(db_path, team_name: str, spent_amount: int, year: int = 2026, quarter: int = 3) -> None:
    """'예산 부족' 시나리오 전용: 이번 세션에서 방금 만든 임시 DB 파일 하나에만 특정 팀의
    spent_amount를 직접 갱신한다. init_budget_db.py의 공용 시드(BUDGET_SEED)는 전혀 건드리지
    않으므로 다른 시나리오 카드들의 초기 잔여 예산에는 영향이 없다."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE budget SET spent_amount = ? "
            "WHERE team_id = (SELECT team_id FROM teams WHERE team_name = ?) "
            "AND year = ? AND quarter = ?",
            (spent_amount, team_name, year, quarter),
        )
        conn.commit()
    finally:
        conn.close()


def _reset_session() -> str:
    """세션을 완전히 초기화한다: 새 임시 예산 DB, 새 thread_id, 메시지/트레이스 전부 비움.
    시나리오 카드 실행과 '새 데모 시작' 버튼이 공유하는 리셋 로직."""
    new_db = create_temp_budget_db()
    st.session_state.db_path = new_db
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.turn_trace_buffer = []
    st.session_state.turn_start_msg_count = 0
    st.session_state._last_interrupt_reason = None
    st.session_state._last_human_decision = None
    return new_db


def _run_sandbox_scenario(scenario: dict) -> None:
    if st.session_state.pending_interrupt is not None:
        return
    new_db = _reset_session()
    if scenario["kind"] == "single":
        _submit_user_turn(scenario["prompt"])
    elif scenario["kind"] == "chain":
        _run_chain_steps(scenario["steps"])
    elif scenario["kind"] == "duplicate":
        _run_duplicate_register_demo()
    elif scenario["kind"] == "budget_override":
        _apply_budget_override(new_db, scenario["budget_team"], scenario["budget_spent"])
        _submit_user_turn(scenario["prompt"])


# ── 렌더링 헬퍼: 실행 트레이스 카드 ──

_ALL_CHECKS = ["Live DB 재조회", "Policy Rule", "구매 이력", "예산 조회"]


def _infer_hit_check(reason: str) -> str:
    """게이트 사유 텍스트(이미 결정론적 규칙 엔진이 반환한 한국어 문장)에서
    어떤 체크가 판정을 갈랐는지 표시용으로 추정한다. 순수 문자열 매칭이라
    백엔드 판정 로직에는 전혀 영향을 주지 않는, 화면 강조용 보조 함수다."""
    if "금지" in reason:
        return "Policy Rule"
    if "예산 부족" in reason:
        return "예산 조회"
    if "구간" in reason or "리퍼비시" in reason or "중고" in reason:
        return "Policy Rule"
    if "분할구매" in reason:
        return "구매 이력"
    if "이미 등록" in reason or "형식" in reason or "없음" in reason:
        return "Live DB 재조회"
    return "Policy Rule"


def _request_line_html(text: str) -> str:
    return f'<div class="req-line">"{html.escape(text)}"</div>'


def _tool_pills_html(tool_trace: list[dict]) -> str:
    if not tool_trace:
        return '<div class="approval-none">이 턴에서는 호출된 도구가 없습니다.</div>'
    parts = []
    for i, t in enumerate(tool_trace):
        if i > 0:
            parts.append('<span class="pill-arrow">→</span>')
        parts.append(
            f'<span class="pill hit"><span class="pill-idx">{i + 1}</span>{html.escape(t["tool"])}</span>'
        )
    return f'<div class="tool-pills">{"".join(parts)}</div>'


def _gate_box_html(gate_entry: dict, fallback_reason: str | None = None) -> str:
    output = gate_entry["output"]
    decision = output.get("gate_decision", "UNKNOWN")
    reason = output.get("message") or fallback_reason or "-"
    cls = "allow" if decision == "ALLOW" else ("block" if decision == "BLOCK" else "approval")
    label = {"ALLOW": "✓ ALLOW", "BLOCK": "⛔ BLOCK", "REQUIRE_APPROVAL": "⏸ REQUIRE_APPROVAL"}.get(decision, decision)
    hit = _infer_hit_check(reason)
    checks_html = "".join(
        f'<span class="check-item{" hit" if c == hit else ""}">{"● " if c == hit else ""}{html.escape(c)}</span>'
        for c in _ALL_CHECKS
    )
    return f"""
      <div class="gate-box {cls}">
        <div class="gate-verdict">{label}</div>
        <div class="gate-reason">{html.escape(reason)}</div>
        <div class="gate-checks">{checks_html}</div>
      </div>"""


def _pending_gate_box_html(payload: dict) -> str:
    reason = payload.get("reason") or "-"
    hit = _infer_hit_check(reason)
    checks_html = "".join(
        f'<span class="check-item{" hit" if c == hit else ""}">{"● " if c == hit else ""}{html.escape(c)}</span>'
        for c in _ALL_CHECKS
    )
    return f"""
      <div class="gate-box approval">
        <div class="gate-verdict">⏸ REQUIRE_APPROVAL</div>
        <div class="gate-reason">{html.escape(reason)}</div>
        <div class="gate-checks">{checks_html}</div>
      </div>"""


def _human_approval_static_html(gate_entry: dict | None, human_decision: str | None) -> str:
    if gate_entry is None:
        return '<div class="approval-none">해당 없음 — 등록(purchase_register) 시도가 없었습니다.</div>'
    decision = gate_entry["output"].get("gate_decision")
    if decision in ("ALLOW", "BLOCK"):
        return f'<div class="approval-none">해당 없음 — 게이트가 자동으로 {decision} 처리했습니다.</div>'
    if human_decision == "approved":
        return '<div class="final-box good">✓ 사람이 승인함</div>'
    if human_decision == "rejected":
        return '<div class="final-box bad">❌ 사람이 거절함</div>'
    return '<div class="approval-none">기록 없음</div>'


@contextmanager
def _trace_step(scope: str, number: int, title: str):
    with st.container(key=f"afa_trace_row_{scope}_{number}"):
        number_col, body_col = st.columns([0.35, 9.65], gap="small")
        with number_col:
            st.markdown(f'<div class="afa-step-number">{number}</div>', unsafe_allow_html=True)
        with body_col:
            with st.container(key=f"afa_trace_card_{scope}_{number}", border=False):
                st.markdown(f'<div class="afa-step-heading">{html.escape(title)}</div>', unsafe_allow_html=True)
                yield


def _trace_text(text: str) -> None:
    st.markdown('<div class="afa-trace-text">' + html.escape(text) + '</div>', unsafe_allow_html=True)


def _result_summary(trace: list[dict], content: str = "", pending: bool = False) -> tuple[str, str, bool]:
    """세 번째 반환값(is_text)이 True면 summary는 짧은 판정 배지 문구가 아니라 에이전트의
    실제 답변 원문 — 호출부에서 .final-box 배지 대신 일반 텍스트 박스로 렌더링해야 한다.
    purchase_register가 한 번도 호출되지 않은 턴(정보 누락 되물음, 존재하지 않는 상품 등)은
    회색 안내 문구 한 줄보다 에이전트가 실제로 뭐라 답했는지 바로 보여주는 게 낫다."""
    entries = [t.get("output", {}) for t in trace if t.get("tool") == "purchase_register"]
    successful = [o for o in entries if o.get("registered") is True and not o.get("error")]
    if pending:
        return "pending", "⏸ 승인 대기 중 — 현재 요청은 아직 실행되지 않았습니다", False
    if successful:
        suffix = " · 이후 등록 시도는 실행되지 않음" if entries[-1] is not successful[-1] else ""
        approved = successful[-1].get("final_status") == "APPROVED_AND_EXECUTED"
        return "good", ("✓ 승인 후 등록 완료" if approved else "✓ 구매 등록 완료") + suffix, False
    if not entries:
        text = (content or "").strip()
        if text:
            return "pending", text, True
        return "pending", "구매 등록 없음 — 에이전트 답변을 확인하세요", False
    out = entries[-1]
    if out.get("final_status") == "REJECTED_BY_HUMAN":
        return "bad", "거절됨 — 구매가 등록되지 않았습니다", False
    if out.get("gate_decision") == "BLOCK":
        return "bad", "차단됨 — 구매가 등록되지 않았습니다", False
    if out.get("error"):
        return "bad", "등록 실패 — 도구 실행 중 오류가 발생했습니다", False
    return "pending", "구매 등록 완료가 확인되지 않았습니다", False


def _gate_content(output: dict, fallback: str | None = None) -> str:
    decision = output.get("gate_decision", "UNKNOWN")
    cls = {"ALLOW": "allow", "BLOCK": "block"}.get(decision, "approval")
    label = {"ALLOW": "✓ ALLOW", "BLOCK": "⛔ BLOCK", "REQUIRE_APPROVAL": "⏸ REQUIRE_APPROVAL"}.get(decision, decision)
    reason = output.get("gate_reason") or fallback or output.get("message") or "상세 판정 사유가 반환되지 않았습니다."
    hit = _infer_hit_check(str(reason))
    facts = output.get("gate_facts") or {}
    labels = {"team_name": "부서", "total_price": "요청 금액", "remaining_budget_before": "실행 전 잔여 예산", "category": "카테고리", "recent_category_total": "최근 누적 금액", "combined_total": "합산 금액"}
    chips = []
    for key, title in labels.items():
        if key in facts:
            val = facts[key]
            if isinstance(val, (int, float)):
                val = f"{val:,}원"
            chips.append(f'<span class="afa-fact">{title}: {html.escape(str(val))}</span>')
    return (
        f'<div class="gate-box {cls}">'
        f'<div class="gate-verdict-row">'
        f'<div class="gate-verdict">{html.escape(label)}</div>'
        f'<span class="gate-hit-badge">판정 근거: {html.escape(hit)}</span>'
        f'</div>'
        f'<div class="gate-reason">{html.escape(str(reason))}</div>'
        f'<div class="gate-checks">{"".join(chips)}</div>'
        f'</div>'
    )


def _step_html(number: int, title: str, body: str) -> str:
    return f'<section class="afa-result-step"><div class="afa-result-num">{number}</div><div class="afa-result-body"><h3>{html.escape(title)}</h3>{body}</div></section>'


def _text_html(text: str) -> str:
    return '<div class="afa-result-text">' + html.escape(text) + '</div>'


def _render_completed_turn(user_content: str, assistant_msg: dict) -> None:
    trace = assistant_msg.get("tool_trace") or []
    entries = [t for t in trace if t["tool"] == "purchase_register"]
    gate = entries[-1] if entries else None
    cls, summary, is_text = _result_summary(trace, assistant_msg.get("content") or "")
    result_html = _text_html(summary) if is_text else f'<div class="final-box {cls}">{html.escape(summary)}</div>'
    parts = [
        _step_html(1, "시나리오 / 요청", _text_html(user_content)),
        _step_html(2, "구매 에이전트 실행", _text_html("구매 에이전트의 이번 요청 처리가 완료되었습니다.")),
        _step_html(3, "AGENT의 판단 / TOOL CALL", _tool_pills_html(trace)),
        _step_html(4, "AfA GATE", _gate_content(gate["output"], assistant_msg.get("approval_reason")) if gate else _text_html("구매 등록 도구 호출이 없어 게이트 판정이 없습니다.")),
        _step_html(5, "HUMAN APPROVAL", _human_approval_static_html(gate, assistant_msg.get("human_decision"))),
        _step_html(6, "최종 실행 결과", result_html),
    ]
    # One HTML layout owns heading, content and height together.
    st.html('<div class="afa-results">'+"".join(parts)+'</div>')
    with st.container(key=f"afa_details_{idx}"):
        with st.expander("에이전트 답변 전문"):
            st.markdown(assistant_msg.get("content") or "(빈 응답)")
        if trace:
            with st.expander(f"도구 실행 상세 · {len(trace)}개 호출"):
                for i, t in enumerate(trace, 1):
                    st.markdown(f"**{i}. `{t['tool']}`**")
                    st.json({"입력": t.get("input", {}), "출력": t.get("output", {})})


def _render_pending_turn() -> None:
    payload = st.session_state.pending_interrupt
    trace = st.session_state.turn_trace_buffer
    user_content = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
    pills = _tool_pills_html(trace)
    pills += '<div class="afa-wait-tool">⏸ purchase_register · 실행 전 승인 대기 (미실행)</div>'
    parts = [
        _step_html(1, "시나리오 / 요청", _text_html(user_content)),
        _step_html(2, "구매 에이전트 실행", _text_html("구매 등록 직전, 사람의 승인을 기다리고 있습니다.")),
        _step_html(3, "AGENT의 판단 / TOOL CALL", pills),
        _step_html(4, "AfA GATE", _gate_content({"gate_decision": "REQUIRE_APPROVAL", "gate_reason": payload.get("reason"), "gate_facts": payload.get("facts", {})})),
    ]
    st.html('<div class="afa-results">'+"".join(parts)+'</div>')
    with st.container(key="afa_pending_controls"):
        st.html('<div class="afa-approval-heading"><span class="afa-result-num">5</span><h3>05 · HUMAN APPROVAL</h3></div>')
        col1, col2 = st.columns(2)
        approve_clicked = col1.button("승인", key="approve_btn", use_container_width=True, disabled=st.session_state.is_running)
        reject_clicked = col2.button("거절", key="reject_btn", use_container_width=True, disabled=st.session_state.is_running)
    st.html('<div class="afa-results">'+_step_html(6, "06 · 최종 실행 결과", '<div class="final-box pending">⏸ 승인 대기 중 — 현재 요청은 아직 실행되지 않았습니다</div>')+'</div>')
    if payload.get("facts"):
        with st.container(key="afa_details_pending"):
            with st.expander("판정 근거 데이터"):
                st.json(payload["facts"])

    if (approve_clicked or reject_clicked) and not st.session_state.is_running:
        st.session_state.is_running = True
        st.session_state.pending_decision = approve_clicked
        st.rerun()

    if st.session_state.is_running and st.session_state.pending_decision is not None:
        st.markdown(
            """
            <div class="afa-loading-overlay">
              <div class="afa-loading-box">
                <div class="afa-loading-text">에이전트가 실행중입니다</div>
                <div class="afa-loading-bar-track">
                  <div class="afa-loading-bar-fill"></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        decision = st.session_state.pending_decision
        # pending_decision도 run_request와 동일한 이유로 작업이 끝날 때(finally)까지
        # 비우지 않는다 — "진짜 실행 중"과 "고아 상태"를 상태값만으로 구분하기 위해.
        st.session_state._last_human_decision = "approved" if decision else "rejected"
        try:
            result = app.invoke(Command(resume=decision), config=_thread_config())
        except Exception as e:  # noqa: BLE001
            st.error(f"승인 처리 중 오류 발생: {e}. pending 상태는 유지되니 다시 시도해줘.")
        else:
            st.session_state.turn_trace_buffer = _extract_turn_trace(result)
            if "__interrupt__" in result:
                # 정책상 한 턴에 승인 대기가 두 번 이상 걸리는 케이스는 없지만 방어적으로 처리
                _set_pending(result["__interrupt__"][0].value)
            else:
                _finalize_turn(result)
        finally:
            st.session_state.pending_decision = None
            st.session_state.is_running = False
        st.rerun()


def _render_guide_page() -> None:
    st.markdown('<div class="try-it-label page-title">사용 가이드</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="afa-compare-tagline">AfA 데모의 실행 방식과 화면을 확인하세요.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="afa-guide">
          <section class="afa-guide-section">
            <h3>이 데모는 무엇을 보여주나요?</h3>
            <p>AfA(Agent for Agent)는 업무 에이전트가 실제 작업을 실행하기 직전에 개입해 검증하는
            에이전트입니다.<br> 구매 에이전트가 <code>purchase_register</code> 같은 도구를 실행하려 할 때,
            AfA는 최신 정책·예산·구매 이력을 다시 조회해 검증합니다.<br> 에이전트가 스스로 실행 가능하다고
            판단했더라도, AfA는 검증 결과에 따라 ALLOW · REQUIRE_APPROVAL · BLOCK으로 실행 경로를
            통제합니다.</p>
          </section>

          <section class="afa-guide-section">
            <h3>판정 결과 세 가지</h3>
            <div class="afa-guide-badges">
              <span class="afa-guide-badge allow">✓ ALLOW</span>
              <span class="afa-guide-badge-desc">50만원 미만 · 정책 위반 없음 → 즉시 등록</span>
            </div>
            <div class="afa-guide-badges">
              <span class="afa-guide-badge warn">⏸ REQUIRE_APPROVAL</span>
              <span class="afa-guide-badge-desc">50만원 이상(팀장) · 200만원 이상(부서장) · 리퍼비시/중고 · 분할구매 의심 → 사람 승인 후 실행</span>
            </div>
            <div class="afa-guide-badges">
              <span class="afa-guide-badge block">⛔ BLOCK</span>
              <span class="afa-guide-badge-desc">금지 품목 · 잔여 예산 초과 → 실행 차단</span>
            </div>
          </section>

          <section class="afa-guide-section">
            <h3>실행 트레이스 읽는 법 (1~6단계)</h3>
            <ol>
              <li><b>시나리오 / 요청</b> — 실제로 에이전트에 전달된 사용자 메시지</li>
              <li><b>구매 에이전트 실행</b> — 이번 턴 처리가 끝났는지 여부</li>
              <li><b>Agent의 판단 / Tool Call</b> — 에이전트가 순서대로 호출한 도구 목록</li>
              <li><b>AfA Gate</b> — purchase_register 직전에 내려진 판정과 근거(어떤 체크가 판정을 갈랐는지)</li>
              <li><b>Human Approval</b> — 승인/거절 여부, 자동 판정이면 "해당 없음"</li>
              <li><b>최종 실행 결과</b> — 실제로 구매가 등록됐는지, 등록 안 됐으면 에이전트의 답변 원문</li>
            </ol>
          </section>

          <section class="afa-guide-section">
            <h3>시나리오 카드 사용법</h3>
            <p>기본 6개 카드가 ALLOW·REQUIRE_APPROVAL·BLOCK 경로를 대표합니다. 오른쪽 위 "더 보기"를
            누르면 실험에 쓰인 14개 시나리오 전체(경계값, 정보 누락, 존재하지 않는 상품 등)를 확인할 수
            있습니다. 카드를 누르면 새로운 세션(새 임시 DB + 새 대화 스레드)이 시작됩니다. 따라서 이전
            시나리오의 구매 이력이나 대화 상태에 영향을 받지 않고 독립된 초기 상태에서 실행됩니다.</p>
          </section>

          <section class="afa-guide-section">
            <h3>직접 입력할 때 주의할 점</h3>
            <p>이 데모는 사전에 구성된 상품 카탈로그와 사내 구매 정책을 기준으로 동작합니다.
            카탈로그에 없는 상품명이나 소속팀·요청자 정보 없이 요청하면, 에이전트가 등록을 진행하지 않고
            먼저 되묻습니다. 구체적인 판정 흐름을 보고 싶다면 시나리오 카드로 시작하는 걸 권장합니다.</p>
          </section>

          <section class="afa-guide-section">
            <h3>세션 / 새 데모 시작</h3>
            <p>화면 상단의 "세션 xxxxxxxx"는 지금 실행 중인 대화·DB의 식별자입니다. 시나리오 카드를
            누르면 자동으로 새 세션이 시작되고, 직접 입력으로 대화를 이어가다 완전히 처음부터 다시
            시작하고 싶으면 "↻ 새 데모 시작"을 누르면 됩니다.</p>
          </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_compare_page() -> None:
    st.markdown('<div class="try-it-label page-title">Before / After — Gate 유무 비교</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="afa-compare-tagline">Same agent, Same request. 다른 건 실행 게이트 하나뿐입니다.</div>',
        unsafe_allow_html=True,
    )

    results_path = Path(__file__).resolve().parent.parent / "experiments" / "results_repeated.json"
    if not results_path.exists():
        st.markdown(
            '<div class="afa-compare-empty">실험 결과 파일(experiments/results_repeated.json)을 '
            '찾을 수 없습니다. experiments/run_comparison.py를 먼저 실행해주세요.</div>',
            unsafe_allow_html=True,
        )
        return

    data = json.loads(results_path.read_text(encoding="utf-8"))
    summary = data["summary"]
    scenarios = data["scenarios"]
    n = summary["total_scenarios"]
    repeats = summary["repeats_per_scenario"]
    total_runs = n * repeats
    viol_off = summary["violations_without_gate_total"]
    viol_on = summary["violations_with_gate_total"]

    st.markdown(
        f"""
        <div class="afa-compare-stat">
          {n}개 시나리오 × {repeats}회 반복 = 총 {total_runs}회 실행 ·
          정책 위반 <b>{viol_off}/{total_runs}</b> → <b class="ok">{viol_on}/{total_runs}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    example = next((s for s in scenarios if s["scenario_id"] == "S02_팀장승인_필요"), None)
    if example and example["runs"]:
        run = example["runs"][0]
        st.markdown(
            f"""
            <div class="afa-compare-example">
              <div class="afa-compare-example-label">예시 — {html.escape(example["category"])}</div>
              <div class="afa-compare-example-query">"{html.escape(example["user_query"])}"</div>
              <div class="afa-compare-flow">
                <div class="afa-compare-flow-col off">
                  <div class="afa-compare-flow-header off">Gate OFF (게이트 없음)</div>
                  <div class="flow-step neutral">User Request</div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step neutral">Agent 판단<div class="flow-step-sub">"구매 가능합니다"</div></div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step neutral">Tool Call<br><code>purchase_register()</code></div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step danger">즉시 실행</div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step danger">⛔ 정책 위반</div>
                </div>
                <div class="afa-compare-flow-col on">
                  <div class="afa-compare-flow-header on">Gate ON (AfA 적용)</div>
                  <div class="flow-step neutral">User Request</div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step neutral">Agent 판단<div class="flow-step-sub">"구매 가능합니다"</div></div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step neutral">Tool Proposal<br><code>purchase_register()</code></div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step gate">
                    <div class="flow-gate-title">AfA EXECUTION GATE</div>
                    <div class="flow-gate-chips">
                      <span>LIVE DB</span><span>POLICY</span><span>HISTORY</span><span>BUDGET</span>
                    </div>
                  </div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step warn-badge">REQUIRE_APPROVAL</div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step waiting">
                    Human Approval 대기 중
                    <div class="flow-approval-btns"><span class="approve">승인</span><span class="reject">거절</span></div>
                  </div>
                  <div class="flow-arrow">↓</div>
                  <div class="flow-step good">✓ 승인 전 실행 차단</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="afa-compare-table-title">전체 시나리오 결과</div>'
        '<div class="afa-compare-table-sub">14개 시나리오의 3회 반복 실행 결과입니다.</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for s in scenarios:
        agg = s["aggregate"]
        rows.append(
            f'<tr><td>{html.escape(s["scenario_id"])}</td>'
            f'<td>{html.escape(s["category"])}</td>'
            f'<td class="num">{agg["violations_without_gate"]}/{repeats}</td>'
            f'<td class="num ok">{agg["violations_with_gate"]}/{repeats}</td></tr>'
        )
    st.markdown(
        f"""
        <table class="afa-compare-table">
          <thead><tr><th>시나리오</th><th>분류</th><th>위반 (Gate OFF)</th><th>위반 (Gate ON)</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# 메인 영역 — 목업과 동일한 단일 컬럼 (사이드바 없음)
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div class="afa-side-brand"><span class="afa-side-logo">AfA</span>'
        '<span class="afa-side-tagline">Agent Action Gate</span></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="afa_side_nav", border=False):
        for page_key, nav_title, nav_desc in (
            ("demo", "Try it out", "실제 에이전트 실행 데모"),
            ("guide", "사용 가이드", "시나리오와 직접 입력 사용법"),
            ("compare", "Before / After", "Gate 유무 실험 결과"),
        ):
            active = st.session_state.active_page == page_key
            if st.button(
                f"**{nav_title}**\n\n{nav_desc}", key=f"nav_{page_key}",
                use_container_width=True, type="primary" if active else "secondary",
            ):
                st.session_state.active_page = page_key
                st.rerun()
    st.markdown('<div class="afa-side-footer">Wanted AI Championship 2026</div>', unsafe_allow_html=True)

if st.session_state.active_page == "guide":
    _render_guide_page()
    st.stop()
elif st.session_state.active_page == "compare":
    _render_compare_page()
    st.stop()

st.markdown(
    """
    <div class="head">
      <div class="brand">AfA</div>
      <div class="tagline">Agent Action Gate</div>
      <div class="desc">AfA(Agent for Agent)는 업무 에이전트의 도구 실행 직전에 정책 준수 여부를 검증하고, 실행을 통제합니다.<br>
      이 데모에서는 구매 업무 시나리오를 통해 실행 허용·승인 요청·차단 과정을 체험할 수 있습니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# System Layer 포지셔닝 다이어그램 — AfA가 구매 에이전트 전용이 아니라
# 더 큰 시스템의 한 계층이라는 것을 한 번만 보여준다.
st.markdown(
    """
    <div class="position-strip">
      <div class="ps-label">System Layer</div>
      <div class="ps-top"><span>Enterprise AI Service</span></div>
      <div class="agents-row">
        <div class="agent-box">HR Agent</div>
        <div class="agent-box">Finance Agent</div>
        <div class="agent-box active">Purchase Agent<span class="sub">이번 데모</span></div>
        <div class="agent-box">…</div>
      </div>
      <div class="arrows-row" aria-hidden="true">↓</div>
      <div class="gate-layer">
        <div class="t">AfA — 실행 전 정책 검증</div>
        <div class="s">업무 에이전트와 분리된 검증 계층 · 이 데모에서는 구매 에이전트에 적용합니다</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="try-it-label">Try it out</div>', unsafe_allow_html=True)

demo_disabled = st.session_state.is_running or st.session_state.pending_interrupt is not None

# Markdown emphasis provides three separately styled text levels inside one real button.
def _card_text(value: str) -> str:
    for char in ("\\", "*", "_", "[", "]", "`", "<", ">"):
        value = value.replace(char, "\\" + char)
    return value

with st.container(key="afa_sandbox", border=False):
    with st.container(key="afa_session_bar", border=False):
        session_id_col, session_reset_col = st.columns([5, 1.5], gap="small", vertical_alignment="center")
        with session_id_col:
            st.markdown(
                f'<div class="session-bar-label">세션 {st.session_state.thread_id[:8]}</div>',
                unsafe_allow_html=True,
            )
        with session_reset_col:
            if st.button(
                "↻ 새 데모 시작", key="reset_session_btn",
                use_container_width=True, disabled=demo_disabled,
            ):
                _reset_session()
                st.rerun()
    prompt_label_col, prompt_toggle_col = st.columns([8.9, 1.1], gap="small", vertical_alignment="center")
    with prompt_label_col:
        st.markdown(
            '<div class="prompt-label">시나리오를 선택하거나, 구매 요청을 직접 입력해보세요</div>',
            unsafe_allow_html=True,
        )
    with prompt_toggle_col:
        with st.container(key="afa_toggle_btn", border=False):
            toggle_label = "접기 ▲" if st.session_state.show_all_scenarios else "더 보기 ▾"
            if st.button(toggle_label, key="scenario_toggle", use_container_width=True, disabled=demo_disabled):
                st.session_state.show_all_scenarios = not st.session_state.show_all_scenarios
                st.rerun()
    # 실제 Streamlit 레이아웃 요소로 안내문과 카드 사이 공간 확보.
    with st.container(height=34, border=False, key="afa_heading_gap"):
        pass

    with st.container(key="afa_scenario_grid", border=False):
        visible_scenarios = SANDBOX_SCENARIOS if st.session_state.show_all_scenarios else SANDBOX_SCENARIOS[:6]
        for row_start in range(0, len(visible_scenarios), 3):
            cols = st.columns(3, gap="small")
            for j, scenario in enumerate(visible_scenarios[row_start:row_start + 3]):
                with cols[j]:
                    label = (
                        f"**{_card_text(scenario['tag'])}** "
                        f"*{_card_text(scenario['title'])}* "
                        f"{_card_text(scenario['example'])}"
                    )
                    if st.button(
                        label, use_container_width=True,
                        disabled=demo_disabled, key=f"sc_{scenario['id']}",
                    ):
                        _start_run("scenario", scenario)

    with st.container(key="afa_freeform", border=False):
        label_col, input_col, run_col = st.columns([1.25, 7.1, 0.65], gap="small", vertical_alignment="center")
        with label_col:
            st.markdown(
                '<div class="freeform-label">구매 요청 직접 입력'
                '<span class="freeform-help" data-tooltip="실제 상품 카탈로그와 사내 구매 정책을 기반으로 '
                '판단합니다.\n정보가 부족하거나 애매하면 에이전트가 되묻습니다.\n시나리오 카드 또는 직접 '
                '입력으로 실행 흐름을 확인해보세요.">?</span></div>',
                unsafe_allow_html=True,
            )
        with input_col:
            free_text = st.text_input(
                "직접 입력", key="freeform_input", label_visibility="collapsed",
                placeholder="예: 인사팀에서 모니터 하나 사줘",
                disabled=demo_disabled,
            )
        with run_col:
            if st.button("실행", use_container_width=True, disabled=demo_disabled, key="freeform_run"):
                if free_text and free_text.strip():
                    _start_run("freeform", free_text.strip())

st.markdown("")

# ── 실행 트레이스 ──
messages = st.session_state.messages
has_completed = len(messages) >= 2 and any(m["role"] == "assistant" for m in messages)

if has_completed:
    idx = 0
    while idx + 1 < len(messages):
        if messages[idx]["role"] == "user" and messages[idx + 1]["role"] == "assistant":
            _render_completed_turn(messages[idx]["content"], messages[idx + 1])
        idx += 2

# ── 승인 대기 중이면: 새 입력 받지 않고 트레이스 안에서 승인/거절 UI만 표시 (불변조건 1) ──
if st.session_state.pending_interrupt:
    _render_pending_turn()
elif has_completed:
    # ── 응답이 나온 뒤에만 후속 입력 표시 (승인 대기 중에는 숨김) ──
    # (목업엔 없는 유일한 실제 기능 추가 — 에이전트가 "소속 팀을 알려주세요" 같은
    #  되물음을 했을 때 답할 방법이 없으면 대화가 막히기 때문에 최소한으로 유지함)
    # 트레이스 카드(_trace_step)와 좌측 시작점을 맞추기 위해 동일한 비율의
    # number_col/body_col 분할을 그대로 사용 (번호 칸은 비워서 여백으로만 씀)
    _, followup_col = st.columns([0.35, 9.65], gap="small")
    with followup_col:
        st.markdown(
            '<div class="continue-label">에이전트가 추가 정보를 물어보면 여기서 이어서 답할 수 있습니다</div>',
            unsafe_allow_html=True,
        )
        user_input = st.chat_input(
            "이어서 답변하기",
            key="followup_input",
            disabled=demo_disabled,
        )

    # 자유 입력 박스와 동일하게 _start_run("freeform", ...)을 거친다 — 여기서 바로
    # _submit_user_turn을 부르면 로딩 모달이 뜨기 전에 무거운 app.invoke()가 시작돼버린다.
    if user_input and user_input.strip():
        _start_run("freeform", user_input.strip())

if st.session_state.is_running and st.session_state.run_request is not None:
    # run_request가 있을 때만 이 블록이 처리한다 — is_running만 보고 들어가면
    # 승인/거절 흐름(pending_decision)이 is_running=True로 만든 경우까지 여기서
    # 가로채 is_running을 꺼버려서 _render_pending_turn이 처리할 기회를 뺏는 버그가 있었음.
    # 모달은 position:fixed 전체화면 오버레이라 DOM 위치는 화면 표시와 무관하다 — 여기
    # (실행 트레이스가 이미 다 그려진 뒤)에 둬야, 블로킹 호출 도중에도 과거 기록이 이번
    # 스크립트 실행에서 이미 한 번 다시 그려진 상태라 화면에서 사라지지 않는다.
    st.markdown(
        """
        <div class="afa-loading-overlay">
          <div class="afa-loading-box">
            <div class="afa-loading-text">에이전트가 실행중입니다</div>
            <div class="afa-loading-bar-track">
              <div class="afa-loading-bar-fill"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    req = st.session_state.run_request
    # run_request는 작업이 끝날 때(finally)까지 비우지 않는다 — 중간에 비워버리면
    # "진짜 실행 중"과 "run_request를 못 받은 고아 상태"가 is_running=True라는
    # 점에서 똑같아 보여서, 아래에 있는 고아-상태 자가복구 로직이 실행 중인
    # 작업까지 오진할 여지가 생긴다. 끝까지 값을 들고 있으면 그 둘이 상태값만으로
    # 명확히 구분된다.
    try:
        if req["kind"] == "scenario":
            _run_sandbox_scenario(req["payload"])
        elif req["kind"] == "freeform":
            _submit_user_turn(req["payload"])
    finally:
        st.session_state.run_request = None
        st.session_state.is_running = False
    # _run_sandbox_scenario/_submit_user_turn은 보통 끝나면서 st.rerun()을 호출해
    # (RerunException으로) 이 아래로 내려오지 않는다. 그런데 app.invoke()가 예외를
    # 던지고 내부에서 st.error만 찍고 정상 반환하는 경로(예: _submit_user_turn의
    # except 분기)는 rerun을 안 걸기 때문에, 그 경우 로딩 모달이 화면에 그대로
    # 남는 채로 멈춰버린다. 여기까지 정상적으로 도달했다는 건 위에서 rerun이 아직
    # 안 걸렸다는 뜻이므로 안전망으로 한 번 더 강제 rerun한다.
    st.rerun()