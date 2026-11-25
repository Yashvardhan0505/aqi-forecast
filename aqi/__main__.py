"""Command line: python -m aqi {run,site,backtest,demo}."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import openaq, pipeline, weather
from .config import load_configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aqi", description="Daily PM2.5 / AQI forecast")
    parser.add_argument("--config", help="path to config.toml (default: the one in the repo)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="daily job: download data, forecast, score, build the pages")
    p_run.add_argument("--force-backtest", action="store_true", help="rerun the backtests today")
    sub.add_parser("site", help="rebuild the pages in docs/ from stored data (no downloads)")
    sub.add_parser("backtest", help="rerun the backtests on stored data and rebuild the pages")
    p_demo = sub.add_parser("demo", help="offline demo with synthetic data (no API key needed)")
    p_demo.add_argument("--days", type=int, default=16, help="simulated daily runs (default 16)")
    p_demo.add_argument("--out", default="demo", help="output folder (default ./demo)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfgs = load_configs(args.config)

    try:
        if args.command == "run":
            oa_http = openaq.make_http(os.environ.get("OPENAQ_API_KEY", "").strip())
            wx_http = weather.make_http()
            pipeline.run(cfgs, oa_http, wx_http, backtest="force" if args.force_backtest else "auto")
            print(f"Done. Pages written to {cfgs[0].site_root}")
        elif args.command == "site":
            pipeline.rebuild(cfgs, backtest="skip")
            print(f"Pages written to {cfgs[0].site_root}")
        elif args.command == "backtest":
            pipeline.rebuild(cfgs, backtest="force")
            print("Backtests saved in each city's data folder")
        elif args.command == "demo":
            from .demo import run_demo
            page = run_demo(args.out, days=args.days)
            print(f"Demo page: {page.resolve()}")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
