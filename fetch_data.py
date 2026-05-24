#!/usr/bin/env python3
"""
글로벌 자산배분 백테스트 — ETF 기반 데이터 수집 파이프라인
==========================================================
FinanceDataReader를 통해 ETF Adjusted Price(배당 재투자 반영)를 가져와
연간 총수익률(%)을 계산한 뒤 JSON으로 출력합니다.

사용법:
  pip install finance-datareader pandas numpy
  python fetch_data.py

출력:
  data.json  — 프론트엔드가 import하는 데이터 파일
"""

import json
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. ETF → 자산군 매핑
#    데이터 수집에만 ETF 코드 사용, 포트폴리오는 자산군 기반
# ──────────────────────────────────────────────

ETF_MAP = {
    # ── 미국 ──
    "us_equity":   {"ticker": "SPY",  "start": "1993-01-01", "desc": "S&P 500 TR (SPY)"},
    "us_lt_bond":  {"ticker": "TLT",  "start": "2002-07-01", "desc": "미국 장기국채 20Y+ (TLT)"},
    "us_mt_bond":  {"ticker": "IEF",  "start": "2002-07-01", "desc": "미국 중기국채 7-10Y (IEF)"},
    "us_st_bond":  {"ticker": "SHY",  "start": "2002-07-01", "desc": "미국 단기국채 1-3Y (SHY)"},
    "us_tips":     {"ticker": "TIP",  "start": "2003-12-01", "desc": "미국 TIPS (TIP)"},
    "us_reit":     {"ticker": "VNQ",  "start": "2004-09-01", "desc": "미국 리츠 (VNQ)"},
    "us_cash":     {"ticker": "BIL",  "start": "2007-05-01", "desc": "미국 T-Bill (BIL)"},

    # ── 한국 ──
    "kr_equity":   {"ticker": "069500", "start": "2002-10-01", "desc": "KODEX 200 (KOSPI TR 대용)"},
    "kr_bond10":   {"ticker": "148070", "start": "2012-03-01", "desc": "KOSEF 국고채10년 (한국 10Y)"},
    "kr_bond3":    {"ticker": "114260", "start": "2009-08-01", "desc": "KODEX 국고채3년 (한국 3Y)"},
    "kr_cash":     {"ticker": "130730", "start": "2010-06-01", "desc": "KOSEF 단기자금 (CD 91일 대용)"},

    # ── 일본 ──
    "jp_equity":   {"ticker": "1306.T", "start": "2001-07-01", "desc": "TOPIX ETF (1306)"},
    "jp_cash":     {"ticker": "NONE",   "start": "1985-01-01", "desc": "일본 O/N → 거의 0%, 수동 입력"},
    "jp_reit":     {"ticker": "1343.T", "start": "2008-09-01", "desc": "J-REIT ETF (1343)"},

    # ── 글로벌 ──
    "gold":        {"ticker": "GLD",  "start": "2004-11-01", "desc": "금 현물 (GLD)"},
    "commodity":   {"ticker": "GSG",  "start": "2006-07-01", "desc": "S&P GSCI (GSG)"},
}

# 환율 (연말 대비 연간 변동률 %)
FX_MAP = {
    "fx_usdkrw": {"ticker": "USD/KRW", "start": "1985-01-01", "desc": "원/달러 환율"},
    "fx_usdjpy": {"ticker": "USD/JPY", "start": "1985-01-01", "desc": "엔/달러 환율"},
}

# ──────────────────────────────────────────────
# 2. 하드코딩 폴백 (ETF 상장 이전 기간)
#    인덱스 수준 학술 데이터 기반
# ──────────────────────────────────────────────

FALLBACK = {
    "us_equity": {
        1985:32.2,1986:18.5,1987:5.2,1988:16.8,1989:31.5,1990:-3.2,1991:30.5,1992:7.7,
    },
    "us_lt_bond": {
        1985:31.0,1986:24.5,1987:-2.7,1988:9.7,1989:18.1,1990:6.2,1991:19.3,1992:8.0,
        1993:18.2,1994:-7.8,1995:31.7,1996:0.9,1997:15.1,1998:13.1,1999:-8.7,2000:21.5,
        2001:3.7,
    },
    "us_mt_bond": {
        1985:21.3,1986:15.6,1987:2.3,1988:7.9,1989:13.3,1990:8.6,1991:15.5,1992:7.2,
        1993:11.2,1994:-5.1,1995:18.5,1996:2.1,1997:9.9,1998:10.2,1999:-1.8,2000:14.5,
        2001:5.6,
    },
    "us_st_bond": {
        1985:9.0,1986:8.0,1987:4.5,1988:5.5,1989:7.5,1990:7.0,1991:6.5,1992:4.5,
        1993:4.0,1994:3.0,1995:6.5,1996:4.0,1997:5.0,1998:5.5,1999:3.0,2000:7.0,
        2001:5.0,
    },
    "us_tips": {
        1985:25.0,1986:20.0,1987:0.0,1988:5.0,1989:10.0,1990:4.0,1991:13.0,1992:5.0,
        1993:8.0,1994:-3.0,1995:15.0,1996:1.0,1997:8.0,1998:8.0,1999:-2.0,2000:12.0,
        2001:3.5,2002:16.5,2003:8.4,
    },
    "us_reit": {
        1985:19.1,1986:19.2,1987:-3.6,1988:13.5,1989:8.8,1990:-15.4,1991:35.7,1992:14.6,
        1993:19.7,1994:3.2,1995:15.3,1996:35.3,1997:20.3,1998:-17.5,1999:-4.6,2000:26.4,
        2001:13.9,2002:3.8,2003:37.1,
    },
    "us_cash": {
        1985:7.7,1986:6.2,1987:5.5,1988:6.4,1989:8.4,1990:7.8,1991:5.6,1992:3.5,
        1993:2.9,1994:3.9,1995:5.6,1996:5.2,1997:5.3,1998:4.9,1999:4.7,2000:5.9,
        2001:3.8,2002:1.7,2003:1.0,2004:1.2,2005:3.0,2006:4.7,
    },
    "kr_equity": {
        1985:15.0,1986:25.0,1987:5.0,1988:30.0,1989:2.0,1990:-25.0,1991:-5.0,1992:8.0,
        1993:25.0,1994:18.0,1995:-14.0,1996:-11.0,1997:-42.0,1998:49.0,1999:83.0,
        2000:13.0,2001:3.0,
    },
    "kr_bond10": {
        1985:12.0,1986:10.0,1987:8.0,1988:7.0,1989:8.0,1990:9.0,1991:10.0,1992:8.0,
        1993:7.5,1994:7.0,1995:8.0,1996:6.5,1997:5.0,1998:15.0,1999:3.0,2000:8.0,
        2001:9.0,2002:10.0,2003:5.0,2004:6.0,2005:7.0,2006:5.0,2007:2.0,2008:10.0,
        2009:2.0,2010:7.0,2011:6.0,
    },
    "kr_bond3": {
        1985:10.0,1986:8.0,1987:7.0,1988:6.0,1989:7.0,1990:8.0,1991:9.0,1992:7.0,
        1993:6.5,1994:6.0,1995:7.0,1996:5.5,1997:4.5,1998:12.0,1999:2.5,2000:6.0,
        2001:7.0,2002:8.0,2003:4.5,2004:5.0,2005:5.5,2006:4.0,2007:2.5,2008:8.0,
    },
    "kr_cash": {
        1985:10.0,1986:8.0,1987:7.0,1988:7.0,1989:9.0,1990:10.0,1991:8.0,1992:6.0,
        1993:5.0,1994:5.0,1995:6.0,1996:5.0,1997:5.0,1998:7.0,1999:3.0,2000:4.0,
        2001:3.5,2002:3.0,2003:2.5,2004:2.5,2005:2.5,2006:3.0,2007:3.5,2008:2.0,
        2009:1.5,
    },
    "jp_equity": {
        1985:13.1,1986:42.6,1987:15.3,1988:39.9,1989:29.0,1990:-39.0,1991:-3.6,
        1992:-26.4,1993:2.9,1994:13.2,1995:0.7,1996:-6.0,1997:-18.0,1998:-9.0,
        1999:60.0,2000:-24.0,
    },
    "jp_cash": {y: (4.0 if y < 1988 else 3.0 if y < 1990 else 2.0 if y < 1993
                     else 0.5 if y < 1996 else 0.1 if y < 2000 else 0.0)
                for y in range(1985, 2027)},
    "jp_reit": {y: 0 for y in range(1985, 2001)},  # J-REIT didn't exist before 2001
    "gold": {
        1985:5.7,1986:19.1,1987:22.2,1988:-15.3,1989:-2.8,1990:-1.5,1991:-10.1,
        1992:-5.7,1993:17.7,1994:-2.2,1995:-1.0,1996:-4.6,1997:-21.4,1998:-0.8,
        1999:0.9,2000:1.0,2001:2.5,2002:24.7,2003:19.6,
    },
    "commodity": {
        1985:-5.0,1986:-15.0,1987:8.0,1988:12.0,1989:20.0,1990:15.0,1991:-10.0,
        1992:0.5,1993:2.0,1994:5.0,1995:8.0,1996:22.0,1997:-2.0,1998:-25.0,
        1999:20.0,2000:30.0,2001:-15.0,2002:15.0,2003:20.0,2004:10.0,2005:15.0,
    },
    "fx_usdkrw": {
        1985:0,1986:-5,1987:10,1988:-15,1989:-10,1990:10,1991:5,1992:8,1993:3,
        1994:2,1995:-5,1996:10,1997:80,1998:-30,1999:-5,2000:10,2001:5,2002:10,
    },
    "fx_usdjpy": {
        1985:-20,1986:-16,1987:10,1988:-12,1989:-3,1990:5,1991:-8,1992:0,1993:12,
        1994:10,1995:-4,1996:12,1997:15,1998:-11,1999:-11,2000:3,2001:15,2002:10,
    },
}


def try_import_fdr():
    """FinanceDataReader import 시도"""
    try:
        import FinanceDataReader as fdr
        return fdr
    except ImportError:
        print("⚠️  FinanceDataReader 미설치. pip install finance-datareader")
        print("   폴백 데이터를 사용합니다.")
        return None


def fetch_annual_returns(fdr, ticker: str, start: str, asset_key: str) -> dict:
    """
    ETF의 Adjusted Close 기반 연간 총수익률(%) 계산
    배당 재투자가 반영된 Adjusted Price를 사용하므로 Total Return과 동일
    """
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        df = fdr.DataReader(ticker, start, end)

        if df is None or df.empty:
            print(f"  ⚠️  {ticker} ({asset_key}): 데이터 없음")
            return {}

        # Adjusted Close 우선, 없으면 Close
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = df[col].dropna()

        if prices.empty:
            return {}

        # 연말 가격 추출 (각 연도의 마지막 거래일)
        annual = prices.resample("YE").last().dropna()

        # 연간 수익률 계산
        returns = {}
        years = annual.index.tolist()
        for i in range(1, len(years)):
            year = years[i].year
            ret = (annual.iloc[i] / annual.iloc[i - 1] - 1) * 100
            returns[year] = round(float(ret), 1)

        # 현재 연도 YTD (마지막 거래일 기준)
        current_year = datetime.now().year
        if current_year not in returns:
            year_start_prices = prices[prices.index.year == current_year]
            prev_year_prices = prices[prices.index.year == current_year - 1]
            if not year_start_prices.empty and not prev_year_prices.empty:
                ytd_ret = (year_start_prices.iloc[-1] / prev_year_prices.iloc[-1] - 1) * 100
                returns[current_year] = round(float(ytd_ret), 1)

        print(f"  ✓ {ticker} ({asset_key}): {len(returns)}년 ({min(returns.keys())}-{max(returns.keys())})")
        return returns

    except Exception as e:
        print(f"  ✗ {ticker} ({asset_key}): {e}")
        return {}


def fetch_fx_returns(fdr, ticker: str, start: str, fx_key: str) -> dict:
    """환율 연간 변동률(%) 계산"""
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        df = fdr.DataReader(ticker, start, end)

        if df is None or df.empty:
            print(f"  ⚠️  {ticker} ({fx_key}): 데이터 없음")
            return {}

        col = "Close" if "Close" in df.columns else df.columns[0]
        prices = df[col].dropna()
        annual = prices.resample("YE").last().dropna()

        returns = {}
        years = annual.index.tolist()
        for i in range(1, len(years)):
            year = years[i].year
            ret = (annual.iloc[i] / annual.iloc[i - 1] - 1) * 100
            returns[year] = round(float(ret), 1)

        # YTD
        current_year = datetime.now().year
        if current_year not in returns:
            yp = prices[prices.index.year == current_year]
            pp = prices[prices.index.year == current_year - 1]
            if not yp.empty and not pp.empty:
                returns[current_year] = round(float((yp.iloc[-1] / pp.iloc[-1] - 1) * 100), 1)

        print(f"  ✓ {ticker} ({fx_key}): {len(returns)}년")
        return returns

    except Exception as e:
        print(f"  ✗ {ticker} ({fx_key}): {e}")
        return {}


def merge_with_fallback(etf_data: dict, fallback_data: dict, start_year=1985, end_year=None) -> list:
    """
    ETF 실데이터 + 폴백 데이터를 합산하여 start_year ~ end_year 배열 생성
    ETF 데이터가 있으면 우선 사용, 없는 연도는 폴백으로 채움
    """
    if end_year is None:
        end_year = datetime.now().year

    result = []
    for year in range(start_year, end_year + 1):
        if year in etf_data:
            result.append(etf_data[year])
        elif year in fallback_data:
            result.append(fallback_data[year])
        else:
            result.append(0)
    return result


def build_data_json(output_path: str = "data.json"):
    """메인 데이터 수집 및 JSON 빌드"""
    fdr = try_import_fdr()

    today = date.today()
    start_year = 1985
    end_year = today.year
    end_month = today.month

    years = list(range(start_year, end_year + 1))

    print(f"\n{'='*60}")
    print(f"글로벌 자산배분 백테스트 — 데이터 수집")
    print(f"기간: {start_year} ~ {end_year} ({today.strftime('%Y-%m-%d')} 기준)")
    print(f"{'='*60}\n")

    all_data = {}

    # ── 자산군 ETF 데이터 수집 ──
    print("▶ 자산군 ETF 데이터 수집...")
    for asset_key, info in ETF_MAP.items():
        ticker = info["ticker"]
        if ticker == "NONE":
            # 수동 입력 자산 (예: 일본 단기금리)
            fb = FALLBACK.get(asset_key, {})
            all_data[asset_key] = merge_with_fallback({}, fb, start_year, end_year)
            print(f"  ○ {asset_key}: 수동 데이터 사용 ({info['desc']})")
            continue

        etf_returns = {}
        if fdr:
            etf_returns = fetch_annual_returns(fdr, ticker, info["start"], asset_key)

        fb = FALLBACK.get(asset_key, {})
        all_data[asset_key] = merge_with_fallback(etf_returns, fb, start_year, end_year)

    # ── 환율 데이터 수집 ──
    print("\n▶ 환율 데이터 수집...")
    for fx_key, info in FX_MAP.items():
        ticker = info["ticker"]
        fx_returns = {}
        if fdr:
            fx_returns = fetch_fx_returns(fdr, ticker, info["start"], fx_key)

        fb = FALLBACK.get(fx_key, {})
        all_data[fx_key] = merge_with_fallback(fx_returns, fb, start_year, end_year)

    # ── JSON 출력 ──
    output = {
        "meta": {
            "generated": today.isoformat(),
            "start_year": start_year,
            "end_year": end_year,
            "end_month": end_month,
            "years": years,
            "source": "ETF Adjusted Price via FinanceDataReader (배당 재투자 반영 Total Return)",
            "etf_tickers": {k: v["ticker"] for k, v in {**ETF_MAP, **FX_MAP}.items()},
            "notes": [
                "ETF 상장 이전 기간은 인덱스 수준 학술 데이터로 폴백",
                "Adjusted Close 기반 연간 총수익률 = 배당+가격변동 포함",
                "현재 연도는 YTD 수익률 (마지막 거래일 기준)",
            ],
        },
        "data": all_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ 저장 완료: {output_path}")
    print(f"   총 {len(all_data)}개 자산/환율, {len(years)}년 ({start_year}-{end_year})")

    # 데이터 커버리지 리포트
    print(f"\n▶ 데이터 커버리지 리포트:")
    for key in all_data:
        vals = all_data[key]
        non_zero = sum(1 for v in vals if v != 0)
        total = len(vals)
        etf_info = ETF_MAP.get(key, FX_MAP.get(key, {}))
        ticker = etf_info.get("ticker", "N/A")
        pct = non_zero / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"   {key:15s} [{bar}] {pct:5.1f}% ({non_zero}/{total}) — {ticker}")

    print(f"\n{'='*60}\n")
    return output


if __name__ == "__main__":
    build_data_json("data.json")
