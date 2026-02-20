"""
Groq AI Engine — Investment Committee Reports.
RAG-style: Real data context → Groq LLM → Structured JSON response.
With retry logic for 429 rate limits + fallback analysis.
"""
import json
import time
import logging
from typing import Optional
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL
from src.ai.prompts import INVESTMENT_COMMITTEE_PROMPT, build_analysis_context

logger = logging.getLogger("matrix_trader.ai.groq_engine")


class GroqEngine:
    """Groq-powered AI analysis engine with retry logic."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or GROQ_API_KEY
        self.client = None
        self._rate_limited = False
        self._call_count = 0
        self._max_calls_per_scan = 15
        self._retry_count = 0
        self._max_retries = 3
        self._consecutive_429s = 0
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Groq client init failed: {e}")

    @property
    def available(self) -> bool:
        if self._rate_limited:
            return False
        if self._call_count >= self._max_calls_per_scan:
            logger.info(f"Groq call limit reached ({self._max_calls_per_scan}/scan)")
            return False
        return self.client is not None

    def _handle_rate_limit(self, e: Exception) -> bool:
        """Handle 429 rate limit errors with retry logic.
        Returns True if should retry, False if permanently rate-limited."""
        self._consecutive_429s += 1
        if self._consecutive_429s >= 3:
            self._rate_limited = True
            logger.warning(f"Groq permanently rate limited after {self._consecutive_429s} consecutive 429s")
            return False
        
        # Parse retry-after header from error if available
        wait_time = 60  # Default: wait 60s for rate limit reset
        err_str = str(e)
        if "retry" in err_str.lower():
            # Try to extract retry-after seconds from the error message
            import re
            match = re.search(r'(\d+\.?\d*)\s*s', err_str)
            if match:
                wait_time = min(int(float(match.group(1))) + 2, 90)
        
        logger.info(f"Groq 429 — waiting {wait_time}s before retry ({self._consecutive_429s}/3)")
        time.sleep(wait_time)
        return True

    def _call_groq(self, messages: list, temperature: float, max_tokens: int) -> Optional[str]:
        """Call Groq API with retry logic for 429 errors."""
        for attempt in range(self._max_retries):
            try:
                self._call_count += 1
                response = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._consecutive_429s = 0  # Reset on success
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    should_retry = self._handle_rate_limit(e)
                    if not should_retry:
                        return None
                    continue  # Retry
                else:
                    raise  # Re-raise non-rate-limit errors
        return None

    def _safe_json_parse(self, text: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        if not text:
            return None
        # Strip markdown json block
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON within the text
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
        logger.warning(f"Could not parse JSON from Groq response")
        return None

    def get_investment_analysis(
        self,
        symbol: str,
        direction: str,
        indicators: dict,
        risk_mgmt: dict,
        confidence: int,
        mtf_result: dict = None,
        sentiment: dict = None,
        smart_money: dict = None,
        macro: dict = None,
        fundamental: dict = None,
        news: list = None,
        is_bist: bool = False,
    ) -> Optional[dict]:
        """
        Generate AI investment committee report.
        Returns structured dict with karar, guven, hedef_fiyat, yorum, etc.
        """
        if not self.available:
            logger.warning("Groq not available, skipping AI analysis")
            return None

        context = build_analysis_context(
            symbol, direction, indicators, risk_mgmt, confidence,
            mtf_result, sentiment, smart_money, macro, fundamental, news, is_bist
        )

        try:
            text = self._call_groq(
                messages=[
                    {"role": "system", "content": INVESTMENT_COMMITTEE_PROMPT},
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            if not text:
                return None
            result = self._safe_json_parse(text)
            if result:
                logger.info(f"AI analysis for {symbol}: {result.get('karar', 'N/A')}")
                return result
            else:
                # Return raw text as fallback
                return {"yorum": text[:500], "karar": direction, "guven": confidence}

        except Exception as e:
            logger.error(f"Groq analysis failed for {symbol}: {e}")
            return None

    def get_summary_report(self, signals: list[dict], market_type: str = "CRYPTO") -> Optional[str]:
        """Generate a daily summary report of all signals."""
        if not self.available or not signals:
            return None

        signal_text = "\n".join(
            f"• {s.get('symbol', '?')} {s.get('direction', '?')} ({s.get('confidence', 0)}%) - {s.get('tier_name', '')}"
            for s in signals[:20]
        )

        prompt = f"""Bugünkü {market_type} tarama sonuçlarını özetle:

{signal_text}

Kısa ve net bir piyasa özeti yaz (max 200 kelime). Genel trend, dikkat çeken sinyaller ve piyasa durumu hakkında yorum yap. TÜRKÇE yaz."""

        try:
            response = self._call_groq(
                messages=[
                    {"role": "system", "content": "Sen kısa ve öz piyasa yorumları yapan bir finansal analistsin."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=500,
            )
            return response

        except Exception as e:
            logger.error(f"Groq summary failed: {e}")
            return None

    @staticmethod
    def generate_fallback_analysis(
        symbol: str,
        direction: str,
        indicators: dict,
        risk_mgmt: dict,
        confidence: int,
        sentiment: dict = None,
        smart_money: dict = None,
        macro: dict = None,
    ) -> dict:
        """Generate rule-based AI analysis when Groq is unavailable.
        Uses actual indicator data — no random or fabricated values."""
        price = indicators.get('currentPrice', 0)
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 0)
        macd_hist = indicators.get('macd_hist', 0)
        bb_pctb = indicators.get('bb_pctb', 0.5)
        vol_ratio = indicators.get('volume_ratio', 1.0)
        ema9 = indicators.get('ema9', 0)
        ema21 = indicators.get('ema21', 0)

        # Build technical synthesis from real data
        teknik_parts = []
        if rsi > 70:
            teknik_parts.append(f"RSI {rsi:.0f} aşırı alım bölgesinde")
        elif rsi < 30:
            teknik_parts.append(f"RSI {rsi:.0f} aşırı satım bölgesinde")
        else:
            teknik_parts.append(f"RSI {rsi:.0f} nötr")

        if macd_hist and macd_hist > 0:
            teknik_parts.append("MACD pozitif momentum")
        elif macd_hist and macd_hist < 0:
            teknik_parts.append("MACD negatif momentum")

        if adx > 25:
            teknik_parts.append(f"ADX {adx:.0f} güçlü trend")
        else:
            teknik_parts.append(f"ADX {adx:.0f} zayıf trend")

        if vol_ratio > 2.0:
            teknik_parts.append(f"hacim {vol_ratio:.1f}x yüksek")
        elif vol_ratio < 0.5:
            teknik_parts.append(f"hacim {vol_ratio:.1f}x düşük")

        teknik_sentez = ", ".join(teknik_parts)

        # Build comment from real data
        yorum_parts = [f"{symbol} {direction} yönünde sinyal. {teknik_sentez}."]

        if ema9 and ema21:
            if ema9 > ema21:
                yorum_parts.append("Kısa vadeli EMA yukarı kesişimde.")
            else:
                yorum_parts.append("Kısa vadeli EMA aşağı kesişimde.")

        # Macro assessment
        makro_etki = "Makro veri mevcut değil"
        if macro and macro.get('summary'):
            makro_etki = macro['summary']

        # Smart money assessment
        akilli_para = "Akıllı para verisi mevcut değil"
        if smart_money:
            sm_dir = smart_money.get('direction', 'N/A')
            vol_anom = smart_money.get('volume_anomaly', {})
            if vol_anom.get('anomaly'):
                akilli_para = f"Hacim anomalisi tespit edildi (Z={vol_anom.get('z_score', 0):.1f}), yön: {sm_dir}"
            else:
                akilli_para = f"Normal hacim profili, yön: {sm_dir}"

        # Risks based on real data
        riskler = []
        if rsi > 70:
            riskler.append(f"RSI {rsi:.0f} — aşırı alım, geri çekilme riski")
        if rsi < 30:
            riskler.append(f"RSI {rsi:.0f} — aşırı satım, dip tuzağı riski")
        if vol_ratio < 0.7:
            riskler.append(f"Düşük hacim ({vol_ratio:.1f}x) — sahte kırılım riski")
        if adx < 20:
            riskler.append(f"Zayıf trend (ADX {adx:.0f}) — yön belirsizliği")
        if not riskler:
            riskler.append("Genel piyasa riski")

        # Opportunities based on real data
        firsatlar = []
        if adx > 30 and vol_ratio > 1.5:
            firsatlar.append(f"Güçlü trend (ADX {adx:.0f}) + yüksek hacim ({vol_ratio:.1f}x)")
        if bb_pctb < 0.2 and direction == "BUY":
            firsatlar.append("Bollinger alt bandına yakın — dip fırsatı")
        if bb_pctb > 0.8 and direction == "SELL":
            firsatlar.append("Bollinger üst bandına yakın — tepe sinyali")
        if not firsatlar:
            firsatlar.append(f"R/R oranı: 1:{risk_mgmt.get('reward_risk', 'N/A')}")

        targets = risk_mgmt.get('targets', {})

        return {
            "karar": direction.replace("BUY", "AL").replace("SELL", "SAT"),
            "guven": confidence,
            "hedef_fiyat": {
                "kisa_vade": targets.get('t1'),
                "orta_vade": targets.get('t2'),
            },
            "stop_loss": risk_mgmt.get('stop_loss'),
            "yorum": " ".join(yorum_parts),
            "teknik_sentez": teknik_sentez,
            "makro_etki": makro_etki,
            "akilli_para_yorum": akilli_para,
            "riskler": riskler,
            "firsatlar": firsatlar,
            "strateji": f"Giriş: {price}, SL: {risk_mgmt.get('stop_loss')}, T1: {targets.get('t1')}, T2: {targets.get('t2')}",
            "zaman_dilimi": "Kısa-orta vade",
            "onem_notu": "",
            "_fallback": True,  # Flag to indicate this is rule-based, not AI
        }
