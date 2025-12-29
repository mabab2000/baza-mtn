from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, MetaData
import os
from dotenv import load_dotenv
import logging
import json
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

engine = create_async_engine(DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'), echo=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

app = FastAPI()

class ChatRequest(BaseModel):
    phone: str
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async with async_session() as session:
        metadata = MetaData()
        async with engine.begin() as conn:
            await conn.run_sync(metadata.reflect)
        users = metadata.tables.get("users")
        if users is None:
            raise HTTPException(status_code=500, detail="Users table not found")

        # Log users table metadata (column name and type)
        cols = [{"name": c.name, "type": str(c.type)} for c in users.columns]
        logger.info("Users table metadata: %s", cols)

        # Tables to fetch and log
        tables_to_log = [
            "airtime_balance",
            "main_category",
            "sub_category",
            "period",
            "quantity_price",
        ]

        tables_rows = {}
        for tbl_name in tables_to_log:
            tbl = metadata.tables.get(tbl_name)
            if tbl is None:
                logger.warning("Table not found: %s", tbl_name)
                tables_rows[tbl_name] = None
                continue
            try:
                # For airtime_balance, filter by phone_number
                if tbl_name == "airtime_balance" and hasattr(tbl.c, "phone_number"):
                    q = select(tbl).where(tbl.c.phone_number == request.phone)
                else:
                    q = select(tbl)
                res = await session.execute(q)
                rows = res.fetchall()
                rows_list = [dict(r._mapping) for r in rows]
                tables_rows[tbl_name] = rows_list
                logger.info("Table '%s' rows (%d): %s", tbl_name, len(rows_list), rows_list)
            except Exception as e:
                logger.exception("Failed to fetch rows for table %s: %s", tbl_name, e)
                tables_rows[tbl_name] = "error"

        # Fetch user row by phone
        if not hasattr(users.c, "phone_number"):
            raise HTTPException(status_code=400, detail="No phone_number column in users table")
        query = select(users).where(users.c.phone_number == request.phone)
        result = await session.execute(query)
        user = result.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        user_row = dict(user._mapping)

        # Build package list locally from retrieved tables
        period_rows = tables_rows.get("period") or []
        sub_rows = tables_rows.get("sub_category") or []
        main_rows = tables_rows.get("main_category") or []
        qty_rows = tables_rows.get("quantity_price") or []

        periods = {r.get("id"): r.get("label") for r in period_rows}
        subs = {r.get("id"): {"name": r.get("name"), "main_id": r.get("main_id")} for r in sub_rows}
        mains = {r.get("id"): r.get("name") for r in main_rows}

        packages = []
        period_to_sub = {r.get("id"): r.get("sub_id") for r in period_rows}
        for q in qty_rows:
            period_id = q.get("period_id")
            period_label = periods.get(period_id, "unknown")
            sub_id = period_to_sub.get(period_id)
            sub_name = None
            main_name = None
            if sub_id:
                sub = subs.get(sub_id)
                if sub:
                    sub_name = sub.get("name")
                    main_name = mains.get(sub.get("main_id"))
            pkg = {
                "quantity": str(q.get("quantity")),
                "price": str(q.get("price")),
                "period": period_label,
                "sub_category": sub_name,
                "main_category": main_name,
            }
            packages.append(pkg)

        # Filter packages for main category 'internet' if available
        internet_packages = [p for p in packages if p.get("main_category") and p.get("main_category").lower() == "internet"]
        chosen = internet_packages if internet_packages else packages

        if not chosen:
            return {"reply": "No packages available at this time."}

        # Initialize reply_text
        reply_text = None

        # Use OpenAI to understand the user's message and generate the final reply
        if OPENAI_API_KEY:
            model_name = "gpt-4o-mini"  # Fixed: was "gpt-5-mini"
            
            system_msg = """You are a professional telecom assistant helping customers find the best internet packages.

TASK: Analyze the user's message and available packages, then provide a clear, concise response.

ANALYSIS STEPS:
1. Parse the user's request for:
   - Desired period (daily/day, weekly/week, monthly/month)
   - Category preference (internet, data, etc.)
   - Number of options requested (e.g., "3 packages", "top 5")
   - Quantity preference (high quantity, low cost, etc.)

2. Filter packages based on user criteria:
   - If period mentioned → filter to that period only
   - If "internet" mentioned → prioritize internet packages
   - If number requested → return that many (or fewer if unavailable)

3. Sort packages intelligently:
   - For "high quantity" requests: sort by quantity DESC, then price ASC
   - For "cheap/affordable" requests: sort by price ASC, then quantity DESC
   - Convert quantity to float for proper numeric sorting

4. Format response professionally:
   - List each package as: "• [quantity]MB for [price] RWF - [sub_category] ([period])"
   - Add one brief sentence explaining the recommendation
   - If no matches found, ask ONE clarifying question

IMPORTANT RULES:
- Return ONLY the response text (no JSON, no markdown formatting, no extra commentary)
- Keep responses concise (3-5 lines maximum)
- Use clear, customer-friendly language
- Handle edge cases gracefully (no matches, ambiguous requests)

Example good responses:
"Here are the top 3 monthly internet packages with high data:
• 500MB for 400 RWF - izindi_pack (month)
• 200MB for 180 RWF - izindi_pack (month)
• 100MB for 100 RWF - izindi_pack (month)
These offer the best data allowance for monthly plans."

"I found these affordable weekly internet options:
• 100MB for 100 RWF - gwamon (week)
• 200MB for 180 RWF - gwamon (week)
These are our most cost-effective weekly packages."
"""

            model_payload = {
                "user_message": request.message,
                "available_packages": chosen,
                "user_phone": request.phone,
            }

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    req_json = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": json.dumps(model_payload, indent=2)},
                        ],
                        "temperature": 0.3,  # Slightly higher for more natural responses
                        "max_tokens": 512,
                    }

                    logger.info("Sending request to OpenAI with model: %s", model_name)
                    
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json=req_json,
                    )
                    
                    logger.info("OpenAI response status: %s", resp.status_code)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices") or []
                        if choices:
                            content = choices[0].get("message", {}).get("content", "").strip()
                            if content:
                                reply_text = content
                                logger.info("OpenAI generated reply successfully")
                            else:
                                logger.warning("OpenAI returned empty content")
                    else:
                        logger.error("OpenAI request failed: %s %s", resp.status_code, resp.text)
                        
            except Exception as e:
                logger.exception("OpenAI request error: %s", e)

        # Fallback: if reply_text not set by OpenAI, build a simple local reply
        if not reply_text:
            logger.warning("Using fallback response generation")
            # Try to apply some basic filtering
            filtered = chosen
            msg_lower = request.message.lower()
            
            # Filter by period if mentioned
            if "month" in msg_lower:
                filtered = [p for p in filtered if p.get("period", "").lower() == "month"]
            elif "week" in msg_lower:
                filtered = [p for p in filtered if p.get("period", "").lower() == "week"]
            elif "day" in msg_lower or "daily" in msg_lower:
                filtered = [p for p in filtered if p.get("period", "").lower() == "day"]
            
            if not filtered:
                filtered = chosen
            
            # Sort by quantity descending
            try:
                filtered.sort(key=lambda x: float(x.get("quantity", 0)), reverse=True)
            except:
                pass
            
            # Limit to requested number
            import re
            num_match = re.search(r'\b(\d+)\b', request.message)
            if num_match:
                limit = int(num_match.group(1))
                filtered = filtered[:limit]
            else:
                filtered = filtered[:5]  # Default to top 5
            
            lines = ["Here are the available packages:"]
            for p in filtered:
                sub = f" - {p.get('sub_category')}" if p.get('sub_category') else ""
                lines.append(f"• {p.get('quantity')}MB for {p.get('price')} RWF{sub} ({p.get('period')})")
            reply_text = "\n".join(lines)

        return {"reply": reply_text}