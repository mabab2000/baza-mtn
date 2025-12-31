from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, MetaData
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import logging
import json
import httpx
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

engine = create_async_engine(DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'), echo=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

app = FastAPI()

# In-memory session storage (use Redis in production)
chat_sessions: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    phone: str
    message: str

def get_session(phone: str) -> Dict[str, Any]:
    """Get or create chat session for user"""
    if phone not in chat_sessions:
        chat_sessions[phone] = {
            "state": "idle",
            "intent": None,
            "collected_data": {},
            "conversation_history": [],
            "last_packages": [],
            "timestamp": datetime.now().isoformat()
        }
    chat_sessions[phone]["timestamp"] = datetime.now().isoformat()
    return chat_sessions[phone]

async def get_user_balance(session: AsyncSession, phone: str, metadata: MetaData) -> Optional[float]:
    """Fetch user's airtime balance"""
    airtime_table = metadata.tables.get("airtime_balance")
    if airtime_table is None:
        return None
    
    try:
        query = select(airtime_table).where(airtime_table.c.phone_number == phone)
        result = await session.execute(query)
        balance_row = result.fetchone()
        if balance_row:
            return float(balance_row._mapping.get("balance", 0))
    except Exception as e:
        logger.exception("Error fetching balance: %s", e)
    return None

async def process_airtime_purchase(session: AsyncSession, phone: str, package_data: Dict[str, Any], metadata: MetaData) -> Dict[str, Any]:
    """Process airtime payment and activate bundle"""
    try:
        price = float(package_data.get("price", 0))
        balance = await get_user_balance(session, phone, metadata)
        
        if balance is None or balance < price:
            return {
                "success": False,
                "message": f"Insufficient airtime balance. You have {balance if balance else 0} RWF but need {price} RWF."
            }
        
        # Deduct airtime balance
        airtime_table = metadata.tables.get("airtime_balance")
        if airtime_table:
            from sqlalchemy import update
            stmt = update(airtime_table).where(
                airtime_table.c.phone_number == phone
            ).values(balance=balance - price)
            await session.execute(stmt)
            await session.commit()
        
        # TODO: Call your bundle activation API here
        # Example: await activate_bundle(phone, package_data)
        
        logger.info(f"Airtime purchase successful: {phone} bought {package_data['quantity']}MB for {price} RWF")
        
        return {
            "success": True,
            "message": f"✅ Purchase successful!\n\n📦 Package: {package_data['quantity']}MB ({package_data['period']})\n💰 Amount: {price} RWF\n💳 Payment: Airtime Balance\n🆔 Transaction: TXN{datetime.now().strftime('%Y%m%d%H%M%S')}\n\nYour bundle has been activated!"
        }
        
    except Exception as e:
        logger.exception("Airtime purchase error: %s", e)
        return {
            "success": False,
            "message": f"Purchase failed: {str(e)}"
        }

async def process_momo_purchase(phone: str, package_data: Dict[str, Any], momo_pin: str) -> Dict[str, Any]:
    """Process Mobile Money payment"""
    try:
        price = float(package_data.get("price", 0))
        
        # TODO: Call your Mobile Money API here
        # Example: response = await momo_api.charge(phone, price, momo_pin)
        
        # Simulate API call
        if len(momo_pin) < 4:
            return {
                "success": False,
                "message": "Invalid PIN format. Please provide a valid Mobile Money PIN."
            }
        
        logger.info(f"MoMo purchase successful: {phone} bought {package_data['quantity']}MB for {price} RWF")
        
        return {
            "success": True,
            "message": f"✅ Purchase successful!\n\n📦 Package: {package_data['quantity']}MB ({package_data['period']})\n💰 Amount: {price} RWF\n💳 Payment: Mobile Money\n🆔 Transaction: TXN{datetime.now().strftime('%Y%m%d%H%M%S')}\n\nYour bundle has been activated!"
        }
        
    except Exception as e:
        logger.exception("MoMo purchase error: %s", e)
        return {
            "success": False,
            "message": f"Purchase failed: {str(e)}"
        }

def extract_package_from_message(message: str, packages: list) -> Optional[Dict[str, Any]]:
    """Extract package selection from user message"""
    msg_lower = message.lower()
    
    # Try to match by quantity and price
    for pkg in packages:
        quantity = pkg.get("quantity", "")
        price = pkg.get("price", "")
        period = pkg.get("period", "")
        
        if quantity and price:
            # Match patterns like "500MB", "500 MB", "500mb for 400", etc.
            if f"{quantity}mb" in msg_lower.replace(" ", "") or f"{quantity} mb" in msg_lower:
                if price in message or period in msg_lower:
                    return pkg
    
    # Try to match by number selection (e.g., "number 1", "option 2", "the first one")
    number_match = re.search(r'(?:number|option|choice)?\s*(\d+)', msg_lower)
    if number_match:
        try:
            index = int(number_match.group(1)) - 1
            if 0 <= index < len(packages):
                return packages[index]
        except (ValueError, IndexError):
            pass
    
    return None

def extract_payment_method(message: str) -> Optional[str]:
    """Extract payment method from user message"""
    msg_lower = message.lower()
    
    if any(word in msg_lower for word in ["airtime", "balance", "credit"]):
        return "airtime"
    elif any(word in msg_lower for word in ["momo", "mobile money", "mtn", "airtel"]):
        return "momo"
    elif re.search(r'\b[12]\b', message):  # User typed "1" or "2"
        if "1" in message:
            return "airtime"
        elif "2" in message:
            return "momo"
    
    return None

def extract_pin(message: str) -> Optional[str]:
    """Extract PIN from user message"""
    # Look for 4-6 digit numbers
    pin_match = re.search(r'\b(\d{4,6})\b', message)
    if pin_match:
        return pin_match.group(1)
    return None

def is_confirmation(message: str) -> bool:
    """Check if message is a confirmation"""
    msg_lower = message.lower()
    confirmations = ["yes", "confirm", "ok", "okay", "proceed", "continue", "buy", "purchase", "sure"]
    return any(word in msg_lower for word in confirmations)

def is_cancellation(message: str) -> bool:
    """Check if message is a cancellation"""
    msg_lower = message.lower()
    cancellations = ["no", "cancel", "stop", "abort", "nevermind", "never mind"]
    return any(word in msg_lower for word in cancellations)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async with async_session() as session:
        metadata = MetaData()
        async with engine.begin() as conn:
            await conn.run_sync(metadata.reflect)
        
        users = metadata.tables.get("users")
        if users is None:
            raise HTTPException(status_code=500, detail="Users table not found")

        # Verify user exists
        if not hasattr(users.c, "phone_number"):
            raise HTTPException(status_code=400, detail="No phone_number column in users table")
        
        query = select(users).where(users.c.phone_number == request.phone)
        result = await session.execute(query)
        user = result.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get user's airtime balance
        user_balance = await get_user_balance(session, request.phone, metadata)
        
        # Fetch all relevant tables
        tables_to_log = ["main_category", "sub_category", "period", "quantity_price"]
        tables_rows = {}
        
        for tbl_name in tables_to_log:
            tbl = metadata.tables.get(tbl_name)
            if tbl is None:
                logger.warning("Table not found: %s", tbl_name)
                tables_rows[tbl_name] = None
                continue
            try:
                q = select(tbl)
                res = await session.execute(q)
                rows = res.fetchall()
                rows_list = [dict(r._mapping) for r in rows]
                tables_rows[tbl_name] = rows_list
            except Exception as e:
                logger.exception("Failed to fetch rows for table %s: %s", tbl_name, e)
                tables_rows[tbl_name] = "error"

        # Build package list
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
                "id": str(q.get("id")),
                "quantity": str(q.get("quantity")),
                "price": str(q.get("price")),
                "period": period_label,
                "period_id": str(period_id),
                "sub_category": sub_name,
                "main_category": main_name,
            }
            packages.append(pkg)

        # Filter for internet packages
        internet_packages = [p for p in packages if p.get("main_category") and p.get("main_category").lower() == "internet"]
        available_packages = internet_packages if internet_packages else packages

        # Get user session
        user_session = get_session(request.phone)
        
        # Add to conversation history
        user_session["conversation_history"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })

        reply_text = None
        msg_lower = request.message.lower()

        # STATE MACHINE LOGIC
        
        # Check for cancellation at any point
        if is_cancellation(request.message):
            user_session["state"] = "idle"
            user_session["intent"] = None
            user_session["collected_data"] = {}
            reply_text = "No problem! Your session has been cancelled. How else can I help you today?"
        
        # STATE: CONFIRMING - Waiting for final confirmation
        elif user_session["state"] == "confirming":
            if is_confirmation(request.message):
                collected = user_session["collected_data"]
                payment_method = collected.get("payment_method")
                package_data = collected.get("package")
                
                # Process the purchase
                if payment_method == "airtime":
                    result = await process_airtime_purchase(session, request.phone, package_data, metadata)
                elif payment_method == "momo":
                    momo_pin = collected.get("momo_pin")
                    result = await process_momo_purchase(request.phone, package_data, momo_pin)
                else:
                    result = {"success": False, "message": "Invalid payment method."}
                
                reply_text = result["message"]
                
                # Reset session after purchase attempt
                user_session["state"] = "idle"
                user_session["intent"] = None
                user_session["collected_data"] = {}
            else:
                reply_text = "Please type 'confirm' to complete your purchase or 'cancel' to abort."
        
        # STATE: COLLECTING MOMO PIN
        elif user_session["state"] == "collecting_momo_pin":
            pin = extract_pin(request.message)
            if pin:
                user_session["collected_data"]["momo_pin"] = pin
                user_session["state"] = "confirming"
                
                pkg = user_session["collected_data"]["package"]
                reply_text = f"📋 Purchase Summary:\n\n"
                reply_text += f"📦 Package: {pkg['quantity']}MB ({pkg['period']})\n"
                reply_text += f"💰 Price: {pkg['price']} RWF\n"
                reply_text += f"💳 Payment: Mobile Money\n\n"
                reply_text += f"Type 'confirm' to complete your purchase or 'cancel' to abort."
            else:
                reply_text = "Please provide your Mobile Money PIN (4-6 digits) to proceed with the purchase."
        
        # STATE: COLLECTING PAYMENT METHOD
        elif user_session["state"] == "collecting_payment":
            payment_method = extract_payment_method(request.message)
            
            if payment_method:
                user_session["collected_data"]["payment_method"] = payment_method
                pkg = user_session["collected_data"]["package"]
                
                if payment_method == "airtime":
                    # Check balance
                    price = float(pkg["price"])
                    if user_balance and user_balance >= price:
                        user_session["state"] = "confirming"
                        reply_text = f"📋 Purchase Summary:\n\n"
                        reply_text += f"📦 Package: {pkg['quantity']}MB ({pkg['period']})\n"
                        reply_text += f"💰 Price: {pkg['price']} RWF\n"
                        reply_text += f"💳 Payment: Airtime Balance\n"
                        reply_text += f"💵 Current Balance: {user_balance} RWF\n"
                        reply_text += f"💵 Balance After: {user_balance - price} RWF\n\n"
                        reply_text += f"Type 'confirm' to complete your purchase or 'cancel' to abort."
                    else:
                        user_session["state"] = "idle"
                        user_session["collected_data"] = {}
                        reply_text = f"❌ Insufficient balance. You have {user_balance if user_balance else 0} RWF but need {pkg['price']} RWF.\n\n"
                        reply_text += f"Would you like to:\n1. Use Mobile Money instead\n2. Browse different packages"
                
                elif payment_method == "momo":
                    user_session["state"] = "collecting_momo_pin"
                    reply_text = f"Please provide your Mobile Money PIN to complete the purchase of {pkg['quantity']}MB for {pkg['price']} RWF."
            else:
                reply_text = "Please choose a payment method:\n1. Airtime Balance"
                if user_balance:
                    reply_text += f" ({user_balance} RWF available)"
                reply_text += "\n2. Mobile Money"
        
        # STATE: COLLECTING PACKAGE SELECTION
        elif user_session["state"] == "collecting_package":
            selected_package = extract_package_from_message(request.message, user_session["last_packages"])
            
            if selected_package:
                user_session["collected_data"]["package"] = selected_package
                user_session["state"] = "collecting_payment"
                
                reply_text = f"Great choice! You've selected:\n"
                reply_text += f"📦 {selected_package['quantity']}MB ({selected_package['period']}) - {selected_package['price']} RWF\n\n"
                reply_text += f"How would you like to pay?\n"
                reply_text += f"1. Airtime Balance"
                if user_balance:
                    reply_text += f" ({user_balance} RWF available)"
                reply_text += f"\n2. Mobile Money"
            else:
                reply_text = "I couldn't identify which package you want. Please specify:\n"
                reply_text += "- The package number (e.g., '1' or 'number 2')\n"
                reply_text += "- Or the quantity (e.g., '500MB' or '1GB')"
        
        # DEFAULT: Use OpenAI for intent detection and browsing
        else:
            # Detect purchase intent
            purchase_keywords = ["buy", "purchase", "get", "want", "need", "subscribe"]
            has_purchase_intent = any(keyword in msg_lower for keyword in purchase_keywords)
            
            if has_purchase_intent and user_session["intent"] != "buy_internet":
                user_session["intent"] = "buy_internet"
            
            # Use OpenAI for response generation
            if OPENAI_API_KEY:
                model_name = "gpt-4o-mini"
                
                system_msg = """You are a professional telecom assistant.

TASK: Analyze user's message and provide appropriate response.

USER BALANCE: {balance} RWF
USER INTENT: {intent}

RESPONSE GUIDELINES:

1. FOR INFORMATION/BROWSING REQUESTS:
   - Show relevant packages based on their query
   - Format: "• [quantity]MB for [price] RWF - [sub_category] ([period])"
   - Keep it concise (show top 5-7 packages)
   - Add one helpful sentence about the packages

2. FOR PURCHASE REQUESTS:
   - Show relevant packages with numbers
   - Format: "1. [quantity]MB for [price] RWF ([period])"
   - End with: "Which package would you like? (Reply with the number or quantity)"
   - Be encouraging and helpful

3. FILTERING:
   - If they mention period (daily/weekly/monthly), filter to that
   - If they mention quantity preference, sort accordingly
   - Default to showing diverse options

4. TONE:
   - Friendly and professional
   - Clear and concise
   - No emojis unless customer uses them
   - No markdown formatting

Return ONLY the response text."""

                context = {
                    "user_message": request.message,
                    "available_packages": available_packages[:15],
                    "user_balance": user_balance,
                    "user_intent": user_session.get("intent")
                }

                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        req_json = {
                            "model": model_name,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": system_msg.format(
                                        balance=user_balance if user_balance else "Unknown",
                                        intent=user_session.get("intent") or "browsing"
                                    )
                                },
                                {"role": "user", "content": json.dumps(context, indent=2)},
                            ],
                            "temperature": 0.3,
                            "max_tokens": 600,
                        }

                        resp = await client.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {OPENAI_API_KEY}",
                                "Content-Type": "application/json",
                            },
                            json=req_json,
                        )
                        
                        if resp.status_code == 200:
                            data = resp.json()
                            choices = data.get("choices") or []
                            if choices:
                                content = choices[0].get("message", {}).get("content", "").strip()
                                if content:
                                    reply_text = content
                                    
                                    # If purchase intent, prepare for package selection
                                    if user_session["intent"] == "buy_internet":
                                        user_session["state"] = "collecting_package"
                                        
                                        # Filter packages shown to user
                                        filtered = available_packages
                                        if "month" in msg_lower:
                                            filtered = [p for p in filtered if p.get("period", "").lower() == "month"]
                                        elif "week" in msg_lower:
                                            filtered = [p for p in filtered if p.get("period", "").lower() == "week"]
                                        elif "day" in msg_lower or "daily" in msg_lower:
                                            filtered = [p for p in filtered if p.get("period", "").lower() == "day"]
                                        
                                        if not filtered:
                                            filtered = available_packages
                                        
                                        # Sort by quantity
                                        try:
                                            filtered.sort(key=lambda x: float(x.get("quantity", 0)), reverse=True)
                                        except:
                                            pass
                                        
                                        user_session["last_packages"] = filtered[:10]
                                        
                except Exception as e:
                    logger.exception("OpenAI error: %s", e)

        # Fallback
        if not reply_text:
            reply_text = "I'm here to help you find and purchase internet packages. What are you looking for today?"

        # Store response in history
        user_session["conversation_history"].append({
            "role": "assistant",
            "content": reply_text,
            "timestamp": datetime.now().isoformat()
        })

        return {"reply": reply_text}