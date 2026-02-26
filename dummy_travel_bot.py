def travel_bot_response(question: str) -> str:
    q = question.lower()

    if "book" in q and "flight" in q:
        return "Search flights, choose flight, enter details and pay."

    if "cancel" in q:
        return "Cancel from My Trips. Refund depends on airline policy."

    if "visa" in q:
        return "We provide visa assistance after document submission."

    if "baggage" in q:
        return "15kg check-in and 7kg cabin baggage allowed."

    if "payment failed" in q:
        return "Refund processed within 5-7 working days."

    if "reschedule" in q or "change date" in q:
        return "Reschedule via My Trips with fare difference."

    if "insurance" in q:
        return "Travel insurance available at checkout."

    if "support" in q or "contact" in q:
        return "Call 1800-000-111 or use chat support."

    return "Please check My Trips or contact support."