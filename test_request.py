"""Send one example booking to the running service."""
import requests

booking = {
    "lead_time": 200, "arrival_date_week_number": 27, "arrival_date_day_of_month": 5,
    "stays_in_weekend_nights": 1, "stays_in_week_nights": 3, "adults": 2, "children": 0, "babies": 0,
    "is_repeated_guest": 0, "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
    "days_in_waiting_list": 0, "adr": 100.0, "total_of_special_requests": 0, "has_agent": 1,
    "hotel": "City Hotel", "arrival_date_month": "July", "meal": "BB", "country": "PRT",
    "market_segment": "Online TA", "reserved_room_type": "A", "deposit_type": "No Deposit",
    "customer_type": "Transient",
}

print(requests.post("http://localhost:9696/predict", json=booking).json())
