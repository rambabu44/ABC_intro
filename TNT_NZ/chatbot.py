"""
New Zealand Travel Assistant - A Langchain RAG-based Chatbot
------------------------------------------------------------
This implementation creates a chatbot for New Zealand Tours & Travel company
with intent classification and RAG capabilities.
"""

import os
import json
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Install required packages first
# !pip install langchain langchain-community langchain-groq groq chromadb sentence-transformers

# Core components
from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Embeddings and LLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Guardrails
from langchain.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Groq LLM
api_key = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API"),
    model_name="llama3-70b-8192",  # Using Llama 3 70B as it's available on Groq
    temperature=0.2,
    max_tokens=800
)

# Define paths
PERSIST_DIRECTORY = "./chroma_db"
DATASET_DIRECTORY = "./data"

# Ensure directories exist
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(DATASET_DIRECTORY, exist_ok=True)

# Define all intents
INTENTS = [
    "check_baggage_allowance", "get_boarding_pass", "print_boarding_pass",
    "check_cancellation_fee", "check_in", "human_agent", "book_flight",
    "cancel_flight", "change_flight", "check_flight_insurance_coverage",
    "check_flight_offers", "check_flight_prices", "check_flight_reservation",
    "check_flight_status", "purchase_flight_insurance", "search_flight",
    "search_flight_insurance", "check_trip_prices", "get_refund", "change_seat",
    "choose_seat", "check_arrival_time", "check_departure_time", "book_trip",
    "cancel_trip", "change_trip", "check_trip_details", "check_trip_insurance_coverage",
    "check_trip_offers", "check_trip_plan", "purchase_trip_insurance",
    "search_trip", "search_trip_insurance"
]

# --------------------------------
# 1. CREATE SAMPLE DATASETS
# --------------------------------

# Define New Zealand-specific locations and attractions
NZ_LOCATIONS = {
    'cities': ['Auckland', 'Wellington', 'Christchurch', 'Queenstown', 'Rotorua', 'Dunedin', 'Hamilton', 'Tauranga', 'Nelson', 'Napier'],
    'attractions': ['Milford Sound', 'Waitomo Glowworm Caves', 'Hobbiton', 'Tongariro Alpine Crossing', 'Abel Tasman National Park', 
                   'Franz Josef Glacier', 'Bay of Islands', 'Lake Tekapo', 'Fiordland National Park', 'Waiheke Island'],
    'airports': ['Auckland Airport (AKL)', 'Wellington Airport (WLG)', 'Christchurch Airport (CHC)', 'Queenstown Airport (ZQN)', 
                'Dunedin Airport (DUD)', 'Hamilton Airport (HLZ)', 'Rotorua Airport (ROT)', 'Tauranga Airport (TRG)']
}

def generate_flight_data():
    """Generate sample flight data for New Zealand routes"""
    airlines = ["Air New Zealand", "Jetstar", "Qantas", "Singapore Airlines", "Emirates", "Cathay Pacific"]
    flight_data = []
    
    # Generate domestic flights
    for origin in NZ_LOCATIONS['airports']:
        for destination in NZ_LOCATIONS['airports']:
            if origin != destination:
                origin_code = re.search(r'\(([^)]+)', origin).group(1)
                dest_code = re.search(r'\(([^)]+)', destination).group(1)
                
                # Create multiple flights for each route
                for airline in airlines[:2]:  # Use first 2 airlines for domestic
                    flight_number = f"{airline[:3].upper()}{100 + hash(origin + destination) % 900}"
                    base_price = 100 + hash(origin + destination) % 300  # Between $100-$400
                    
                    # Add morning flight
                    flight_data.append({
                        "flight_id": str(uuid.uuid4())[:8],
                        "airline": airline,
                        "flight_number": flight_number + "M",
                        "origin": origin,
                        "destination": destination,
                        "departure_time": "07:30",
                        "arrival_time": "09:15",
                        "duration": "1h 45m",
                        "aircraft": "Airbus A320",
                        "price": base_price,
                        "route_type": "domestic",
                        "baggage_allowance": "1 x 23kg checked, 7kg carry-on"
                    })
                    
                    # Add afternoon flight
                    flight_data.append({
                        "flight_id": str(uuid.uuid4())[:8],
                        "airline": airline,
                        "flight_number": flight_number + "A", 
                        "origin": origin,
                        "destination": destination,
                        "departure_time": "14:45",
                        "arrival_time": "16:30",
                        "duration": "1h 45m",
                        "aircraft": "Airbus A320",
                        "price": base_price - 20,  # Slightly cheaper
                        "route_type": "domestic",
                        "baggage_allowance": "1 x 23kg checked, 7kg carry-on"
                    })
    
    # Generate international flights to/from Auckland
    international_destinations = [
        "Sydney Airport (SYD)", "Melbourne Airport (MEL)", "Los Angeles Airport (LAX)", 
        "Singapore Changi Airport (SIN)", "Tokyo Narita Airport (NRT)", "London Heathrow (LHR)"
    ]
    
    for destination in international_destinations:
        origin = "Auckland Airport (AKL)"
        dest_code = re.search(r'\(([^)]+)', destination).group(1)
        
        for airline in airlines:
            flight_number = f"{airline[:3].upper()}{500 + hash(destination) % 500}"
            base_price = 800 + hash(destination) % 1200  # Between $800-$2000
            
            # Adjust duration based on destination
            if "SYD" in destination or "MEL" in destination:
                duration = "3h 30m"
                arrival_offset = timedelta(hours=3, minutes=30)
            elif "SIN" in destination or "NRT" in destination:
                duration = "10h 15m"
                arrival_offset = timedelta(hours=10, minutes=15)
            elif "LAX" in destination:
                duration = "12h 30m"
                arrival_offset = timedelta(hours=12, minutes=30)
            else:  # London
                duration = "24h 0m"
                arrival_offset = timedelta(hours=24)
            
            # Add international flight
            flight_data.append({
                "flight_id": str(uuid.uuid4())[:8],
                "airline": airline,
                "flight_number": flight_number,
                "origin": origin,
                "destination": destination,
                "departure_time": "23:45",
                "arrival_time": "varies by timezone",
                "duration": duration,
                "aircraft": "Boeing 787-9" if hash(airline) % 2 == 0 else "Airbus A350",
                "price": base_price,
                "route_type": "international",
                "baggage_allowance": "2 x 23kg checked, 7kg carry-on"
            })
            
            # Add return flight
            flight_data.append({
                "flight_id": str(uuid.uuid4())[:8],
                "airline": airline,
                "flight_number": flight_number + "R",
                "origin": destination,
                "destination": origin,
                "departure_time": "09:30",
                "arrival_time": "varies by timezone",
                "duration": duration,
                "aircraft": "Boeing 787-9" if hash(airline) % 2 == 0 else "Airbus A350",
                "price": base_price + 50,  # Slightly more expensive
                "route_type": "international",
                "baggage_allowance": "2 x 23kg checked, 7kg carry-on"
            })
    
    return flight_data

def generate_tour_packages():
    """Generate sample tour packages for New Zealand"""
    tour_types = ["Adventure", "Relaxation", "Cultural", "Nature", "Wine & Food", "Lord of the Rings"]
    durations = [3, 5, 7, 10, 14]
    
    packages = []
    
    package_templates = [
        {
            "name": "North Island Explorer",
            "description": "Discover the volcanic landscapes, Maori culture and urban centers of New Zealand's North Island.",
            "locations": ["Auckland", "Rotorua", "Wellington", "Bay of Islands", "Hobbiton"],
            "type": "Cultural",
            "base_price": 1200,
        },
        {
            "name": "South Island Adventure",
            "description": "Experience the breathtaking mountains, fjords, and glaciers of New Zealand's South Island.",
            "locations": ["Christchurch", "Queenstown", "Milford Sound", "Franz Josef Glacier", "Lake Tekapo"],
            "type": "Adventure",
            "base_price": 1400,
        },
        {
            "name": "Lord of the Rings Journey",
            "description": "Visit the iconic filming locations from the Lord of the Rings trilogy across New Zealand.",
            "locations": ["Hobbiton", "Tongariro National Park", "Wellington", "Queenstown", "Fiordland"],
            "type": "Lord of the Rings",
            "base_price": 1600,
        },
        {
            "name": "Wine & Cuisine Tour",
            "description": "Indulge in New Zealand's finest wines and cuisine across renowned food regions.",
            "locations": ["Marlborough", "Hawke's Bay", "Wellington", "Waiheke Island", "Central Otago"],
            "type": "Wine & Food",
            "base_price": 1800,
        },
        {
            "name": "Extreme Sports Package",
            "description": "Experience the thrill of New Zealand's adventure capital with bungee jumping, skydiving, and more.",
            "locations": ["Queenstown", "Rotorua", "Taupo", "Abel Tasman", "Auckland"],
            "type": "Adventure",
            "base_price": 2000,
        },
        {
            "name": "Relaxation Retreat",
            "description": "Unwind with hot springs, spas, and peaceful landscapes across New Zealand's most serene locations.",
            "locations": ["Rotorua", "Hanmer Springs", "Coromandel", "Waiheke Island", "Lake Tekapo"],
            "type": "Relaxation",
            "base_price": 1700,
        },
        {
            "name": "Hiking & Nature Immersion",
            "description": "Trek through New Zealand's most stunning natural environments on guided hiking tours.",
            "locations": ["Tongariro", "Abel Tasman", "Milford Track", "Routeburn Track", "Fiordland"],
            "type": "Nature",
            "base_price": 1300,
        }
    ]
    
    # Generate packages with variations
    for template in package_templates:
        for duration in durations:
            # Price increases with duration
            price_per_day = template["base_price"] / 7  # Base price is normalized to 7 days
            price = int(duration * price_per_day)
            
            # Adjust locations based on duration
            if duration <= 5:
                locations = template["locations"][:3]
            else:
                locations = template["locations"]
                
            # Add some premium and budget variations
            variants = ["Standard"]
            if duration >= 7:
                variants.append("Premium")
            if duration <= 5:
                variants.append("Budget")
                
            for variant in variants:
                variant_price = price
                if variant == "Premium":
                    variant_price = int(price * 1.4)
                    accommodation = "4-5 star hotels"
                    meals = "All meals included with premium dining experiences"
                    transport = "Private transportation and domestic flights where applicable"
                elif variant == "Budget":
                    variant_price = int(price * 0.7)
                    accommodation = "Hostels and 2-3 star hotels"
                    meals = "Breakfast included, other meals self-catered"
                    transport = "Public transportation and group shuttles"
                else:  # Standard
                    accommodation = "3-4 star hotels"
                    meals = "Breakfast and dinner included"
                    transport = "Mix of private and public transportation"
                
                package_id = str(uuid.uuid4())[:8]
                package_name = f"{duration}-Day {template['name']}" if variant == "Standard" else f"{duration}-Day {variant} {template['name']}"
                
                packages.append({
                    "package_id": package_id,
                    "name": package_name,
                    "description": template["description"],
                    "type": template["type"],
                    "duration": f"{duration} days",
                    "locations": locations,
                    "accommodation": accommodation,
                    "meals": meals,
                    "transportation": transport,
                    "price": variant_price,
                    "highlights": [f"Explore {locations[0]}", 
                                  f"Experience {locations[1]}", 
                                  f"Discover {locations[2]}"],
                    "insurance_options": [
                        {"name": "Basic Coverage", "price": int(variant_price * 0.05), "coverage": "Trip cancellation, basic medical"},
                        {"name": "Comprehensive", "price": int(variant_price * 0.08), "coverage": "Trip cancellation, medical, belongings, activities"}
                    ]
                })
    
    return packages

def generate_bookings(num_bookings=50):
    """Generate sample customer bookings"""
    bookings = []
    statuses = ["Confirmed", "Pending", "Cancelled", "Completed"]
    payment_methods = ["Credit Card", "PayPal", "Bank Transfer"]
    
    # Get flight and package data
    flights = generate_flight_data()
    packages = generate_tour_packages()
    
    for i in range(num_bookings):
        booking_type = "flight" if i % 3 != 0 else "package"
        booking_id = f"BK{10000 + i}"
        customer_id = f"CUST{5000 + (i % 20)}"  # Create some repeat customers
        booking_date = (datetime.now() - timedelta(days=i % 90)).strftime("%Y-%m-%d")
        status = statuses[i % len(statuses)]
        
        if booking_type == "flight":
            flight = flights[i % len(flights)]
            # Create a random travel date in the future
            travel_date = (datetime.now() + timedelta(days=10 + (i % 180))).strftime("%Y-%m-%d")
            
            bookings.append({
                "booking_id": booking_id,
                "customer_id": customer_id,
                "booking_type": "flight",
                "booking_date": booking_date,
                "status": status,
                "payment_method": payment_methods[i % len(payment_methods)],
                "flight_details": {
                    "flight_id": flight["flight_id"],
                    "airline": flight["airline"],
                    "flight_number": flight["flight_number"],
                    "origin": flight["origin"],
                    "destination": flight["destination"],
                    "departure_date": travel_date,
                    "departure_time": flight["departure_time"],
                    "passengers": 1 + (i % 4),
                    "seat_selection": ["12A", "12B"] if i % 4 > 0 else ["15C"],
                    "total_price": flight["price"] * (1 + (i % 4))
                },
                "has_insurance": i % 5 == 0,
                "checked_in": status == "Confirmed" and i % 3 == 0
            })
        else:
            package = packages[i % len(packages)]
            # Create a random travel date in the future
            travel_date = (datetime.now() + timedelta(days=30 + (i % 180))).strftime("%Y-%m-%d")
            
            # Extract duration as integer
            duration_days = int(package["duration"].split()[0])
            end_date = (datetime.now() + timedelta(days=30 + (i % 180) + duration_days)).strftime("%Y-%m-%d")
            
            bookings.append({
                "booking_id": booking_id,
                "customer_id": customer_id,
                "booking_type": "package",
                "booking_date": booking_date,
                "status": status,
                "payment_method": payment_methods[i % len(payment_methods)],
                "package_details": {
                    "package_id": package["package_id"],
                    "package_name": package["name"],
                    "start_date": travel_date,
                    "end_date": end_date,
                    "travelers": 1 + (i % 4),
                    "total_price": package["price"] * (1 + (i % 4))
                },
                "has_insurance": i % 3 == 0,
                "insurance_tier": "Comprehensive" if i % 3 == 0 else "Basic Coverage" if i % 6 == 0 else None
            })
    
    return bookings

def generate_customers(num_customers=20):
    """Generate sample customer data"""
    first_names = ["James", "John", "Robert", "Michael", "William", "David", "Emma", "Olivia", "Ava", "Isabella", 
                  "Sophia", "Charlotte", "Amelia", "Mia", "Harper", "Liam", "Noah", "Oliver", "Elijah", "William"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson", "Taylor", "Clark",
                 "Hall", "Allen", "Young", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Carter"]
    countries = ["New Zealand", "Australia", "United States", "United Kingdom", "Canada", "Germany", "Japan", "Singapore", "France", "China"]
    
    customers = []
    
    for i in range(num_customers):
        customer_id = f"CUST{5000 + i}"
        first_name = first_names[i % len(first_names)]
        last_name = last_names[i % len(last_names)]
        
        customers.append({
            "customer_id": customer_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": f"{first_name.lower()}.{last_name.lower()}@example.com",
            "phone": f"+64 21 {555000 + i}",
            "nationality": countries[i % len(countries)],
            "passport_number": f"P{1000000 + i}" if countries[i % len(countries)] != "New Zealand" else "",
            "preferences": {
                "seat_preference": "Window" if i % 3 == 0 else "Aisle" if i % 3 == 1 else "No preference",
                "meal_preference": "Vegetarian" if i % 5 == 0 else "Vegan" if i % 7 == 0 else "Regular",
                "communication_preference": "Email" if i % 2 == 0 else "Phone"
            },
            "loyalty_tier": "Gold" if i < 3 else "Silver" if i < 8 else "Bronze" if i < 15 else "Standard",
            "loyalty_points": 10000 - (i * 500) if i < 20 else 0
        })
    
    return customers

def generate_baggage_policies():
    """Generate baggage policy information"""
    return {
        "domestic": {
            "Air New Zealand": {
                "carry_on": {
                    "weight": "7kg",
                    "dimensions": "118cm (46in) total linear dimensions",
                    "items": "1 bag + 1 small personal item"
                },
                "checked": {
                    "included": "1 x 23kg bag",
                    "extra": "$35 NZD per additional bag (up to 23kg)",
                    "overweight": "$60 NZD per bag (23-32kg)"
                }
            },
            "Jetstar": {
                "carry_on": {
                    "weight": "7kg",
                    "dimensions": "56cm x 36cm x 23cm",
                    "items": "1 bag only (no personal item unless upgraded)"
                },
                "checked": {
                    "included": "No free allowance on Starter fares",
                    "extra": "$45 NZD for first bag (up to 20kg), $65 NZD for additional",
                    "overweight": "$25 NZD per kg over allowance"
                }
            }
        },
        "international": {
            "Air New Zealand": {
                "carry_on": {
                    "weight": "7kg",
                    "dimensions": "118cm (46in) total linear dimensions",
                    "items": "1 bag + 1 small personal item"
                },
                "checked": {
                    "included": "Economy: 1 x 23kg, Premium Economy: 2 x 23kg, Business: 3 x 23kg",
                    "extra": "$70 NZD per additional bag (up to 23kg)",
                    "overweight": "$150 NZD per bag (23-32kg)"
                }
            },
            "Singapore Airlines": {
                "carry_on": {
                    "weight": "7kg",
                    "dimensions": "115cm (45in) total linear dimensions",
                    "items": "1 bag + 1 small personal item"
                },
                "checked": {
                    "included": "Economy: 30kg, Premium Economy: 35kg, Business: 40kg, First: 50kg",
                    "extra": "Charged by weight, from $100 NZD",
                    "overweight": "Included in weight-based system"
                }
            },
            "Emirates": {
                "carry_on": {
                    "weight": "7kg",
                    "dimensions": "55cm x 38cm x 20cm",
                    "items": "1 bag + 1 small personal item"
                },
                "checked": {
                    "included": "Economy: 30kg, Business: 40kg, First: 50kg",
                    "extra": "Charged by weight, from $120 NZD",
                    "overweight": "Included in weight-based system"
                }
            },
            "Qantas": {
                "carry_on": {
                    "weight": "7kg",
                    "dimensions": "115cm (45in) total linear dimensions",
                    "items": "1 bag + 1 small personal item"
                },
                "checked": {
                    "included": "Economy: 30kg, Premium Economy: 40kg, Business: 40kg, First: 50kg",
                    "extra": "$90 NZD per additional bag (up to 23kg)",
                    "overweight": "$100 NZD per bag (23-32kg)"
                }
            }
        }
    }

def generate_cancellation_policies():
    """Generate cancellation policy information"""
    return {
        "flights": {
            "Air New Zealand": {
                "flexible_fare": {
                    "up_to_24h": "Full refund minus $50 NZD service fee",
                    "less_than_24h": "Full refund minus $100 NZD service fee",
                    "no_show": "No refund"
                },
                "standard_fare": {
                    "up_to_72h": "75% refund",
                    "24h_to_72h": "50% refund",
                    "less_than_24h": "No refund",
                    "no_show": "No refund"
                },
                "saver_fare": {
                    "any_time": "No refund, credit valid for 12 months minus $100 NZD fee"
                }
            },
            "Jetstar": {
                "flex_bundle": {
                    "any_time": "Full refund as travel credit valid for 12 months"
                },
                "standard_fare": {
                    "any_time": "No refund"
                }
            },
            "Qantas": {
                "flexible_fare": {
                    "up_to_24h": "Full refund",
                    "less_than_24h": "Full refund minus $75 NZD service fee",
                    "no_show": "No refund"
                },
                "standard_fare": {
                    "up_to_72h": "70% refund",
                    "less_than_72h": "No refund",
                    "no_show": "No refund"
                }
            },
            "Singapore Airlines": {
                "flexi": {
                    "up_to_24h": "Full refund",
                    "less_than_24h": "Full refund minus $100 NZD service fee",
                    "no_show": "75% refund"
                },
                "standard": {
                    "up_to_7d": "75% refund",
                    "less_than_7d": "50% refund",
                    "less_than_24h": "25% refund",
                    "no_show": "No refund"
                },
                "lite": {
                    "any_time": "No refund"
                }
            }
        },
        "tour_packages": {
            "standard_packages": {
                "more_than_60d": "Full refund minus $200 NZD deposit",
                "30d_to_60d": "75% refund",
                "15d_to_30d": "50% refund",
                "7d_to_15d": "25% refund",
                "less_than_7d": "No refund"
            },
            "premium_packages": {
                "more_than_60d": "Full refund minus $300 NZD deposit",
                "30d_to_60d": "80% refund",
                "15d_to_30d": "60% refund",
                "7d_to_15d": "40% refund",
                "less_than_7d": "No refund"
            },
            "special_events": {
                "any_time": "No refund unless covered by travel insurance"
            },
            "with_insurance": {
                "covered_reason": "Full refund minus insurance premium",
                "non_covered_reason": "Subject to standard cancellation policies"
            }
        }
    }

def generate_insurance_policies():
    """Generate insurance policy details"""
    return {
        "basic_coverage": {
            "price_percentage": "5% of trip cost",
            "benefits": {
                "trip_cancellation": "Up to 100% of trip cost for covered reasons",
                "trip_interruption": "Up to 100% of trip cost for covered reasons",
                "emergency_medical": "Up to $50,000 NZD",
                "emergency_evacuation": "Up to $100,000 NZD",
                "baggage_loss": "Up to $1,000 NZD",
                "baggage_delay": "$100 NZD per day (maximum $300 NZD)",
                "travel_delay": "$200 NZD per day (maximum $600 NZD)"
            },
            "covered_reasons": [
                "Illness or injury of traveler or family member",
                "Death of traveler or family member",
                "Natural disaster at destination",
                "Terrorism at destination (within 30 days of arrival)",
                "Involuntary job termination"
            ],
            "exclusions": [
                "Pre-existing medical conditions",
                "Extreme sports and activities",
                "Self-inflicted injuries",
                "Alcohol or drug-related incidents",
                "War or civil unrest"
            ]
        },
        "comprehensive_coverage": {
            "price_percentage": "8% of trip cost",
            "benefits": {
                "trip_cancellation": "Up to 150% of trip cost for any reason",
                "trip_interruption": "Up to 150% of trip cost for any reason",
                "emergency_medical": "Up to $100,000 NZD",
                "emergency_evacuation": "Up to $250,000 NZD",
                "baggage_loss": "Up to $2,500 NZD",
                "baggage_delay": "$200 NZD per day (maximum $600 NZD)",
                "travel_delay": "$300 NZD per day (maximum $900 NZD)",
                "missed_connection": "Up to $500 NZD",
                "rental_car_damage": "Up to $35,000 NZD",
                "adventure_activities": "Covered"
            },
            "covered_reasons": [
                "All reasons covered under Basic plan",
                "Pre-existing medical conditions (with stability period)",
                "Work-related reasons",
                "School-related reasons",
                "Change of mind (Cancel For Any Reason - 75% reimbursement)",
                "Pregnancy complications",
                "Military obligations"
            ],
            "exclusions": [
                "Illegal activities",
                "Self-inflicted injuries",
                "Participating in professional sports",
                "Traveling against physician advice"
            ]
        },
        "adventure_coverage": {
            "price_percentage": "10% of trip cost",
            "benefits": {
                "trip_cancellation": "Up to 150% of trip cost for any reason",
                "trip_interruption": "Up to 150% of trip cost for any reason",
                "emergency_medical": "Up to $250,000 NZD",
                "emergency_evacuation": "Up to $500,000 NZD",
                "baggage_loss": "Up to $3,500 NZD",
                "baggage_delay": "$300 NZD per day (maximum $900 NZD)",
                "travel_delay": "$400 NZD per day (maximum $1,200 NZD)",
                "missed_connection": "Up to $1,000 NZD",
                "rental_car_damage": "Up to $50,000 NZD",
                "search_and_rescue": "Up to $25,000 NZD",
                "extreme_sports": "Full coverage for bungee jumping, skydiving, white water rafting, etc."
            },
            "covered_reasons": [
                "All reasons covered under Comprehensive plan",
                "Adventure sports and activities injuries",
                "Weather conditions affecting sporting events",
                "Equipment damage or loss"
            ],
            "exclusions": [
                "Illegal activities",
                "Self-inflicted injuries",
                "Intoxication-related incidents"
            ]
        }
    }

def generate_faq_data():
    """Generate FAQ data for common questions"""
    return {
        "check_in": {
            "online_check_in": "Online check-in opens 24 hours before your flight and closes 90 minutes before departure for domestic flights and 2 hours before for international flights.",
            "airport_check_in": "Airport check-in counters open 2 hours before domestic flights and 3 hours before international flights. Counters close 45 minutes before departure for domestic and 60 minutes for international.",
            "documents_required": "For domestic flights: Photo ID and booking reference. For international flights: Valid passport, visa (if required), and booking reference.",
            "baggage_drop": "Baggage drop closes 45 minutes before domestic flights and 60 minutes before international flights."
        },
        "boarding_pass": {
            "digital_pass": "Digital boarding passes are available through our mobile app or can be emailed to you after check-in.",
            "print_requirements": "If you prefer a printed boarding pass, you can print it at home after online check-in or at airport kiosks/check-in counters.",
            "lost_pass": "If you lose your boarding pass, please visit the check-in counter with your ID for a replacement.",
            "mobile_pass": "Mobile boarding passes can be added to Apple Wallet or Google Pay for convenient access."
        },
        "baggage": {
            "prohibited_items": "Prohibited items include flammable materials, explosives, weapons, some lithium batteries. For full list, visit our website safety section.",
            "special_items": "Special items like sports equipment, musical instruments, or medical devices may require pre-approval. Please contact us 48 hours before departure.",
            "delayed_baggage": "For delayed baggage, please file a report at the airport baggage service counter before leaving. We'll deliver your bag to your accommodation when found.",
            "damaged_baggage": "Report damaged baggage immediately at the airport. Claims must be filed within 24 hours for domestic and 7 days for international flights."
        },
        "booking_changes": {
            "name_change": "Name corrections up to 3 characters are free. Full name changes are subject to a fee of $50 NZD for domestic and $100 NZD for international bookings.",
            "date_change": "Date changes are subject to fare difference plus a change fee depending on your fare type and how far in advance the change is made.",
            "route_change": "Route changes are treated as a new booking. Cancellation terms apply to the original booking, and the new booking will be at current rates.",
            "add_passenger": "Adding passengers to an existing booking isn't possible. Please make a separate booking for additional travelers."
        },
        "refunds": {
            "processing_time": "Refunds typically take 7-10 business days to process and appear on your statement, depending on your payment method and bank.",
            "partial_refunds": "Partial refunds may apply when only portion of the journey is cancelled or based on our cancellation policy terms.",
            "refund_methods": "Refunds are processed to the original payment method. For expired cards, please contact our customer service.",
            "tax_refunds": "Airport taxes and fees are refundable even on non-refundable tickets if you do not travel."
        },
        "travel_requirements": {
            "covid19": "COVID-19 requirements vary by destination and change frequently. Please check the latest requirements on our website before travel.",
            "visa_info": "Visa requirements depend on your nationality and destination. New Zealand citizens typically need a visa for many countries outside of New Zealand, Australia, and visa waiver countries.",
            "passport_validity": "Your passport should be valid for at least 6 months beyond your planned return date for most international travel.",
            "travel_insurance": "Travel insurance is highly recommended for all international travel and optional for domestic. We offer insurance during the booking process."
        }
    }

# Create all datasets
def create_datasets():
    """Create and save all datasets for the RAG system"""
    datasets = {
        "flights": generate_flight_data(),
        "tour_packages": generate_tour_packages(),
        "bookings": generate_bookings(),
        "customers": generate_customers(),
        "baggage_policies": generate_baggage_policies(),
        "cancellation_policies": generate_cancellation_policies(),
        "insurance_policies": generate_insurance_policies(),
        "faqs": generate_faq_data()
    }
    
    # Ensure directory exists
    os.makedirs(DATASET_DIRECTORY, exist_ok=True)
    
    # Save each dataset to a JSON file
    for name, data in datasets.items():
        with open(f"{DATASET_DIRECTORY}/{name}.json", "w") as f:
            json.dump(data, f, indent=2)
    
    print(f"Created datasets in {DATASET_DIRECTORY}")
    return datasets

# --------------------------------
# 2. INTENT CLASSIFICATION
# --------------------------------

class IntentClassifier:
    """Intent classifier using LLM"""
    
    def __init__(self, llm):
        self.llm = llm
        self.intents = INTENTS
        self.prompt = self._create_intent_classification_prompt()
        
    def _create_intent_classification_prompt(self):
        """Create the prompt template for intent classification"""
        intent_descriptions = {
            "check_baggage_allowance": "Inquire about how much baggage is allowed on a flight",
            "get_boarding_pass": "Request to receive a boarding pass",
            "print_boarding_pass": "Request to print a physical boarding pass",
            "check_cancellation_fee": "Ask about fees for cancelling a booking",
            "check_in": "Request to check in for a flight",
            "human_agent": "Request to speak with a human customer service agent",
            "book_flight": "Request to book a new flight",
            "cancel_flight": "Request to cancel an existing flight booking",
            "change_flight": "Request to make changes to an existing flight booking",
            "check_flight_insurance_coverage": "Inquire about what flight insurance covers",
            "check_flight_offers": "Ask about current flight deals or special offers",
            "check_flight_prices": "Inquire about prices for specific flights",
            "check_flight_reservation": "Check details of an existing flight reservation",
            "check_flight_status": "Inquire if a flight is on time, delayed, or cancelled",
            "purchase_flight_insurance": "Request to buy insurance for a flight",
            "search_flight": "Search for available flights",
            "search_flight_insurance": "Look for information about flight insurance options",
            "check_trip_prices": "Ask about the cost of travel packages",
            "get_refund": "Request a refund for a cancelled booking",
            "change_seat": "Request to change seat assignment on a booked flight",
            "choose_seat": "Request to select a seat on a flight",
            "check_arrival_time": "Inquire about when a flight will arrive",
            "check_departure_time": "Inquire about when a flight will depart",
            "book_trip": "Request to book a complete travel package",
            "cancel_trip": "Request to cancel an existing trip or travel package",
            "change_trip": "Request to modify an existing trip booking",
            "check_trip_details": "Check details of an existing trip reservation",
            "check_trip_insurance_coverage": "Inquire about what travel package insurance covers",
            "check_trip_offers": "Ask about current travel package deals or special offers",
            "check_trip_plan": "Review itinerary or plan for a trip",
            "purchase_trip_insurance": "Request to buy insurance for a travel package",
            "search_trip": "Search for available travel packages",
            "search_trip_insurance": "Look for information about travel package insurance options"
        }
        
        # Create a list of intents with descriptions for the prompt
        intent_list = "\n".join([f"- {intent}: {intent_descriptions.get(intent, 'No description')}" 
                                for intent in self.intents])
        
        template = """You are an AI assistant for a New Zealand tour and travel company. Your task is to classify the user's message into one of the predefined intents.

Available intents:
{intent_list}

User message: {user_input}

First, think step by step about the user's request and what they're trying to accomplish.
Then, identify the SINGLE most appropriate intent from the list above that matches their request.

Output your answer as a single intent name from the list, without any additional text or explanation.
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["user_input"],
            partial_variables={"intent_list": intent_list}
        )
    
    def classify(self, user_input):
        """Classify user input into an intent"""
        # chain = LLMChain(llm=self.llm, prompt=self.prompt, output_parser=StrOutputParser())
        chain = self.prompt | self.llm | StrOutputParser()
        intent = chain.invoke({"user_input": user_input}).strip()
        
        # Validate that returned intent is in our list
        if intent not in self.intents:
            return "human_agent"  # Default to human agent if intent not recognized
        
        return intent

# --------------------------------
# 3. RAG SYSTEM WITH CHROMA
# --------------------------------

def prep_documents_for_chroma(datasets):
    """Prepare documents for loading into Chroma"""
    documents = []
    
    # Process flight data
    for flight in datasets["flights"]:
        content = f"""
        Flight Information:
        Flight ID: {flight['flight_id']}
        Airline: {flight['airline']}
        Flight Number: {flight['flight_number']}
        Origin: {flight['origin']}
        Destination: {flight['destination']}
        Departure Time: {flight['departure_time']}
        Arrival Time: {flight['arrival_time']}
        Duration: {flight['duration']}
        Aircraft: {flight['aircraft']}
        Price: ${flight['price']} NZD
        Route Type: {flight['route_type']}
        Baggage Allowance: {flight['baggage_allowance']}
        """
        metadata = {
            "type": "flight",
            "id": flight['flight_id'],
            "airline": flight['airline'],
            "origin": flight['origin'],
            "destination": flight['destination'],
            "route_type": flight['route_type']
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Process tour packages
    for package in datasets["tour_packages"]:
        content = f"""
        Tour Package Information:
        Package ID: {package['package_id']}
        Name: {package['name']}
        Description: {package['description']}
        Type: {package['type']}
        Duration: {package['duration']}
        Locations: {', '.join(package['locations'])}
        Accommodation: {package['accommodation']}
        Meals: {package['meals']}
        Transportation: {package['transportation']}
        Price: ${package['price']} NZD
        Highlights: {', '.join(package['highlights'])}
        Insurance Options: {', '.join([f"{ins['name']} (${ins['price']} NZD)" for ins in package['insurance_options']])}
        """
        metadata = {
            "type": "package",
            "id": package['package_id'],
            "name": package['name'],
            "package_type": package['type'],
            "duration": package['duration'],
            "price": str(package['price'])
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Process baggage policies
    domestic_content = "Domestic Baggage Policies:\n"
    for airline, policy in datasets["baggage_policies"]["domestic"].items():
        domestic_content += f"""
        {airline}:
        Carry-on: {policy['carry_on']['weight']} weight limit, {policy['carry_on']['dimensions']} dimensions, {policy['carry_on']['items']}
        Checked: {policy['checked']['included']} included, {policy['checked']['extra']} for extra bags, {policy['checked']['overweight']} for overweight
        """
    
    documents.append(Document(
        page_content=domestic_content,
        metadata={"type": "baggage_policy", "category": "domestic"}
    ))
    
    international_content = "International Baggage Policies:\n"
    for airline, policy in datasets["baggage_policies"]["international"].items():
        international_content += f"""
        {airline}:
        Carry-on: {policy['carry_on']['weight']} weight limit, {policy['carry_on']['dimensions']} dimensions, {policy['carry_on']['items']}
        Checked: {policy['checked']['included']} included, {policy['checked']['extra']} for extra bags, {policy['checked']['overweight']} for overweight
        """
    
    documents.append(Document(
        page_content=international_content,
        metadata={"type": "baggage_policy", "category": "international"}
    ))
    
    # Process cancellation policies
    flight_cancel_content = "Flight Cancellation Policies:\n"
    for airline, policies in datasets["cancellation_policies"]["flights"].items():
        flight_cancel_content += f"\n{airline}:\n"
        for fare_type, rules in policies.items():
            flight_cancel_content += f"  {fare_type.replace('_', ' ').title()}:\n"
            for timeframe, rule in rules.items():
                flight_cancel_content += f"    - {timeframe.replace('_', ' ')}: {rule}\n"
    
    documents.append(Document(
        page_content=flight_cancel_content,
        metadata={"type": "cancellation_policy", "category": "flights"}
    ))
    
    tour_cancel_content = "Tour Package Cancellation Policies:\n"
    for package_type, policies in datasets["cancellation_policies"]["tour_packages"].items():
        tour_cancel_content += f"\n{package_type.replace('_', ' ').title()}:\n"
        for timeframe, rule in policies.items():
            tour_cancel_content += f"  - {timeframe.replace('_', ' ')}: {rule}\n"
    
    documents.append(Document(
        page_content=tour_cancel_content,
        metadata={"type": "cancellation_policy", "category": "tour_packages"}
    ))
    
    # Process insurance policies
    for policy_type, details in datasets["insurance_policies"].items():
        content = f"""
        {policy_type.replace('_', ' ').title()} Insurance:
        Cost: {details['price_percentage']}
        
        Benefits:
        """
        
        for benefit, coverage in details['benefits'].items():
            content += f"- {benefit.replace('_', ' ').title()}: {coverage}\n"
        
        content += "\nCovered Reasons:\n"
        for reason in details['covered_reasons']:
            content += f"- {reason}\n"
        
        content += "\nExclusions:\n"
        for exclusion in details['exclusions']:
            content += f"- {exclusion}\n"
        
        documents.append(Document(
            page_content=content,
            metadata={"type": "insurance_policy", "policy_type": policy_type}
        ))
    
    # Process FAQs
    for category, faqs in datasets["faqs"].items():
        for question, answer in faqs.items():
            content = f"Q: {question.replace('_', ' ').title()}\nA: {answer}"
            documents.append(Document(
                page_content=content,
                metadata={"type": "faq", "category": category, "question": question}
            ))
    
    return documents

def load_into_chroma(documents):
    """Load documents into Chroma vector store"""
    # Initialize Chroma with documents
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Persist the database
    vectorstore.persist()
    print(f"Loaded {len(documents)} documents into Chroma at {PERSIST_DIRECTORY}")
    
    return vectorstore

def get_or_create_vectorstore(datasets=None):
    """Get existing Chroma DB or create a new one"""
    # Check if the Chroma directory exists and has content
    if os.path.exists(PERSIST_DIRECTORY) and len(os.listdir(PERSIST_DIRECTORY)) > 0:
        print(f"Loading existing Chroma DB from {PERSIST_DIRECTORY}")
        return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    
    # If no datasets provided, create them
    if not datasets:
        datasets = create_datasets()
    
    # Prepare documents and load into Chroma
    documents = prep_documents_for_chroma(datasets)
    return load_into_chroma(documents)

# --------------------------------
# 4. RESPONSE GENERATION
# --------------------------------

def build_prompt_for_intent(intent):
    """Build a prompt template based on the user intent"""
    
    # Common system message for all intents
    system_message = """You are an intelligent virtual assistant for New Zealand Tours & Travel. 
You provide helpful, accurate, and friendly responses about flights, accommodations, tours, and travel services in New Zealand.
Always be polite and professional. If you don't know something, say so rather than making up information.
"""

    # Intent-specific prompts
    intent_prompts = {
        "check_baggage_allowance": """
Based on the retrieved information about baggage policies, answer the user's question about baggage allowance.
Include details about weight limits, dimensions, and any fees for extra or overweight baggage if relevant.
Make sure to distinguish between domestic and international policies if applicable.
""",
        "get_boarding_pass": """
Explain to the user how they can get their boarding pass based on the retrieved information.
Include options like online check-in, mobile app, airport kiosks, or check-in counters.
Mention any timing requirements or document needs.
""",
        "print_boarding_pass": """
Provide instructions on how to print a boarding pass based on the retrieved information.
Include options such as online printing after check-in, printing at airport kiosks, or at check-in counters.
""",
        "check_cancellation_fee": """
Based on the retrieved cancellation policy information, explain the applicable cancellation fees.
Consider the type of booking (flight or tour package), airline or package type, and timing of cancellation if mentioned.
Be specific about refund amounts or percentages when possible.
""",
        "check_in": """
Provide information about the check-in process based on the retrieved information.
Include online and airport check-in options, timing requirements, and necessary documents.
If the user has a specific booking reference, provide specific check-in instructions if possible.
""",
        "human_agent": """
Acknowledge the user's request to speak with a human agent. 
Provide contact information for customer service: phone +64 9 123 4567, email support@nztours.co.nz.
Mention operating hours: Monday-Friday 8am-8pm, Saturday-Sunday 9am-5pm (New Zealand Time).
""",
        "book_flight": """
Based on the retrieved flight information, help the user book a flight.
Ask for any missing details like origin, destination, dates, passenger count, and preferences.
Present flight options with times and prices if available in the retrieved information.
Explain next steps in the booking process.
""",
        "cancel_flight": """
Guide the user through cancelling their flight based on the retrieved information.
Explain the cancellation policy and any fees that may apply.
Request the booking reference if not provided and outline the cancellation process.
""",
        "change_flight": """
Help the user change their flight booking based on the retrieved information.
Explain any change fees or fare differences that might apply.
Request the booking reference if not provided and outline the change process.
""",
        "check_flight_insurance_coverage": """
Based on the retrieved insurance policy information, explain what is covered under flight insurance.
Include details about cancellation coverage, medical emergencies, baggage loss, etc.
Clarify any exclusions or limitations in the coverage.
""",
        "check_flight_offers": """
Share current flight offers and deals based on the retrieved information.
Include details about special prices, promotions, or seasonal offers.
Specify routes, travel periods, and any conditions that apply to the offers.
""",
        "check_flight_prices": """
Provide flight pricing information based on the retrieved data.
Include prices for different airlines, routes, or dates if available.
Mention any factors that affect pricing like season, advance booking, or fare class.
""",
        "check_flight_reservation": """
Help the user check their flight reservation details.
Ask for their booking reference if not provided.
Provide information about the flight times, dates, passenger details, and status if available.
""",
        "check_flight_status": """
Provide information about the status of flights based on the retrieved data.
Include details about scheduled departure/arrival times and any known delays.
Request flight number or route details if not provided.
""",
        "purchase_flight_insurance": """
Guide the user through purchasing flight insurance based on the retrieved insurance options.
Explain the different coverage options, costs, and benefits.
Outline the process for adding insurance to their booking.
""",
        "search_flight": """
Help the user search for flights based on their criteria and the retrieved flight data.
Ask for any missing details like origin, destination, dates, and preferences.
Present matching flight options with times, airlines, and prices if available.
""",
        "search_flight_insurance": """
Provide information about available flight insurance options based on the retrieved data.
Compare different coverage levels, benefits, and prices.
Explain how the user can select and purchase appropriate insurance.
""",
        "check_trip_prices": """
Share information about tour package prices based on the retrieved data.
Include details about different package types, durations, and what's included in the price.
Mention factors that affect pricing like season, group size, or accommodation level.
""",
        "get_refund": """
Explain the refund process based on the retrieved information.
Clarify eligible refund amounts based on cancellation policies.
Outline the steps for requesting a refund and expected processing times.
""",
        "change_seat": """
Guide the user through changing their seat assignment based on the retrieved information.
Explain any fees or limitations that may apply.
Outline the process for selecting a new seat and making the change.
""",
        "choose_seat": """
Help the user select a seat based on the retrieved information.
Explain seat selection options, any associated fees, and how to make the selection.
Request booking details if needed to assist with the seat selection process.
""",
        "check_arrival_time": """
Provide information about flight arrival times based on the retrieved data.
Include scheduled arrival times and any known updates or delays.
Request flight number or route details if not provided.
""",
        "check_departure_time": """
Share information about flight departure times based on the retrieved data.
Include scheduled departure times and any known updates or delays.
Request flight number or route details if not provided.
""",
        "book_trip": """
Help the user book a tour package based on the retrieved information.
Ask for any missing details like destinations of interest, dates, number of travelers, and preferences.
Present suitable package options with details and prices if available.
Explain the booking process and next steps.
""",
        "cancel_trip": """
Guide the user through cancelling their tour package based on the retrieved information.
Explain the cancellation policy and any fees that may apply.
Request the booking reference if not provided and outline the cancellation process.
""",
        "change_trip": """
Help the user change their tour package booking based on the retrieved information.
Explain any change fees or price differences that might apply.
Request the booking reference if not provided and outline the change process.
""",
        "check_trip_details": """
Provide information about the user's tour package booking based on the retrieved data.
Ask for their booking reference if not provided.
Share details about itinerary, accommodations, included services, and important dates.
""",
        "check_trip_insurance_coverage": """
Based on the retrieved insurance policy information, explain what is covered under tour package insurance.
Include details about cancellation coverage, medical emergencies, activity coverage, etc.
Clarify any exclusions or limitations in the coverage.
""",
        "check_trip_offers": """
Share current tour package offers and deals based on the retrieved information.
Include details about special prices, promotions, or seasonal offers.
Specify package types, travel periods, and any conditions that apply to the offers.
""",
        "check_trip_plan": """
Help the user review their tour itinerary or plan based on the retrieved information.
Ask for their booking reference if not provided.
Provide day-by-day breakdown of activities, accommodations, and services if available.
""",
        "purchase_trip_insurance": """
Guide the user through purchasing tour package insurance based on the retrieved options.
Explain the different coverage options, costs, and benefits.
Outline the process for adding insurance to their booking.
""",
        "search_trip": """
Help the user search for tour packages based on their criteria and the retrieved data.
Ask for any missing details like destinations of interest, dates, duration, and preferences.
Present matching package options with details and prices if available.
""",
        "search_trip_insurance": """
Provide information about available tour package insurance options based on the retrieved data.
Compare different coverage levels, benefits, and prices.
Explain how the user can select and purchase appropriate insurance.
"""
    }
    
    # Get intent-specific prompt or use a default if not found
    intent_prompt = intent_prompts.get(intent, "Answer the user's question based on the retrieved information.")
    
    # Combine system message and intent-specific prompt
    template = f"""
{system_message}

{intent_prompt}

Context information from our knowledge base:
{{context}}

User query: {{question}}

Please provide a helpful, accurate, and friendly response that directly addresses the user's query.
"""
    
    return ChatPromptTemplate.from_template(template)

class IntentHandler:
    """Handler for processing different intents"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        
    def handle(self, intent, user_query):
        """Handle a specific intent with RAG retrieval"""
        # Create retriever with search type and k based on intent
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Get prompt template for this intent
        prompt = build_prompt_for_intent(intent)
        
        # Create the RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Run the chain
        response = rag_chain.invoke(user_query)
        return response

# --------------------------------
# 5. CHAT HISTORY AND MEMORY
# --------------------------------

class ChatHistory:
    """Simple chat history manager"""
    
    def __init__(self):
        self.history = []
        
    def add_message(self, role, content):
        """Add a message to the history"""
        self.history.append({"role": role, "content": content})
        
    def get_last_n_messages(self, n=5):
        """Get the last n messages from history"""
        return self.history[-n:] if len(self.history) >= n else self.history
    
    def clear_history(self):
        """Clear the chat history"""
        self.history = []
        
    def __str__(self):
        """Convert history to string format"""
        result = ""
        for msg in self.history:
            result += f"{msg['role'].capitalize()}: {msg['content']}\n"
        return result

# --------------------------------
# 6. GUARDRAILS
# --------------------------------

class GuardrailValidator:
    """Validate and sanitize user inputs"""
    
    @staticmethod
    def validate_user_input(user_input):
        """Validate and sanitize user input"""
        # Check for empty input
        if not user_input or user_input.strip() == "":
            return False, "Please enter a message."
        
        # Check for excessively long input
        if len(user_input) > 2000:
            return False, "Your message is too long. Please keep it under 2000 characters."
        
        # Check for potentially harmful inputs (basic example)
        harmful_patterns = [
            r'<script.*?>.*?</script>',
            r'DROP TABLE',
            r'DELETE FROM',
            r'rm -rf',
            r'system\(',
            r'eval\('
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, "Your message contains harmful content. Please try again."
        
        return True, user_input
    
    @staticmethod
    def filter_sensitive_info(text):
        """Filter out sensitive information from responses"""
        # Mask credit card numbers
        text = re.sub(r'\b(?:\d{4}[ -]?){3}\d{4}\b', '[CREDIT CARD REDACTED]', text)
        
        # Mask passport numbers (basic pattern)
        text = re.sub(r'\b[A-Z]{1,2}\d{6,9}\b', '[PASSPORT NUMBER REDACTED]', text)
        
        # Mask what appear to be passwords
        text = re.sub(r'password\s*[=:]\s*\S+', 'password = [REDACTED]', text, flags=re.IGNORECASE)
        
        return text

# --------------------------------
# 7. MAIN CHATBOT CLASS
# --------------------------------

class NZTravelChatbot:
    """Main chatbot class that integrates all components"""
    
    def __init__(self):
        # Initialize datasets
        self.datasets = create_datasets()
        
        # Initialize vector store
        self.vectorstore = get_or_create_vectorstore(self.datasets)
        
        # Initialize components
        self.intent_classifier = IntentClassifier(llm)
        self.intent_handler = IntentHandler(self.vectorstore, llm)
        self.chat_history = ChatHistory()
        self.guardrails = GuardrailValidator()
        
        print("NZ Travel Chatbot initialized and ready!")
        
    def process_message(self, user_message):
        """Process an incoming user message"""
        # Apply guardrails
        valid, processed_input = self.guardrails.validate_user_input(user_message)
        if not valid:
            return processed_input
        
        # Add to history
        self.chat_history.add_message("user", processed_input)
        
        try:
            # Classify intent
            intent = self.intent_classifier.classify(processed_input)
            print(f"Detected intent: {intent}")
            
            # Handle intent to generate response
            response = self.intent_handler.handle(intent, processed_input)
            
            # Apply guardrails to response
            response = self.guardrails.filter_sensitive_info(response)
            
            # Add to history
            self.chat_history.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error while processing your request: {str(e)}"
            self.chat_history.add_message("assistant", error_msg)
            return error_msg
    
    def get_chat_history(self, n=5):
        """Get the last n messages from chat history"""
        return self.chat_history.get_last_n_messages(n)
    
    def clear_history(self):
        """Clear the chat history"""
        self.chat_history.clear_history()
        return "Chat history has been cleared."
    
    def search_knowledge_base(self, query, filter_dict=None, k=3):
        """Search the vectorstore directly with a query"""
        if filter_dict is None:
            filter_dict = {}
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        return results
    
    