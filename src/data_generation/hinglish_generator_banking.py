#!/usr/bin/env python3
"""
src/data_generation/hinglish_generator_banking.py

Generates synthetic Hinglish banking PII samples and writes JSONL with fields:
- source_text
- masked_text
- privacy_mask (JSON list)
- uid
- language
- region
- script
- mBERT_tokens (JSON list)
- mBERT_token_classes (JSON list)
- entities (JSON dict)

Usage:
  python src/data_generation/hinglish_generator_banking.py --num 1000 --out data/synthetic/hinglish_banking_1k.jsonl

Requirements:
  pip install transformers pandas tqdm
"""

from typing import List, Dict, Tuple, Optional
import random
import json
import re
import uuid
from pathlib import Path
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer


class HinglishBankingGenerator:
    def __init__(self, seed: Optional[int] = 42, schema_path: Optional[str] = "data/docs/data_schema.json"):
        random.seed(seed)
        self.seed = seed

        # load mBERT tokenizer (fast) for offsets
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)

        # try to load schema (regexes & examples)
        self.schema = {}
        self.schema_regex = {}
        if schema_path and Path(schema_path).exists():
            try:
                with open(schema_path, "r", encoding="utf-8") as fh:
                    self.schema = json.load(fh)
                # compile regex patterns for validation
                for label, info in self.schema.items():
                    pat = info.get("regex", None)
                    if pat:
                        try:
                            self.schema_regex[label] = re.compile(pat)
                        except re.error:
                            # fallback to raw
                            self.schema_regex[label] = re.compile(pat, flags=re.UNICODE)
            except Exception:
                self.schema = {}
                self.schema_regex = {}
        else:
            # no schema found: we'll still generate using internal heuristics
            self.schema = {}
            self.schema_regex = {}

        # small curated lists (expand as needed)
        self.indian_given_names = [
            "Arun", "Deepak", "Krishna", "Priya", "Anita", "Rahul", "Neha", "Amit",
            "Sunita", "Raj", "Pooja", "Vikram", "Kavya", "Arjun", "Meera", "Rohit",
            "Shreya", "Kiran", "Nisha", "Sanjay", "Divya", "Manoj", "Ritu", "Suresh",
            "Lalita", "Naveen", "Geeta", "Ramesh", "Sita", "Ajay", "Rekha", "Vinod",
            "Shanti", "Rajesh", "Kamala", "Ashok", "Usha", "Prakash", "Sarita", "Mohan",

            "Sanjana", "Kunal", "Isha", "Harsh", "Anjali", "Vikas", "Neelam", "Tarun",
            "Madhuri", "Aakash", "Sneha", "Raghav", "Bhavna", "Dinesh", "Radha", "Yash",
            "Swati", "Karthik", "Anita", "Suresh", "Anurag", "Smita", "Gaurav", "Manisha",
            "Nitin", "Vaishali", "Lokesh", "Snehal", "Siddharth", "Pallavi", "Jatin", "Bhavya",
            "Kavitha", "Lokesh", "Divya", "Suraj", "Bharti", "Arvind", "Mala", "Rakesh",
            "Sonal", "Vandana", "Sandeep", "Rina"
        ]

        self.indian_surnames = [
            "Sharma", "Patel", "Kumar", "Singh", "Gupta", "Agarwal", "Shah", "Jain",
            "Mehta", "Verma", "Yadav", "Mishra", "Tiwari", "Pandey", "Chopra", "Malhotra",
            "Bansal", "Kapoor", "Arora", "Saxena", "Joshi", "Srivastava", "Chandra", "Bhatia",
            "Khanna", "Goel", "Agnihotri", "Bhardwaj", "Chauhan", "Desai", "Kulkarni", "Reddy",

            "Nair", "Iyer", "Menon", "Pillai", "Naidu", "Shetty", "Rao", "Khan",
            "Ansari", "Syed", "Farooqi", "Qureshi", "Ali", "Fernandes", "Pereira", 
            "Rodrigues", "Thomas", "George", "Mathew", "John", "Singh", "Kaur", 
            "Gill", "Sandhu", "Dhillon", "Brar", "Bajwa", "Bains", "Chahal", "Lal", 
            "Bhatt", "Dubey", "Khandelwal", "Tandon", "Jadhav", "Ghosh", "Chatterjee"
        ]

        self.indian_cities = [
            "Mumbai", "Delhi", "Bangalore", "Pune", "Chennai", "Hyderabad", "Kolkata",
            "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
            "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad", "Ludhiana",
            "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Kalyan", "Varanasi",

            "Coimbatore", "Madurai", "Trichy", "Kochi", "Thiruvananthapuram", "Mysore",
            "Mangalore", "Warangal", "Nellore", "Hubli", "Salem", "Vijayawada", "Surat",
            "Navi Mumbai", "Aurangabad", "Kolhapur", "Solapur", "Ujjain", "Bilaspur",
            "Raipur", "Jabalpur", "Gwalior", "Durgapur", "Dehradun", "Shimla", "Manali",
            "Srinagar", "Jammu", "Haridwar", "Roorkee", "Amritsar", "Panipat", "Ambala",
            "Guntur", "Latur", "Gulbarga", "Mandi", "Jodhpur", "Aligarh"
        ]


        # banking + generic Hinglish templates
        self.templates = [
            "Hi, mera naam {GIVENNAME} {SURNAME} hai. Account balance check karna hai. Phone number {TELEPHONENUM} hai.",
            "Mumbai branch mein ja kar withdrawal kiya. Email {EMAIL} pe statement bhej do.",
            "{GIVENNAME} ji, aapka account number {ACCOUNTNUM} aur IFSC {IFSC} hai.",
            "Mera Aadhaar number {AADHAAR} hai, KYC complete karna hai.",
            "Please send the statement to {EMAIL} and call me at {TELEPHONENUM}.",
            "Beneficiary ka IFSC {IFSC} and account {ACCOUNTNUM}. Transfer kar dena.",
            "PAN {PAN} upload karke KYC complete karo.",
            "Credit card {CREDITCARDNUM} se payment hua. Verify karlo.",
            "{GIVENNAME} {SURNAME} ka address {BUILDINGNUM} {STREET}, {CITY} - {ZIPCODE}.",
            "Mere driving license {DRIVERLICENSENUM} aur voter id {VOTERID} ready hai.",
            "Main {CITY} mein rehta hun, age {AGE} hai. Mera number {TELEPHONENUM} hai.",
            "Hello {GIVENNAME}, appointment on {DATE} at {TIME}.",
            "Transaction id {TRANSACTIONID} se payment hua. Check karna hai."
        ]


        self.templates = [
            "Hi, mera naam {GIVENNAME} {SURNAME} hai. Account balance check karna hai. Phone number {TELEPHONENUM} hai.",
            "Mumbai branch mein ja kar withdrawal kiya. Email {EMAIL} pe statement bhej do.",
            "{GIVENNAME} ji, aapka account number {ACCOUNTNUM} aur IFSC {IFSC} hai.",
            "Mera Aadhaar number {AADHAAR} hai, KYC complete karna hai.",
            "Please send the statement to {EMAIL} and call me at {TELEPHONENUM}.",
            "Beneficiary ka IFSC {IFSC} and account {ACCOUNTNUM}. Transfer kar dena.",
            "PAN {PAN} upload karke KYC complete karo.",
            "Credit card {CREDITCARDNUM} se payment hua. Verify karlo.",
            "{GIVENNAME} {SURNAME} ka address {BUILDINGNUM} {STREET}, {CITY} - {ZIPCODE}.",
            "Mere driving license {DRIVERLICENSENUM} aur voter id {VOTERID} ready hai.",
            "Main {CITY} mein rehta hun, age {AGE} hai. Mera number {TELEPHONENUM} hai.",
            "Hello {GIVENNAME}, appointment on {DATE} at {TIME}.",
            "Transaction id {TRANSACTIONID} se payment hua. Check karna hai.",

            # --- New Rich Templates (Indian formats + Hinglish) ---
            "Mujhe loan apply karna hai. Aadhaar {AADHAAR} aur PAN {PAN} ready hai.",
            "{FULLNAME} ji ka phone {TELEPHONENUM} aur email {EMAIL} registered hai.",
            "Mera account {ACCOUNTNUM} IFSC {IFSC} ke sath linked hai.",
            "Credit card statement {EMAIL} par bhejna. Card number {CREDITCARDNUM} hai.",
            "Driver license {DRIVERLICENSENUM} and passport {PASSPORTNUM} KYC ke liye upload karna hai.",
            "Nominee ke details: naam {FULLNAME}, age {AGE}, city {CITY}.",
            "Voter ID {VOTERID} aur Aadhaar {AADHAAR} dono verify kar do.",
            "DOB {DATE} hai aur gender Male likha hai.",
            "Transaction {TRANSACTIONID} ko verify karo, time {TIME} tha.",
            "Address: {BUILDINGNUM}, {STREET}, {CITY}, {ZIPCODE}.",
            "Mera account freeze ho gaya hai. Account number {ACCOUNTNUM} hai.",
            "Kripya balance enquiry karna. Phone {TELEPHONENUM} aur Aadhaar {AADHAAR} registered hai.",
            "Lost my card {CREDITCARDNUM}, block karna hai immediately.",
            "IFSC code {IFSC} galat likha gaya hai, correct karna hai.",
            "Passport number {PASSPORTNUM} diya gaya hai travel insurance ke liye.",
            "{FULLNAME} ne {DATE} ko transaction {TRANSACTIONID} complete kiya.",
            "Email {EMAIL} par OTP bheja gaya tha lekin receive nahi hua.",
            "Beneficiary details: Name {FULLNAME}, Account {ACCOUNTNUM}, IFSC {IFSC}.",
            "Customer {GIVENNAME} ka registered voter id {VOTERID} hai.",
            "KYC form mein DOB {DATE}, Aadhaar {AADHAAR}, PAN {PAN} likha hai.",
            "Mere ghar ka address {BUILDINGNUM} {STREET}, {CITY}, PIN {ZIPCODE}.",
            "Driver license {DRIVERLICENSENUM} pe naam {FULLNAME} hai.",
            "ATM se withdraw {DATE} at {TIME}, transaction id {TRANSACTIONID}.",
            "PAN {PAN} aur Aadhaar {AADHAAR} mismatch aa raha hai.",
            "Mujhe {CITY} branch mein jaana hai, appointment {DATE} ko hai.",
            "Mobile banking registered number {TELEPHONENUM} update karna hai.",
            "Insurance ke liye passport {PASSPORTNUM} aur DOB {DATE} dena hoga.",
            "Credit card {CREDITCARDNUM} ka bill {DATE} tak due hai.",
            "Voter ID {VOTERID} address {CITY} ke liye valid hai.",
            "Transaction at {TIME}, id {TRANSACTIONID}, account {ACCOUNTNUM}.",
            "Loan ke liye salary account {ACCOUNTNUM}, IFSC {IFSC}, PAN {PAN} diya hai.",
            "{FULLNAME} ne branch {CITY} se cheque deposit kiya.",
            "Please update address to {BUILDINGNUM} {STREET}, {CITY} {ZIPCODE}.",
            "Aadhaar {AADHAAR}, voter id {VOTERID}, driver license {DRIVERLICENSENUM} sab upload ho gaya.",
            "Customer ka naam {FULLNAME}, DOB {DATE}, gender Female hai.",
            "Payment with card {CREDITCARDNUM} on {DATE} failed.",
            "Branch visit {CITY} mein {DATE} at {TIME} scheduled hai.",
            "Passport {PASSPORTNUM} aur PAN {PAN} dono ka copy bhejna hai.",
            "New nominee: {FULLNAME}, Age {AGE}, Relationship Son.",
            "Telephonic verification call {TELEPHONENUM} par kiya gaya.",
            "Bank ko email bheja from {EMAIL} with documents.",
            "Transaction {TRANSACTIONID} failed due to wrong IFSC {IFSC}.",
            "Mera zip code {ZIPCODE} update karna hai address mein.",
            "Kripya card {CREDITCARDNUM} ko unblock karo.",
            "Driving license {DRIVERLICENSENUM} aur passport {PASSPORTNUM} dono uploaded.",
            "Statement bhejna hai {EMAIL} pe for account {ACCOUNTNUM}.",
            "Account holder {FULLNAME}, city {CITY}, phone {TELEPHONENUM}.",
            "PAN {PAN} aur Aadhaar {AADHAAR} ke bina loan sanction nahi hoga.",
            "Nominee DOB {DATE}, Aadhaar {AADHAAR}, city {CITY}.",
            "Lost voter id {VOTERID}, new issue karwana hai.",
            "Transaction {TRANSACTIONID} success hua at {TIME}.",
            "Mobile number {TELEPHONENUM} aur email {EMAIL} dono update karna hai.",
            "Passport num {PASSPORTNUM}, voter id {VOTERID}, aadhaar {AADHAAR} required for verification.",
        
            # --- Name, Phone, Email, City ---
            "Mera naam {FULLNAME} hai aur main {CITY} se hoon. Call karo {TELEPHONENUM} par.",
            "{GIVENNAME} ji ka email {EMAIL} hai, please usi pe document bhejna.",
            "Contact number {TELEPHONENUM} update karna hai profile mein.",
            "Registered email {EMAIL} aur phone {TELEPHONENUM} verify karna.",
            "Branch {CITY} mein {DATE} ko visit karunga, please note down.",
            "Customer {FULLNAME} ki city {CITY} aur age {AGE} hai.",
            "Kripya OTP {EMAIL} par bhejo, number {TELEPHONENUM} pe nahi aaya.",
            "Email id {EMAIL} pe account statement bhejna hai.",

            # --- Aadhaar, PAN, Passport, Voter ---
            "Aadhaar {AADHAAR} aur PAN {PAN} dono upload kar diye hain.",
            "Passport number {PASSPORTNUM} verification ke liye dena hai.",
            "Voter ID {VOTERID} registered hai KYC ke liye.",
            "Mere Aadhaar {AADHAAR} mein galti hai, dob {DATE} sahi karna hai.",
            "PAN {PAN} aur Aadhaar {AADHAAR} mismatch ho raha hai.",
            "Passport {PASSPORTNUM} insurance ke liye diya gaya.",
            "Voter id {VOTERID} ke address ko {CITY} update karna hai.",
            "PAN {PAN}, Aadhaar {AADHAAR}, aur DL {DRIVERLICENSENUM} sab attach kiya hai.",

            # --- Account, IFSC, Banking ---
            "Beneficiary account {ACCOUNTNUM}, IFSC {IFSC} add karna hai.",
            "Mera salary account {ACCOUNTNUM} hai aur IFSC {IFSC}.",
            "Account number {ACCOUNTNUM} linked hai Aadhaar {AADHAAR} ke sath.",
            "Wrong IFSC {IFSC} enter ho gaya tha transfer mein.",
            "Cheque deposit account {ACCOUNTNUM} mein karna hai.",
            "Loan disbursement {ACCOUNTNUM} aur IFSC {IFSC} ke through hoga.",
            "Beneficiary {FULLNAME}, account {ACCOUNTNUM}, IFSC {IFSC}.",
            "Account {ACCOUNTNUM} mein unauthorized transaction {TRANSACTIONID} hua.",

            # --- Credit card, Transaction ---
            "Credit card {CREDITCARDNUM} se transaction {TRANSACTIONID} hua.",
            "Lost card {CREDITCARDNUM}, block karna hai immediately.",
            "Transaction {TRANSACTIONID} failed at {TIME}.",
            "Card {CREDITCARDNUM} ka due bill {DATE} tak pay karna hai.",
            "Statement of credit card {CREDITCARDNUM} {EMAIL} pe bhejna.",
            "Fraud alert: card {CREDITCARDNUM} se unauthorized payment.",
            "Transaction id {TRANSACTIONID} reference ke liye likh lo.",
            "Last transaction {DATE} ko hua tha, id {TRANSACTIONID}.",

            # --- Driving License ---
            "Driving license {DRIVERLICENSENUM} upload kiya KYC ke liye.",
            "DL {DRIVERLICENSENUM} aur passport {PASSPORTNUM} dono verify karna hai.",
            "License number {DRIVERLICENSENUM} expire ho gaya hai.",
            "Driver license {DRIVERLICENSENUM} pe naam {FULLNAME} likha hai.",
            "DL {DRIVERLICENSENUM} aur voter id {VOTERID} dono documents required.",

            # --- Address details ---
            "Mera address {BUILDINGNUM}, {STREET}, {CITY}, {ZIPCODE} hai.",
            "PIN code {ZIPCODE} galat update hua hai.",
            "Address change: {BUILDINGNUM} {STREET}, {CITY} {ZIPCODE}.",
            "New address: Flat {BUILDINGNUM}, {STREET}, {CITY}.",
            "Billing address: {BUILDINGNUM}, {STREET}, {CITY}, PIN {ZIPCODE}.",
            "Delivery address update karna hai: {BUILDINGNUM} {STREET}, {CITY}.",
            "House number {BUILDINGNUM}, street {STREET}, city {CITY}, PIN {ZIPCODE}.",
            "Address proof mein {CITY} aur {ZIPCODE} mention hai.",

            # --- Date, Time, Age, Gender ---
            "DOB {DATE} hai aur gender {GENDER} likha hai.",
            "Meeting scheduled {DATE} at {TIME} branch {CITY}.",
            "Nominee {FULLNAME}, age {AGE}, gender {GENDER}.",
            "Appointment {DATE} at {TIME} confirm karna.",
            "Policy ke liye DOB {DATE} aur Aadhaar {AADHAAR} attach karo.",
            "Transaction {TRANSACTIONID} hua tha {DATE} at {TIME}.",
            "Customer ka age {AGE} hai aur DOB {DATE} ke hisaab se match ho raha hai.",
            "Insurance holder {FULLNAME}, gender {GENDER}, DOB {DATE}.",

            # --- Mixed rich Hinglish ---
            "Hello, mera naam {GIVENNAME} hai, aadhaar {AADHAAR} aur pan {PAN} ke bina login nahi ho raha.",
            "Yaar, card {CREDITCARDNUM} block ho gaya, ab kya karun?",
            "Kripya {EMAIL} pe OTP bhej do, urgent hai.",
            "Account {ACCOUNTNUM} freeze ho gaya hai due to wrong PAN {PAN}.",
            "Passport {PASSPORTNUM} details ke bina foreign remittance nahi ho raha.",
            "Transaction id {TRANSACTIONID} se double debit ho gaya hai.",
            "Voter id {VOTERID} aur DL {DRIVERLICENSENUM} dono ki scan bhejni hai.",
            "Salary {ACCOUNTNUM} mein credit hoti hai har {DATE} ko.",
            "Loan ke liye guarantor {FULLNAME}, phone {TELEPHONENUM} diya hai.",
            "Mera number {TELEPHONENUM} aur email {EMAIL} dono update nahi ho pa rahe.",
            "Bank {CITY} branch mein {TIME} visit karunga.",
            "Credit card {CREDITCARDNUM} ka statement {EMAIL} par bhejna hoga.",
            "Insurance ke liye nominee {FULLNAME}, DOB {DATE}, aadhaar {AADHAAR}.",
            "Complaint registered for transaction {TRANSACTIONID} on {DATE}.",
            "Passport {PASSPORTNUM} aur voter id {VOTERID} dono expire ho gaye.",
            "Driving license {DRIVERLICENSENUM} renewal {DATE} ko due hai.",
            "Mobile banking {TELEPHONENUM} number se linked hai.",
            "Current address {BUILDINGNUM} {STREET}, {CITY}, {ZIPCODE}.",
            "Beneficiary {FULLNAME}, IFSC {IFSC}, account {ACCOUNTNUM}.",
            "PAN {PAN} aur Aadhaar {AADHAAR} ke bina KYC pending hai.",
            "Transaction failed {TRANSACTIONID} at {TIME}, retry karo.",
            "Mujhe new cheque book chahiye for account {ACCOUNTNUM}.",
            "ATM withdrawal {DATE} ko hua, id {TRANSACTIONID}.",
            "Card replacement ke liye {CREDITCARDNUM} submit kiya.",
            "Driver license {DRIVERLICENSENUM} and passport {PASSPORTNUM} verify ho gaya."
        ]


        # helpers
        self.bank_codes = ["HDFC", "ICIC", "SBIN", "PNB", "KOTK", "AXIS"]
        self.street_names = [
            "MG Road", "Brigade Road", "Station Road", "Park Street", "Mall Road", "Civil Lines", "Main Road",
            "Church Street", "Jawaharlal Nehru Road", "Lal Bahadur Shastri Marg", "Kingsway", "Ashoka Road",
            "Rajpath", "MG Marg", "Chowringhee Road", "Netaji Subhash Chandra Bose Road", "Connaught Place",
            "Park Avenue", "Victoria Road", "Raja Street", "Queen's Road", "Cantonment Road",
            "Churchgate Street", "Anna Salai", "Rajaji Road", "Dalhousie Road", "Sardar Patel Road",
            "Vijaya Road", "Gandhi Road", "Srinagar Street", "Station Road", "Bazaar Street", "Market Road",
            "MG Marg", "Airport Road", "Industrial Area Road", "Shastri Road", "New Market Road",
            "Old Delhi Road", "High Street", "Hospital Road", "Collectorate Road", "Water Tank Road",
            "Temple Street", "School Road", "Bank Street", "Tank Bund Road"
        ]


    # -------------------- Generators for each PII --------------------
    def _validate(self, label: str, value: str) -> bool:
        """Validate generated value against schema regex if available"""
        if not value:
            return False
        pat = self.schema_regex.get(label)
        if pat:
            return bool(pat.fullmatch(value))
        # if no schema regex present, accept by default
        return True

    def generate_givenname(self) -> str:
        return random.choice(self.indian_given_names)

    def generate_surname(self) -> str:
        return random.choice(self.indian_surnames)

    def generate_fullname(self) -> str:
        g = self.generate_givenname()
        s = self.generate_surname()
        return f"{g} {s}"

    def generate_telephone(self) -> str:
        # ensure first digit of the 10-digit number starts with 6-9
        first_digit = random.choice(["6", "7", "8", "9"])
        rest = ''.join(str(random.randint(0, 9)) for _ in range(9))
        core = first_digit + rest  # 10 digits starting w/ 6-9

        pattern = random.choice(["+91-{m}", "+91 {m}", "0{m}", "{m}"])
        return pattern.format(m=core)

    def generate_email(self, given: str, surname: str) -> str:
        username_patterns = [
            f"{given.lower()}.{surname.lower()}",
            f"{given.lower()}{surname.lower()}",
            f"{given.lower()}{random.randint(1,999)}",
            f"{given[:3].lower()}{surname[:3].lower()}{random.randint(10,99)}"
        ]
        domain = random.choice(["@gmail.com", "@yahoo.co.in", "@hdfc.com", "@icici.com", "@hotmail.com", "@indiatimes.com"])
        mail = random.choice(username_patterns) + domain
        # quick validation fallback (if schema requires stricter, rely on _validate)
        return mail

    def generate_aadhaar(self) -> str:
        while True:
            digits = ''.join(str(random.randint(0, 9)) for _ in range(12))
            val = f"{digits[:4]} {digits[4:8]} {digits[8:]}"
            if self._validate("AADHAAR", val):
                return val

    def generate_pan(self, surname: Optional[str] = None) -> str:
        """
        Generate PAN following the schema regex you provided:
        pattern: ^[A-Z]{3}[PFCHAT][A-Z]\\d{4}[A-Z]$
        We'll produce: 3 random letters + one of [PFCHAT] + surname initial (or random letter) + 4 digits + checksum letter
        """
        ent_types = list("PFCHAT")
        for _ in range(20):
            part1 = ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3))
            part2 = random.choice(ent_types)
            if surname:
                part3 = surname[0].upper()
            else:
                part3 = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            digits = ''.join(str(random.randint(0, 9)) for _ in range(4))
            last = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            pan = f"{part1}{part2}{part3}{digits}{last}"
            if self._validate("PAN", pan):
                return pan
        # fallback (shouldn't usually happen)
        return pan

    def generate_ifsc(self) -> str:
        for _ in range(10):
            bank = random.choice(self.bank_codes)
            branch = ''.join(str(random.randint(0, 9)) for _ in range(6))
            val = f"{bank}0{branch}"
            if self._validate("IFSC", val):
                return val
        return val

    def generate_accountnum(self) -> str:
        # 9-18 digits
        for _ in range(10):
            length = random.randint(9, 18)
            acc = ''.join(str(random.randint(0, 9)) for _ in range(length))
            if self._validate("ACCOUNTNUM", acc):
                return acc
        return acc

    def generate_voterid(self) -> str:
        for _ in range(10):
            pref = ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3))
            nums = ''.join(str(random.randint(0, 9)) for _ in range(7))
            val = pref + nums
            if self._validate("VOTERID", val):
                return val
        return val

    def generate_drivinglicense(self) -> str:
        # produce state(2 letters) + rto(2 digits) + year(4 digits) + 7-digit number, separated by '-' (matches schema)
        # states = ["MH", "DL", "KA", "TN", "UP", "GJ", "WB"]
        states = ["AP", "AN", "AR", "AS", "BR", "CG", "CH", "DN", "DD", "DL", "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"]
        for _ in range(10):
            state = random.choice(states)
            rto = f"{random.randint(1,99):02d}"
            year = str(random.randint(2009, 2022))
            num = f"{random.randint(0, 9999999):07d}"
            val = f"{state}-{rto}-{year}-{num}"
            if self._validate("DRIVERLICENSENUM", val):
                return val
        return val

    def generate_creditcard(self) -> str:
        digits = ''.join(str(random.randint(0, 9)) for _ in range(16))
        val = ' '.join(digits[i:i+4] for i in range(0, 16, 4))
        if self._validate("CREDITCARDNUM", val):
            return val
        return val

    def generate_zipcode(self) -> str:
        val = ''.join(str(random.randint(0, 9)) for _ in range(6))
        if self._validate("ZIPCODE", val):
            return val
        return val

    def generate_date(self) -> str:
        # produce dd/mm/YYYY or dd-mm-YYYY (schema expects these)
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        year = random.randint(2015, 2025)
        sep = random.choice(["/", "-"])
        val = f"{day:02d}{sep}{month:02d}{sep}{year}"
        if self._validate("DATE", val):
            return val
        return val

    def generate_time(self) -> str:
        hh = random.randint(0, 23)
        mm = random.randint(0, 59)
        use_seconds = random.random() < 0.2
        if use_seconds:
            ss = random.randint(0, 59)
            val = f"{hh:02d}:{mm:02d}:{ss:02d}"
        else:
            val = f"{hh:02d}:{mm:02d}"
        if self._validate("TIME", val):
            return val
        return val

    def generate_age(self) -> str:
        val = str(random.randint(18, 99))
        if self._validate("AGE", val):
            return val
        return val

    def generate_transactionid(self) -> str:
        return str(uuid.uuid4())

    def generate_gender(self) -> str:
        return random.choice(["M", "F", "Male", "Female"])
    
    def generate_passportnum(self) -> str:
        # Indian Passport format: 1 capital letter + 7 digits (e.g., A1234567)
        letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        digits = "".join(random.choices("0123456789", k=7))
        return letter + digits


    # -------------------- Token/label utilities --------------------
    def annotate_tokens_with_bio(self, text: str, entities: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """
        Tokenize with fast tokenizer and produce BIO labels aligned to tokens using offsets.
        - tokens: list of token strings (as returned by convert_ids_to_tokens)
        - labels: list of BIO labels: O / B-<LABEL> / I-<LABEL>
        """
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])
        offsets = enc["offset_mapping"]  # list of (start, end) per token

        labels = ["O"] * len(tokens)

        # prepare entity spans (start, end, label)
        spans = []
        for etype, val in entities.items():
            if not val:
                continue
            # find first occurrence
            idx = text.find(val)
            if idx == -1:
                # try normalized search (strip spaces)
                idx = text.replace(" ", "").find(val.replace(" ", ""))
                if idx == -1:
                    continue
            spans.append((idx, idx + len(val), etype))

        # assign labels per token using offsets
        for (s, e, etype) in spans:
            started = False
            for i, (tok_s, tok_e) in enumerate(offsets):
                # skip tokens with zero-length mapping, though usually none with add_special_tokens=False
                if tok_s == tok_e == 0:
                    continue
                # check overlap
                if not (tok_e <= s or tok_s >= e):
                    if not started:
                        labels[i] = f"B-{etype}"
                        started = True
                    else:
                        labels[i] = f"I-{etype}"
        return tokens, labels

    # -------------------- privacy mask & masked text --------------------
    def build_privacy_mask(self, text: str, entities: Dict[str, str]) -> List[Dict]:
        masks = []
        for etype, val in entities.items():
            if not val:
                continue
            start = text.find(val)
            if start == -1:
                # skip if not found
                continue
            masks.append({
                "label": etype,
                "start": start,
                "end": start + len(val),
                "value": val,
                "label_index": 1
            })
        return masks

    def build_masked_text(self, text: str, privacy_mask: List[Dict]) -> str:
        # replace by descending start positions to not break indices
        spans = sorted(privacy_mask, key=lambda x: x["start"], reverse=True)
        masked = text
        for i, sp in enumerate(spans, start=1):
            placeholder = f"[{sp['label']}_1]"
            s, e = sp["start"], sp["end"]
            masked = masked[:s] + placeholder + masked[e:]
        return masked

    # -------------------- Single sample generation --------------------
    def generate_sample(self) -> Dict:
        template = random.choice(self.templates)

        # choose base random values first so dependent fields can use them
        given = self.generate_givenname()
        surname = self.generate_surname()
        fullname = f"{given} {surname}"
        city = random.choice(self.indian_cities)

        entities: Dict[str, str] = {}

        # Fill placeholders (order matters for dependencies)
        if "{GIVENNAME}" in template:
            entities["GIVENNAME"] = given
            template = template.replace("{GIVENNAME}", given)
        if "{SURNAME}" in template:
            entities["SURNAME"] = surname
            template = template.replace("{SURNAME}", surname)
        if "{FULLNAME}" in template:
            entities["FULLNAME"] = fullname
            template = template.replace("{FULLNAME}", fullname)
        if "{CITY}" in template:
            entities["CITY"] = city
            template = template.replace("{CITY}", city)
        if "{TELEPHONENUM}" in template:
            phone = self.generate_telephone()
            entities["TELEPHONENUM"] = phone
            template = template.replace("{TELEPHONENUM}", phone)
        if "{EMAIL}" in template:
            email = self.generate_email(given, surname)
            entities["EMAIL"] = email
            template = template.replace("{EMAIL}", email)
        if "{AADHAAR}" in template:
            entities["AADHAAR"] = self.generate_aadhaar()
            template = template.replace("{AADHAAR}", entities["AADHAAR"])
        if "{PAN}" in template:
            entities["PAN"] = self.generate_pan(surname=surname)
            template = template.replace("{PAN}", entities["PAN"])
        if "{IFSC}" in template:
            entities["IFSC"] = self.generate_ifsc()
            template = template.replace("{IFSC}", entities["IFSC"])
        if "{ACCOUNTNUM}" in template:
            entities["ACCOUNTNUM"] = self.generate_accountnum()
            template = template.replace("{ACCOUNTNUM}", entities["ACCOUNTNUM"])
        if "{CREDITCARDNUM}" in template:
            entities["CREDITCARDNUM"] = self.generate_creditcard()
            template = template.replace("{CREDITCARDNUM}", entities["CREDITCARDNUM"])
        if "{BUILDINGNUM}" in template:
            entities["BUILDINGNUM"] = str(random.randint(1, 9999))
            template = template.replace("{BUILDINGNUM}", entities["BUILDINGNUM"])
        if "{STREET}" in template:
            s = random.choice(self.street_names)
            entities["STREET"] = s
            template = template.replace("{STREET}", s)
        if "{ZIPCODE}" in template:
            entities["ZIPCODE"] = self.generate_zipcode()
            template = template.replace("{ZIPCODE}", entities["ZIPCODE"])
        if "{DRIVERLICENSENUM}" in template:
            entities["DRIVERLICENSENUM"] = self.generate_drivinglicense()
            template = template.replace("{DRIVERLICENSENUM}", entities["DRIVERLICENSENUM"])
        if "{VOTERID}" in template:
            entities["VOTERID"] = self.generate_voterid()
            template = template.replace("{VOTERID}", entities["VOTERID"])
        if "{AGE}" in template:
            entities["AGE"] = self.generate_age()
            template = template.replace("{AGE}", entities["AGE"])
        if "{DATE}" in template:
            entities["DATE"] = self.generate_date()
            template = template.replace("{DATE}", entities["DATE"])
        if "{TIME}" in template:
            entities["TIME"] = self.generate_time()
            template = template.replace("{TIME}", entities["TIME"])
        if "{TRANSACTIONID}" in template:
            entities["TRANSACTIONID"] = self.generate_transactionid()
            template = template.replace("{TRANSACTIONID}", entities["TRANSACTIONID"])
        if "{PASSPORTNUM}" in template:
            entities["PASSPORTNUM"] = self.generate_passportnum()
            template = template.replace("{PASSPORTNUM}", entities["PASSPORTNUM"])


        # small Hinglish flavor replacements (light touch)
        if random.random() < 0.25:
            template = template.replace("Please", random.choice(["Please", "Kripya", "Please bhai"]))
        if random.random() < 0.15:
            template = template.replace("please", random.choice(["please", "kripya"]))

        source_text = template

        # build privacy mask, masked text and token labels
        privacy_mask = self.build_privacy_mask(source_text, entities)
        masked_text = self.build_masked_text(source_text, privacy_mask)
        tokens, token_labels = self.annotate_tokens_with_bio(source_text, entities)

        record = {
            "source_text": source_text,
            "masked_text": masked_text,
            "privacy_mask": privacy_mask,  # keep as list; outer writer will json-encode
            "uid": str(uuid.uuid4().int)[:16],
            "language": "hi-en",
            "region": "IN",
            "script": "Latn",
            "mBERT_tokens": tokens,
            "mBERT_token_classes": token_labels,
            "entities": entities
        }
        return record

    # -------------------- Dataset generation --------------------
    def generate_dataset(self, num_samples: int = 1000, out_path: str = "data/synthetic/hinglish_banking.jsonl"):
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with out_file.open("w", encoding="utf-8") as fh:
            for _ in tqdm(range(num_samples), desc="Generating samples"):
                rec = self.generate_sample()
                # ensure privacy_mask values match substrings (quick integrity)
                valid = True
                for m in rec["privacy_mask"]:
                    # quick check
                    s, e, val = m["start"], m["end"], m["value"]
                    if rec["source_text"][s:e] != val:
                        valid = False
                        break
                if not valid:
                    # skip this sample (rare)
                    continue
                # dump record (encode lists/dicts properly)
                dump = {
                    "source_text": rec["source_text"],
                    "masked_text": rec["masked_text"],
                    "privacy_mask": rec["privacy_mask"],
                    "uid": rec["uid"],
                    "language": rec["language"],
                    "region": rec["region"],
                    "script": rec["script"],
                    "mBERT_tokens": rec["mBERT_tokens"],
                    "mBERT_token_classes": rec["mBERT_token_classes"],
                    "entities": rec["entities"]
                }
                fh.write(json.dumps(dump, ensure_ascii=False) + "\n")
                written += 1
        print(f"Wrote {written} records to {out_path}")


# -------------------- CLI --------------------
def cli():
    parser = argparse.ArgumentParser(description="Generate synthetic Hinglish banking PII JSONL")
    parser.add_argument("--num", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="data/synthetic/hinglish_banking.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--schema", type=str, default="data/docs/data_schema.json", help="Path to data schema JSON")
    args = parser.parse_args()

    gen = HinglishBankingGenerator(seed=args.seed, schema_path=args.schema)
    gen.generate_dataset(num_samples=args.num, out_path=args.out)


if __name__ == "__main__":
    # If run directly, generate a small sample and print a few examples, then exit.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo", action="store_true", help="Run small demo and print examples")
    parser.add_argument("--num", type=int, default=5, help="How many demo samples to print")
    parser.add_argument("--out", type=str, default=None, help="If provided, write JSONL to this path")
    parser.add_argument("--schema", type=str, default="data/docs/data_schema.json")
    parser.add_argument("--seed", type=int, default=42)
    args, remaining = parser.parse_known_args()

    generator = HinglishBankingGenerator(seed=args.seed, schema_path=args.schema)
    if args.demo:
        for i in range(args.num):
            s = generator.generate_sample()
            print(f"\n=== Sample {i+1} ===")
            print(f"Source: {s['source_text']}")
            print(f"Masked: {s['masked_text']}")
            print(f"Entities: {json.dumps(s['entities'], ensure_ascii=False)}")
            print(f"Tokens (first 20): {s['mBERT_tokens'][:20]}")
            print(f"Token Labels (first 20): {s['mBERT_token_classes'][:20]}")
        if args.out:
            generator.generate_dataset(num_samples=args.num, out_path=args.out)
    else:
        # fallback to CLI generation
        cli()