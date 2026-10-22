"""
Generate a synthetic student grievance dataset for a college.

Each grievance is built by joining random pieces:
    opener + problem phrase + (optional) detail + (optional) closer

Every problem phrase carries a priority label (High / Medium / Low), so the
priority of a grievance comes from the problem it describes.

To keep the data realistic (and the model accuracy believable) we add:
    - words shared across categories ("hostel", "portal", "fees", "exam", "office")
    - grievances that mix two topics, labelled with the main topic
    - typos, missing punctuation, lowercase / ALL CAPS rows, filler words
    - short one-liners and longer multi-sentence grievances
    - a small share of label noise (humans do not always agree on a label)

Run from the project root:
    python data/generate_dataset.py
"""

import os
import random

import pandas as pd

SEED = 42                 # fixed seed so the dataset is the same every run
ROWS_PER_CATEGORY = 400   # 6 categories x 400 = 2400 rows
MIX_RATE = 0.09           # about 9% of rows mention a second topic
VAGUE_RATE = 0.10         # share of rows that are short and vague
LABEL_NOISE_RATE = 0.02   # share of rows a "human labeller" put in a different category

# Output path is relative to this file, so the script works from any folder
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "complaints.csv")

# ---------------------------------------------------------------------------
# Building blocks shared by every category
# ---------------------------------------------------------------------------

OPENERS = [
    "", "", "",  # empty strings make "no opener" more likely
    "Respected sir,", "Respected madam,", "Dear sir/madam,", "Sir,", "Hello,",
    "I am a second year student.", "I am a final year student.", "I am a first year student.",
    "I want to raise a complaint.", "This is regarding my problem.", "With due respect,",
    "I am writing this grievance.", "Good morning sir,", "I am from the CSE department.",
    "I am from the mechanical branch.",
]

CLOSERS = [
    "", "", "", "",
    "Kindly do the needful.", "Kindly look into this matter.", "Thanking you.", "Yours sincerely.",
    "Please take action.", "This is affecting my studies.", "I hope this will be solved soon.",
    "Many students are facing the same issue.", "I have already complained at the office.",
    "Please reply soon.", "Nobody in the office is responding.", "Please help.",
    "My parents are also worried.", "I request you to solve this.",
]

# Filler words that real people type but carry no meaning
FILLERS = ["basically", "like", "actually", "honestly", "literally", "really", "just", "so", "um"]

# Words added to some High priority rows to make urgency explicit
URGENT_TAGS = [
    "This is urgent.", "Please act immediately.", "Kindly resolve this asap.",
    "URGENT!", "It is an emergency.",
]

# ---------------------------------------------------------------------------
# Category-specific problem phrases, tagged with a priority.
# Shared words like "hostel", "portal", "fees", "exam" and "office" appear in
# several categories on purpose, so the model cannot rely on them alone.
# Medium and Low phrases avoid the urgent keywords used by the priority rules.
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Hostel & Mess": {
        "High": [
            "many students got food poisoning after the mess dinner",
            "there has been no drinking water in the hostel for two days",
            "there are electric sparks from the switchboard in my hostel room",
            "my laptop was stolen from my hostel room because the door lock is broken",
            "the ceiling plaster in our hostel room fell and nearly hit a student",
            "we found insects in the mess food and two students are in the hospital",
        ],
        "Medium": [
            "the mess food quality is very poor and often stale",
            "there is no hot water in the hostel bathrooms",
            "the hostel washroom has been leaking for a week",
            "my hostel room has not been cleaned for days",
            "the hostel room fan and light are not working",
            "there are too many mosquitoes in the hostel block",
            "the mess serves food late and it is always cold",
            "the hostel water cooler is broken",
        ],
        "Low": [
            "the mess menu is the same every week",
            "please add more variety to the mess breakfast",
            "the common room TV in the hostel is not working",
            "the laundry service in the hostel is slow",
            "the hostel gym equipment is old",
            "please extend the hostel library reading room hours",
        ],
    },
    "Exams & Results": {
        "High": [
            "my hall ticket is not generated and the exam is tomorrow",
            "I was marked absent in the exam even though I wrote it",
            "my result shows fail but I am sure I passed the subject",
            "two of my exams are scheduled at the same time tomorrow",
            "the exam cell lost my answer sheet",
            "my marks are missing from the result and placement verification is this week",
        ],
        "Medium": [
            "my revaluation result has been pending for two months",
            "my exam marks are not updated on the portal",
            "the internal marks in my marksheet are wrong",
            "the exam timetable was announced very late",
            "I have not received my semester marksheet yet",
            "the supplementary exam form date is not announced",
            "my name is spelled wrong on the hall ticket",
            "the question paper had questions out of syllabus",
        ],
        "Low": [
            "the exam hall fans were not working",
            "please give more gap between two exams",
            "the result format on the portal is confusing",
            "the exam hall clock was not visible",
            "please announce exam seating plans earlier",
            "the answer booklet pages are too few",
        ],
    },
    "Fees & Scholarships": {
        "High": [
            "I paid my semester fees twice and the money is not refunded",
            "money was deducted from my account but the portal shows fees unpaid with late fine",
            "my scholarship has not come and I am told I will be removed from the hostel",
            "someone used my student login to change my bank details for the scholarship, this looks like fraud",
            "I am not allowed to write the exam because the fee payment is not reflected",
            "the accounts office charged me a huge fine by mistake",
        ],
        "Medium": [
            "my scholarship amount is pending for three months",
            "my caution deposit refund is still pending",
            "the fee receipt is not generated after payment",
            "I was charged a late fee even though I paid on time",
            "the hostel fee amount shown on the portal is wrong",
            "my fee concession is not applied",
            "the bus fee was charged even though I do not use the college bus",
            "the scholarship form is rejected without any reason",
        ],
        "Low": [
            "please allow paying fees in installments",
            "the fee structure on the website is not clear",
            "there is a spelling mistake on my fee receipt",
            "please add UPI as a fee payment option",
            "the accounts office counter opens late",
            "please send fee reminders by email",
        ],
    },
    "Academics & Faculty": {
        "High": [
            "my attendance is wrongly marked short and I may be detained from exams",
            "a faculty member has not taken classes for a month before the exams",
            "a chemical spilled in the lab and the lab has no safety equipment",
            "our project guide refused to sign the final submission due tomorrow",
            "the department has not uploaded my internship credits and my degree is on hold",
            "the lab machine is dangerous and a student got hurt",
        ],
        "Medium": [
            "the syllabus is not completed and exams are near",
            "the faculty is absent for many classes",
            "our lab sessions are cancelled every week",
            "assignment marks have not been given for this semester",
            "two lectures are clashing in our timetable",
            "the professor does not explain topics clearly",
            "the elective subject I chose was changed without informing me",
            "my attendance is not updated on the portal",
        ],
        "Low": [
            "the faculty does not share notes after class",
            "please arrange more practical sessions",
            "the classes start late most days",
            "the projector in the classroom is dim",
            "please add more books for our subject in the library",
            "the department notice board is not updated",
        ],
    },
    "IT & Wi-Fi": {
        "High": [
            "the student portal is down on the last date of exam registration",
            "my college email account was hacked and is sending spam",
            "the ERP login is not working and the fee deadline is today",
            "the online exam portal crashed while I was writing the exam",
            "unauthorized login attempts are happening on my student account",
            "the assignment submission portal is not opening and the deadline is tonight",
        ],
        "Medium": [
            "the wifi in the hostel has not worked for a week",
            "the LMS is not loading assignments",
            "most computers in the lab are not working",
            "the password reset option on the portal is not working",
            "I cannot log in to the college wifi with my roll number",
            "the online class link keeps disconnecting",
            "the college app crashes when I open my timetable",
            "the portal shows someone else's profile when I log in",
        ],
        "Low": [
            "the wifi in the library is a bit slow",
            "please add more charging points in the lab",
            "the printer in the computer centre is slow",
            "the college website looks outdated",
            "please give students more cloud storage",
            "the portal logs me out too often",
        ],
    },
    "Ragging & Harassment": {
        "High": [
            "seniors are ragging me in the hostel every night",
            "a group of seniors threatened me and took my money",
            "I am being harassed by a classmate through messages",
            "seniors physically assaulted me near the hostel",
            "a staff member is harassing female students",
            "I am a victim of bullying and I feel unsafe on campus",
        ],
        "Medium": [
            "seniors force me to do their assignments",
            "some seniors make fun of my language and accent",
            "a faculty member insulted me in front of the class",
            "the office staff shouted at me when I asked about my documents",
            "my classmates keep posting mean comments about me in the group",
            "seniors make juniors stand outside the hostel late at night",
            "the hostel staff misbehaved with my parents",
            "the security guard was very rude and pushed students",
        ],
        "Low": [
            "the office staff is not polite with students",
            "the librarian speaks rudely to students",
            "the staff ignored my question at the counter",
            "the canteen staff is impolite",
            "the lab assistant does not respond properly",
            "some seniors make jokes about juniors",
        ],
    },
}

# Extra detail sentences per category. Some details use shared words such as
# "hostel", "portal", "fees" and "office" so categories overlap in vocabulary.
DETAILS = {
    "Hostel & Mess": [
        "I stay in hostel block C.", "The warden has been told already.", "We pay high hostel fees for this.",
        "This has been happening since the start of the semester.", "Other students in my wing face this too.",
        "I complained at the hostel office.", "It is very hard to study like this.", "The mess contractor does not listen.",
    ],
    "Exams & Results": [
        "My roll number is on the portal.", "I contacted the exam cell office twice.",
        "This semester exam is very important for me.", "I have the screenshot of the portal.",
        "My placement depends on this.", "Other students from my class have the same issue.",
        "I paid the exam fees on time.", "The exam cell told me to wait.",
    ],
    "Fees & Scholarships": [
        "I have the bank statement as proof.", "I paid through the college portal.",
        "The accounts office keeps sending me to another counter.", "My parents took a loan for these fees.",
        "My transaction ID is in the email.", "I submitted all scholarship documents.",
        "This is for the current semester.", "The hostel office also asked me about this.",
    ],
    "Academics & Faculty": [
        "Our class representative already informed the HOD.", "The exam is in two weeks.",
        "Many students in my class are worried.", "This is for the third semester.",
        "We have no study material for this subject.", "I have attended all the classes.",
        "The department office is not responding.", "This affects our internal marks.",
    ],
    "IT & Wi-Fi": [
        "I tried on both phone and laptop.", "I cleared the browser cache already.",
        "The IT office asked me to wait.", "My roll number is my login ID.",
        "It happens in the hostel and on campus.", "I need it for my online exam.",
        "The portal shows an error message.", "Other students have the same issue.",
    ],
    "Ragging & Harassment": [
        "This happens in the hostel at night.", "I am scared to tell the warden.",
        "I have screenshots of the messages.", "It happened outside the exam hall.",
        "Other juniors are also facing this.", "I do not want my name disclosed.",
        "It happened near the office building.", "My parents do not know about this yet.",
    ],
}

# Short, vague grievances that could belong anywhere. They are labelled with a
# category but contain few clear clues, which makes the task harder (like real data).
VAGUE = {
    "Hostel & Mess": [
        "hostel problem again", "please do something about the hostel", "mess issue sir",
        "facing problem in hostel, office is not helping", "the hostel situation is bad",
    ],
    "Exams & Results": [
        "exam problem please check", "issue with my result", "exam related issue",
        "problem with the portal for my exam", "office is not helping with my exam issue",
    ],
    "Fees & Scholarships": [
        "fees problem", "issue with my payment", "scholarship issue sir",
        "problem with fees on the portal", "the office is not solving my fees issue",
    ],
    "Academics & Faculty": [
        "class problem", "issue with our subject", "problem in the department",
        "teaching is not good", "the department office is not helping",
    ],
    "IT & Wi-Fi": [
        "portal not working", "internet issue", "login problem",
        "something is wrong with the portal", "network problem in hostel",
    ],
    "Ragging & Harassment": [
        "seniors are troubling me", "someone is troubling me in hostel", "facing problem from seniors",
        "bad behaviour from staff in the office", "people in my class trouble me",
    ],
}

PRIORITY_WEIGHTS = {"High": 0.3, "Medium": 0.45, "Low": 0.25}

# Keyboard-neighbour typos give realistic spelling mistakes
NEIGHBOURS = {
    "a": "sq", "e": "wr", "i": "uo", "o": "ip", "u": "yi", "s": "ad", "t": "ry",
    "n": "bm", "r": "et", "l": "k", "d": "sf", "c": "xv", "m": "n", "h": "gj",
}


def add_typos(text, rng, rate=0.04):
    """Randomly swap, drop or replace characters to simulate typing mistakes."""
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].lower() in NEIGHBOURS and rng.random() < rate:
            choice = rng.random()
            if choice < 0.4:
                chars[i] = rng.choice(NEIGHBOURS[chars[i].lower()])  # wrong key
            elif choice < 0.7:
                chars[i] = ""  # missed key
            elif i + 1 < len(chars):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]  # swapped keys
    return "".join(chars)


def add_fillers(text, rng):
    """Insert one or two filler words at random positions."""
    words = text.split()
    for _ in range(rng.randint(1, 2)):
        pos = rng.randint(0, len(words))
        words.insert(pos, rng.choice(FILLERS))
    return " ".join(words)


def capitalise(sentence):
    """Make the first letter uppercase and end the sentence with a full stop."""
    sentence = sentence.strip()
    if not sentence:
        return sentence
    sentence = sentence[0].upper() + sentence[1:]
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def build_complaint(category, rng):
    """Build one grievance for a category and return (text, priority)."""
    # 1. Pick the priority, then a problem phrase with that priority
    priority = rng.choices(list(PRIORITY_WEIGHTS), weights=PRIORITY_WEIGHTS.values())[0]

    # Some rows are short vague grievances with few clues (hard to classify)
    if rng.random() < VAGUE_RATE:
        text = rng.choice(VAGUE[category])
        priority = rng.choice(["Medium", "Low"])
        return text, priority

    problem = rng.choice(CATEGORIES[category][priority])
    parts = [rng.choice(OPENERS), capitalise(problem)]

    # 2. Optional detail sentences decide the length (one-liner to 3+ sentences)
    n_details = rng.choices([0, 1, 2], weights=[0.35, 0.45, 0.20])[0]
    parts += rng.sample(DETAILS[category], n_details)

    # 3. Mixed topic: mention a problem from another category as a side issue.
    #    The label stays the main category, as a human officer would decide.
    if rng.random() < MIX_RATE:
        other = rng.choice([c for c in CATEGORIES if c != category])
        side_problem = rng.choice(CATEGORIES[other]["Medium"] + CATEGORIES[other]["Low"])
        side_sentence = capitalise(side_problem) + " " + rng.choice(DETAILS[other])
        if rng.random() < 0.5:
            parts.append("Also, " + side_sentence)   # side issue after the main one
        else:
            parts.insert(1, side_sentence + " But the main problem is:")  # side issue first

    # 4. Some High rows carry explicit urgency words
    if priority == "High" and rng.random() < 0.35:
        parts.append(rng.choice(URGENT_TAGS))

    parts.append(rng.choice(CLOSERS))
    text = " ".join(p for p in parts if p)
    return text, priority


def add_noise(text, rng):
    """Apply random real-world noise: fillers, typos, case changes, no punctuation."""
    if rng.random() < 0.25:
        text = add_fillers(text, rng)
    if rng.random() < 0.40:
        text = add_typos(text, rng, rate=rng.choice([0.03, 0.06, 0.10]))
    r = rng.random()
    if r < 0.15:
        text = text.lower()            # all-lowercase writer
    elif r < 0.20:
        text = text.upper()            # ANGRY ALL CAPS writer
    if rng.random() < 0.15:
        text = "".join(ch for ch in text if ch not in ".,!?")  # no punctuation
    return text


def main():
    """Generate the dataset, shuffle it and save it as a CSV file."""
    rng = random.Random(SEED)
    rows = []
    for category in CATEGORIES:
        for _ in range(ROWS_PER_CATEGORY):
            text, priority = build_complaint(category, rng)
            label = category
            # Real datasets have some labelling disagreement between humans
            if rng.random() < LABEL_NOISE_RATE:
                label = rng.choice([c for c in CATEGORIES if c != category])
            rows.append({
                "complaint_text": add_noise(text, rng),
                "category": label,
                "priority": priority,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle rows
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} grievances to {OUTPUT_PATH}")
    print("\nRows per category:")
    print(df["category"].value_counts().to_string())
    print("\nRows per priority:")
    print(df["priority"].value_counts().to_string())
    print(f"\nUnique grievance texts: {df['complaint_text'].nunique()}")


if __name__ == "__main__":
    main()
