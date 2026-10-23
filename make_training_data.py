"""
make_training_data.py
=====================
Generates a SYNTHETIC labeled training set for the Email and PDF ML models.

    python make_training_data.py            # writes dataset/email_dataset.csv
                                            #        dataset/email_families.csv
                                            #        dataset/pdf_features.csv
                                            #        dataset/pdf_families.csv

READ THIS FIRST - what synthetic data can and cannot tell you
-------------------------------------------------------------
This project has no real labeled e-mail or PDF corpus and the build machine had
no internet access, so the examples are GENERATED from many templates
("families": workplace mail, bank-KYC scams, CEO fraud, delivery-fee scams ...).
Consequences:
  * The model learns the patterns of THESE templates. Accuracy measured on a
    random split of this data is inflated and is NOT real-world accuracy.
  * evaluate_models.py therefore also reports leave-one-FAMILY-out results
    (train on some scam types, test on a type never seen) and results on the
    hand-written samples in email_pdf_fixtures.py, which are never used here.
  * For production use, retrain with real data (see train_email_model.py /
    train_pdf_model.py docstrings) - the pipeline is identical.

Labels are assigned by construction (what a human analyst would call the
example). Nothing is sent over the network; PDFs are generated in memory and
only parsed by the static analyzer. All domains are made-up or reserved.
"""

import argparse
import csv
import os
import random
import re
import tempfile

import email_pdf_fixtures as F
from email_analyzer import build_model_text, extract_bare_domains, extract_urls_from_text

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "dataset")


# ---------------------------------------------------------------------------
# shared vocab
# ---------------------------------------------------------------------------
FIRST = ["Priya", "Rahul", "Anita", "Ravi", "Maya", "John", "Sara", "Omar", "Li", "Elena", "Carlos", "Fatima",
         "David", "Meera", "Tom", "Aisha", "Kenji", "Nora", "Arjun", "Grace"]
LAST = ["Sharma", "Kumar", "Patel", "Singh", "Brown", "Garcia", "Khan", "Chen", "Rossi", "Nguyen", "Ali", "Smith"]
COMPANIES = ["acme-corp", "northwind", "blueriver", "greenfield", "orbitech", "lumina", "kestrel", "maplewood",
             "ironbridge", "sunrise-labs", "quantix", "harbor-co"]
BRANDS = ["PayPal", "Microsoft", "Apple", "Google", "Amazon", "Netflix", "Facebook", "LinkedIn", "Dropbox",
          "DocuSign", "DHL", "FedEx", "HDFC Bank", "SBI", "ICICI Bank", "Paytm", "Chase", "Wells Fargo"]
REAL = {"PayPal": "paypal.com", "Microsoft": "microsoft.com", "Apple": "apple.com", "Google": "google.com",
        "Amazon": "amazon.com", "Netflix": "netflix.com", "Facebook": "facebook.com", "LinkedIn": "linkedin.com",
        "Dropbox": "dropbox.com", "DocuSign": "docusign.com", "DHL": "dhl.com", "FedEx": "fedex.com",
        "HDFC Bank": "hdfcbank.com", "SBI": "onlinesbi.com", "ICICI Bank": "icicibank.com", "Paytm": "paytm.com",
        "Chase": "chase.com", "Wells Fargo": "wellsfargo.com"}
BAD_TLDS = ["xyz", "top", "click", "work", "support", "icu", "buzz", "monster", "online", "site"]
FREE = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me", "mail.com"]
WORDS = ["secure", "login", "verify", "account", "support", "update", "service", "alert", "billing", "online"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["9am", "10:30am", "2pm", "3pm", "4:15pm", "5pm"]


def person(r):
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def company_domain(r):
    return r.choice(COMPANIES) + r.choice([".example", ".example.org", ".example.com"])


def typo_brand(name, r):
    b = re.sub(r"[^a-z]", "", name.lower())
    kind = r.choice(["digit", "swap", "double", "drop", "plain"])
    if kind == "digit":
        for a, d in (("l", "1"), ("o", "0"), ("i", "1"), ("a", "4"), ("e", "3")):
            if a in b:
                return b.replace(a, d, 1)
    if kind == "swap" and len(b) > 4:
        i = r.randrange(1, len(b) - 2)
        return b[:i] + b[i + 1] + b[i] + b[i + 2:]
    if kind == "double":
        i = r.randrange(1, len(b))
        return b[:i] + b[i] + b[i:]
    if kind == "drop" and len(b) > 5:
        i = r.randrange(1, len(b) - 1)
        return b[:i] + b[i + 1:]
    return b


def bad_domain(r, brand=None):
    """A suspicious domain: look-alike / brand + wording / odd TLD / raw IP / shortener."""
    kind = r.choice(["lookalike", "wordy", "wordy_tld", "ip", "short", "sub", "plain_bad"])
    b = typo_brand(brand or r.choice(BRANDS), r)
    if kind == "lookalike":
        return f"{b}.{r.choice(['com', 'net', 'info'])}"
    if kind == "wordy":
        return f"{b}-{r.choice(WORDS)}.{r.choice(['com', 'net', 'org'])}"
    if kind == "wordy_tld":
        return f"{r.choice(WORDS)}-{b}.{r.choice(BAD_TLDS)}"
    if kind == "ip":
        return f"203.0.113.{r.randint(2, 250)}"
    if kind == "short":
        return r.choice(["bit.ly", "tinyurl.com", "cutt.ly", "is.gd"])
    if kind == "sub":
        return f"{re.sub(r'[^a-z]', '', (brand or 'paypal').lower())}.com.{r.choice(WORDS)}-{r.choice(COMPANIES)}.example"
    return f"{r.choice(COMPANIES)}-{r.choice(WORDS)}.{r.choice(BAD_TLDS)}"


def url_for(domain, r):
    path = r.choice(["login", "verify", "account/confirm", "secure/update", "track/12345", "view", "pay", "claim"])
    scheme = r.choice(["http://", "https://"])
    return f"{scheme}{domain}/{path}"


def add_noise(text, r, level=0.02):
    """Random typos / case changes so the model cannot rely on perfect spelling."""
    out = []
    for ch in text:
        if ch.isalpha() and r.random() < level:
            op = r.choice(["drop", "dup", "case"])
            if op == "drop":
                continue
            if op == "dup":
                out.append(ch)
            if op == "case":
                ch = ch.swapcase()
        out.append(ch)
    s = "".join(out)
    if r.random() < 0.05:
        s = s.upper()
    return s


def pick(r, items, lo, hi):
    n = r.randint(lo, min(hi, len(items)))
    return r.sample(items, n)


# ---------------------------------------------------------------------------
# EMAIL families.  Each family returns (sender, subject, body)
# ---------------------------------------------------------------------------
def fam_workplace(r):
    n = person(r)
    s = [f"Hi team, the planning meeting moved to {r.choice(DAYS)} at {r.choice(TIMES)}.",
         "Notes from the last sync are in the shared folder, please add your comments.",
         "Can you review the draft before the end of the week?",
         "I updated the project board and closed the finished tickets.",
         "The client asked for a short summary of the milestones so far.",
         "Reminder: timesheets are due on Friday afternoon.",
         "Lunch is on me if we finish the release before noon.",
         f"Please find the agenda for {r.choice(DAYS)}'s review below.",
         "Thanks for covering my shift last week, I owe you one.",
         "We need two more volunteers for the onboarding session."]
    subj = r.choice(["Meeting notes", "Re: sprint review", "Agenda for next week", "Quick question", "Draft for review", "Timesheets"])
    body = " ".join(pick(r, s, 2, 4)) + f" Thanks, {n.split()[0]}"
    return f"{n} <{n.split()[0].lower()}@{company_domain(r)}>", subj, body


def fam_dev_notifications(r):
    app = r.choice(["GitHub", "GitLab", "Jira", "Slack", "Trello", "Notion"])
    dom = {"GitHub": "github.com", "GitLab": "gitlab.com", "Jira": "atlassian.com", "Slack": "slack.com",
           "Trello": "trello.com", "Notion": "notion.so"}[app]
    body = r.choice([
        f"Your pull request was merged into main. View it at {url_for(dom, r)}",
        f"{person(r)} commented on issue #{r.randint(10, 999)}: looks good to me, merging after tests pass.",
        f"The build for branch release-{r.randint(1, 9)}.{r.randint(0, 9)} passed. Details: https://{dom}/builds/{r.randint(100, 999)}",
        f"You were mentioned in a thread. Open the conversation at https://{dom}/thread/{r.randint(1000, 9999)}",
    ])
    return f"{app} <noreply@{dom}>", r.choice([f"[{app}] Activity update", f"[{app}] New comment", "Build passed"]), body


def fam_orders_real(r):
    b = r.choice(["Amazon", "Apple", "Netflix", "Google", "Microsoft", "PayPal", "LinkedIn"])
    d = REAL[b]
    body = r.choice([
        f"Your order #{r.randint(10000, 99999)} has shipped. Track it at https://www.{d}/orders/track",
        f"Thanks for your payment of ${r.randint(5, 99)}.00. Your receipt is available at https://www.{d}/receipts",
        f"We noticed a new sign-in to your {b} account from a new device. If this was you, no action is needed. "
        f"Review activity at https://www.{d}/security",
        f"Your {b} subscription renews on the {r.randint(1, 28)}th. Manage your plan at https://www.{d}/account",
        f"Someone requested a password reset for your {b} account. If you did not ask, ignore this message. "
        f"The link expires in one hour: https://www.{d}/reset",
    ])
    return f"{b} <no-reply@{d}>", r.choice(["Your order has shipped", "Receipt for your payment", "New sign-in", "Subscription reminder", "Password reset request"]), body


def fam_personal(r):
    n = person(r)
    s = ["Are we still on for the weekend hike? I can bring sandwiches.",
         "Thanks so much for the birthday wishes, it was a lovely day.",
         "Mom asked if you can come over for dinner on Sunday.",
         "I found the photos from our trip, sharing the album soon.",
         "Let me know if you want to split a cab to the station.",
         "Congratulations on the new job, that is wonderful news!",
         "Do you still have the book I lent you last month?"]
    return f"{n} <{n.split()[0].lower()}@{r.choice(FREE)}>", r.choice(["Weekend plans", "Thanks!", "Dinner on Sunday", "Photos", "Congrats"]), " ".join(pick(r, s, 1, 3))


def fam_school(r):
    s = ["The assignment deadline has been extended to next Thursday.",
         "Your seminar group meets in room 204 at 11am.",
         "The library has the reserved book ready for pickup at the front desk.",
         "Exam timetables are published on the student portal.",
         "Please submit your project proposal by the end of the month.",
         "The campus will be closed on Monday for the holiday."]
    return f"Registrar <office@{r.choice(['university', 'college', 'school'])}-{r.choice(COMPANIES)}.example.edu>", r.choice(["Course update", "Timetable", "Library notice"]), " ".join(pick(r, s, 1, 3))


def fam_it_legit(r):
    s = ["Scheduled maintenance will take place on Saturday between 1am and 3am.",
         "Reminder: our password policy requires at least 12 characters, details are on the intranet.",
         "Please restart your laptop to finish installing the security patch.",
         "The VPN certificate was renewed, no action is needed on your side.",
         "If you forgot your password, use the self-service portal on the intranet.",
         "Two-factor authentication will become mandatory next quarter, a how-to guide is on the wiki.",
         "The help desk is available from 8am to 6pm on working days."]
    return f"IT Helpdesk <helpdesk@{company_domain(r)}>", r.choice(["Maintenance notice", "Security reminder", "Laptop patch", "Helpdesk hours"]), " ".join(pick(r, s, 1, 3))


def fam_finance_legit(r):
    s = [f"Invoice {r.randint(1000, 9999)} for {r.choice(['September', 'October', 'November'])} is available in the customer portal.",
         f"Payment is due within 30 days. Our bank account details are printed on the invoice, IBAN GB00TEST{r.randint(100000, 999999)}.",
         "Your statement for last month is ready, no payment is required.",
         "Thank you for your payment, we have applied it to your account.",
         "Our office will be closed on Friday, payments will be processed on Monday.",
         "Please send the signed purchase order to the procurement team by email."]
    return f"Accounts <billing@{company_domain(r)}>", r.choice(["Invoice available", "Statement ready", "Payment received", "Purchase order"]), " ".join(pick(r, s, 1, 3))


def fam_newsletter_legit(r):
    s = ["This month we published three new tutorials and a guide to getting started.",
         "Join our free webinar on Thursday to learn about the new features.",
         "Here are the top stories of the week from our editors.",
         "You are receiving this newsletter because you subscribed on our website. You can unsubscribe at any time.",
         "Our community meetup is on Saturday, everyone is welcome."]
    return f"Newsletter <news@{company_domain(r)}>", r.choice(["Monthly newsletter", "This week's top stories", "Webinar invitation"]), " ".join(pick(r, s, 2, 3))


def fam_billing_notice_legit(r):
    plan = r.choice(["Premium", "Standard", "Pro", "Family", "Team"])
    price = r.choice(["9.99", "11.99", "14.99", "29", "49.00", "99"])
    s = [f"Your {plan} subscription renews on {r.choice(['12 November', '3 December', '1 January'])} for ${price}.",
         "You can manage or cancel your plan at any time in your account settings.",
         f"Thanks for your payment of ${price}. Your receipt is attached to this email.",
         "This is an automatic billing notice. If you have questions, contact our support team.",
         f"Invoice total ${price}, tax included, is due within 30 days."]
    return f"Billing <billing@{company_domain(r)}>", r.choice(["Your subscription renews soon", "Payment receipt", "Billing notice"]), " ".join(pick(r, s, 2, 3))


def fam_product_update_legit(r):
    s = [f"We have released version {r.randint(2, 9)}.{r.randint(0, 9)} with performance improvements and bug fixes.",
         "Read the release notes in our documentation to see what changed.",
         "Your account settings and saved projects are not affected by this update.",
         "You are receiving this message because you have an account with us.",
         "If you have problems after updating, reply to this email and our team will help."]
    return f"Product Team <updates@{company_domain(r)}>", r.choice(["Release notes", "We updated the app", "Service update"]), " ".join(pick(r, s, 2, 4))


def fam_shipping_legit(r):
    item = r.choice(["USB cable", "notebook", "desk lamp", "phone case", "water bottle"])
    s = [f"Your order has shipped. Item: {item} ${r.choice(['7.99', '12.50', '19.99'])}.",
         f"Shipping ${r.choice(['3.50', '4.99'])}. Tax ${r.choice(['0.80', '1.20', '2.10'])}. Total ${r.choice(['12.29', '18.69', '27.08'])}.",
         "Track your package from Your Orders. Delivery is expected on Thursday.",
         "Need to return something? Visit our returns centre within 30 days."]
    return f"Orders <orders@{company_domain(r)}>", r.choice(["Your order has shipped", "Order confirmation", "Delivery update"]), " ".join(pick(r, s, 2, 4))


SAFE_FAMILIES = {
    "workplace": fam_workplace, "dev_notifications": fam_dev_notifications, "orders_real": fam_orders_real,
    "personal": fam_personal, "school": fam_school, "it_legit": fam_it_legit, "finance_legit": fam_finance_legit,
    "newsletter_legit": fam_newsletter_legit, "billing_notice_legit": fam_billing_notice_legit,
    "product_update_legit": fam_product_update_legit, "shipping_legit": fam_shipping_legit,
}


# ---- spam ----
def fam_spam_lottery(r):
    s = ["CONGRATULATIONS! You have won the international lottery prize of $1,000,000!",
         "You are the lucky winner, claim your free gift now, no obligation!",
         "Reply with your details to receive your cash bonus today.",
         "Limited offer, 100% free, click here to claim your prize!!!"]
    return f"Prize Desk <winner@{r.choice(COMPANIES)}-prizes.{r.choice(BAD_TLDS)}>", r.choice(["YOU WON!!!", "Claim your free prize", "Lottery winner notification"]), " ".join(pick(r, s, 2, 4)) + " " + url_for(bad_domain(r), r)


def fam_spam_pharma(r):
    s = ["Cheap viagra and cialis, no prescription needed, order now!",
         "Best online pharmacy prices, discount on all medicines, buy now.",
         "Lose weight fast with our miracle pills, risk free trial.",
         "Special promotion: 80% off, limited time only."]
    return f"Pharmacy <sales@{r.choice(COMPANIES)}-rx.{r.choice(BAD_TLDS)}>", r.choice(["Cheap meds online", "Special discount", "Weight loss miracle"]), " ".join(pick(r, s, 2, 3)) + " " + url_for(bad_domain(r), r)


def fam_spam_money(r):
    s = ["Work from home and earn money fast, make $5,000 a week with no experience!",
         "Get rich with our proven system, double your income in 30 days.",
         "Free money, cash advance approved, loan approved instantly!",
         "Credit repair services, erase your debt today, no obligation."]
    return f"Income Team <info@{r.choice(COMPANIES)}-cash.{r.choice(BAD_TLDS)}>", r.choice(["Earn money from home", "Loan approved", "Get rich quick"]), " ".join(pick(r, s, 2, 3))


def fam_spam_promo(r):
    s = ["Huge sale this weekend only, exclusive deal for our valued members!",
         "Buy now and get free shipping, limited offer ends soon.",
         "Unlock a free trial of our premium plan today, order now.",
         "Special promotion on watches and handbags, cheap prices!!!"]
    return f"Deals <offers@{r.choice(COMPANIES)}-deals.{r.choice(['com', 'net'])}>", r.choice(["Weekend mega sale", "Exclusive deal for you", "Cheap watches"]), " ".join(pick(r, s, 2, 3)) + " Unsubscribe here."


def fam_spam_casino(r):
    s = ["Play online casino games and win big, get a free bonus on your first deposit!",
         "Crypto investment opportunity, double your coins, risk free guaranteed returns!!!",
         "Click here to join the VIP club and claim your free spins."]
    return f"Casino <vip@{r.choice(COMPANIES)}-bet.{r.choice(BAD_TLDS)}>", r.choice(["Free spins inside", "Win big tonight", "Invest and double your coins"]), " ".join(pick(r, s, 2, 3))


# ---- everyday marketing / promotional mail (NOT scams, but unwanted bulk advertising) ----
MKT_BRANDS = ["Zentra", "Pixelon", "Quorra", "Brightly", "Novacraft", "Fluxo", "Mintly", "Tandem", "Orbita",
              "Lexo", "Vantora", "Kelvio", "Sunvera", "Trailmark"]
MKT_FOOTERS = [
    "You are receiving this email because you signed up to receive offers and newsletters. "
    "If you do not want to receive this type of email in the future, please click to unsubscribe.",
    "Having trouble viewing this email? View in browser. Manage your preferences or unsubscribe at any time.",
    "Please do not reply to this email directly. Privacy Policy | Support | Unsubscribe.",
    "You received this message because you opted in on our website. To opt out of promotional emails, unsubscribe here.",
]
MKT_CTAS = ["Learn More >>", "Shop now", "Explore more", "Get started today", "Click here to claim", "View offer",
            "See more deals", "Buy now"]


def _mkt_sender(r, brand):
    # reuse the same domain style as legitimate mail, so the model must learn from the WORDS, not the domain
    local = r.choice(["offers", "hello", "news", "deals", "team", "promo", brand.lower()])
    return f"{brand} <{local}@{company_domain(r)}>"


def fam_spam_mkt_saas(r):
    b, disc = r.choice(MKT_BRANDS), r.choice([15, 20, 25, 30, 35, 40, 50, 60])
    p1 = r.choice([39, 49, 59, 79, 99, 129])
    tiers = [f"Annual plan ${p1} (was ${p1 + r.randint(20, 90)}).",
             f"Lifetime license ${p1 * 2 + r.randint(0, 60)} (was ${p1 * 3 + r.randint(0, 90)}).",
             f"Team bundle ${p1 * 3 + r.randint(0, 40)} for up to 5 seats.",
             "Free upgrades, priority support and 10GB cloud storage included.",
             f"Use all features on 3 devices. Limited time offer ends {r.choice(['Sunday', 'Friday', 'tonight'])}."]
    head = r.choice([f"{b} {r.choice(['Anniversary', 'Spring', 'Summer', 'Black Friday', 'Mega'])} Sale: up to {disc}% off",
                     f"Save {disc}% on every {b} plan this week", f"Last chance - {disc}% off {b} ends tonight",
                     f"{b} flash deal: {disc}% discount on all plans"])
    body = head + ". " + " ".join(pick(r, tiers, 3, 5)) + " " + r.choice(MKT_CTAS) + ". " + r.choice(MKT_FOOTERS)
    return _mkt_sender(r, b), head, body


def fam_spam_mkt_retail(r):
    b, disc = r.choice(MKT_BRANDS), r.choice([20, 30, 40, 50, 60, 70])
    items = ["shoes", "handbags", "watches", "headphones", "jackets", "sunglasses", "home decor", "kitchen gadgets"]
    s = [f"Up to {disc}% off on {', '.join(pick(r, items, 2, 3))}.",
         f"Free shipping on orders above ${r.choice([25, 50, 75, 100])}. Use promo code SAVE{disc} at checkout.",
         "New arrivals just dropped. Hurry, while stocks last!", "Don't miss out on our best price of the season.",
         f"Earn {r.choice([2, 5, 10])}% cashback on your first order. Members only early access."]
    head = r.choice([f"FLASH SALE: {disc}% off everything", f"{b} End of Season Sale - up to {disc}% off",
                     f"Weekend deals: save up to {disc}%", "Your exclusive coupon inside"])
    return _mkt_sender(r, b), head, head + ". " + " ".join(pick(r, s, 3, 5)) + " " + r.choice(MKT_CTAS) + ". " + r.choice(MKT_FOOTERS)


def fam_spam_mkt_course(r):
    b, disc, price = r.choice(MKT_BRANDS), r.choice([25, 40, 50, 60]), r.choice([99, 149, 199, 299])
    s = [f"Enrol today and get {disc}% off our {r.choice(['data science', 'design', 'marketing', 'coding', 'photography'])} course.",
         f"Early bird price ${price} instead of ${price * 2}. Bonus bundle of templates included.",
         "Limited seats, the offer ends soon. Don't miss out!", "Join thousands of learners and get certified.",
         "Start your free trial now and upgrade any time."]
    head = r.choice([f"Last chance: {disc}% off ends tonight", f"Early bird sale - save {disc}%", f"{b} Academy mega sale"])
    return _mkt_sender(r, b), head, head + ". " + " ".join(pick(r, s, 3, 5)) + " " + r.choice(MKT_CTAS) + ". " + r.choice(MKT_FOOTERS)


def fam_spam_mkt_travel_finance(r):
    b = r.choice(MKT_BRANDS)
    s = [f"Holiday deals: flights from ${r.choice([149, 199, 249])} and hotels up to {r.choice([30, 40, 50])}% off.",
         f"Pre-approved personal loan at {r.choice(['8.99', '9.99', '11.5'])}% - apply now, no paperwork.",
         "Book now and save big on your next getaway. Price drop on selected destinations!",
         "Special offer: new credit card with cashback and reward points, apply today.",
         "Hurry, this offer expires soon. Exclusive deal for our valued subscribers."]
    head = r.choice(["Your dream holiday, now cheaper", "Pre-approved offer inside", "Mega travel sale",
                     "Festive offer: save big"])
    return _mkt_sender(r, b), head, head + ". " + " ".join(pick(r, s, 3, 4)) + " " + r.choice(MKT_CTAS) + ". " + r.choice(MKT_FOOTERS)


SPAM_FAMILIES = {"lottery": fam_spam_lottery, "pharma": fam_spam_pharma, "money": fam_spam_money,
                 "promo": fam_spam_promo, "casino": fam_spam_casino,
                 "mkt_saas": fam_spam_mkt_saas, "mkt_retail": fam_spam_mkt_retail,
                 "mkt_course": fam_spam_mkt_course, "mkt_travel_finance": fam_spam_mkt_travel_finance}


# ---- phishing ----
def _brand_sender(r, brand):
    kind = r.choice(["lookalike", "wordy", "free", "tld", "legit_looking"])
    if kind == "free":
        return f"{brand} Support <{re.sub(r'[^a-z]', '', brand.lower())}.support.{r.randint(10, 999)}@{r.choice(FREE)}>"
    if kind == "legit_looking":
        return f"{brand} <noreply@{company_domain(r)}>"
    return f"{brand} {r.choice(['Security', 'Support', 'Team', 'Alerts'])} <{r.choice(['service', 'noreply', 'security'])}@{bad_domain(r, brand)}>"


def fam_phish_credentials(r):
    b = r.choice(BRANDS)
    s = [f"We detected unusual activity on your {b} account.",
         "Your account has been suspended. Verify your account immediately to restore access.",
         "Confirm your identity and update your password within 24 hours or your account will be closed.",
         "Sign in to review recent activity and verify your login credentials.",
         "Failure to act will result in permanent suspension.",
         "For your security, we need you to validate your account now."]
    return _brand_sender(r, b), r.choice([f"Action required: {b} account", "URGENT: verify your account", "Security alert", "Your account has been locked"]), " ".join(pick(r, s, 3, 5)) + " " + url_for(bad_domain(r, b), r)


def fam_phish_bank_kyc(r):
    b = r.choice(["HDFC Bank", "SBI", "ICICI Bank", "Paytm", "Chase", "Wells Fargo"])
    s = [f"Dear customer, your {b} KYC is incomplete and your account will be blocked today.",
         "Update your PAN, account number and OTP immediately to avoid suspension.",
         "Your net banking access has been restricted due to suspicious activity.",
         "Click the link below to complete your verification and enter the security code.",
         "Do not share this message, this is an official bank notice."]
    return _brand_sender(r, b), r.choice(["KYC update required", "Your account will be blocked", "Net banking suspended"]), " ".join(pick(r, s, 3, 5)) + " " + url_for(bad_domain(r, b), r)


def fam_phish_delivery(r):
    b = r.choice(["DHL", "FedEx", "Amazon"])
    s = [f"Your {b} parcel could not be delivered because of an unpaid customs fee of ${r.randint(2, 9)}.99.",
         "Pay the small fee within 48 hours or the package will be returned to the sender.",
         "Confirm your address and card number to reschedule delivery.",
         "Track your package and pay the redelivery fee using the link below."]
    return _brand_sender(r, b), r.choice(["Parcel delivery failed", "Redelivery fee required", "Package on hold"]), " ".join(pick(r, s, 2, 4)) + " " + url_for(bad_domain(r, b), r)


def fam_phish_invoice(r):
    s = [f"Please find the overdue invoice {r.randint(1000, 9999)} attached, payment is overdue.",
         "Our bank details have changed, please remit payment to the updated bank account by wire transfer today.",
         "Payment failed, update your billing information to avoid late fees.",
         "Open the attached document and enable content to view the remittance advice.",
         "Kindly confirm once the transfer is completed."]
    d = r.choice([company_domain(r), bad_domain(r)])
    return f"Accounts <accounts@{d}>", r.choice(["Overdue invoice", "Updated bank details", "Payment failed", "Remittance advice"]), " ".join(pick(r, s, 2, 4)) + (" " + url_for(d, r) if r.random() < 0.5 else "")


def fam_phish_bec(r):
    n = person(r)
    s = ["I am in a meeting and cannot take calls, are you available for a quick task?",
         "I need you to buy gift cards for a client today and send me the codes.",
         "Please process an urgent wire transfer to the vendor, I will send the bank details shortly.",
         "Keep this confidential and do not tell anyone until it is done.",
         "I will reimburse you later, reply as soon as you see this."]
    sender = f"{n} <{n.split()[0].lower()}.{r.choice(['office', 'ceo', 'director'])}{r.randint(1, 99)}@{r.choice(FREE)}>"
    return sender, r.choice(["Quick favour", "Urgent request", "Are you available?", "Task for today"]), " ".join(pick(r, s, 3, 4))


def fam_phish_gov_refund(r):
    s = [f"You are eligible for a tax refund of ${r.randint(200, 900)}.00. Submit your bank account details to receive it.",
         "Your refund will expire in 24 hours, claim it now using the secure form.",
         "Provide your social security number and card number to verify your identity.",
         "This is an official notice from the revenue department."]
    return f"Revenue Department <refunds@{bad_domain(r, 'Government')}>", r.choice(["Tax refund notification", "Refund pending", "Official notice"]), " ".join(pick(r, s, 2, 4)) + " " + url_for(bad_domain(r), r)


def fam_phish_shared_doc(r):
    b = r.choice(["Dropbox", "Google", "Microsoft", "DocuSign", "LinkedIn"])
    s = [f"{person(r)} shared a document with you on {b}.",
         "Sign in with your email password to view the secure file.",
         "The document expires today, log in to download it.",
         "Review and sign the pending agreement using your credentials."]
    return _brand_sender(r, b), r.choice(["Document shared with you", "Please review and sign", "File waiting for you"]), " ".join(pick(r, s, 2, 4)) + " " + url_for(bad_domain(r, b), r)


def fam_phish_malware(r):
    s = ["Please open the attached file and enable macros to view the protected content.",
         "Run the attached update installer to fix the security problem on your computer.",
         "Download the zip archive, the password is 1234, and run the setup file.",
         "Your PC is infected, install the attached tool immediately."]
    fname = r.choice(["invoice.pdf.exe", "update.scr", "document.docm", "setup.js", "report.zip", "scan.hta"])
    return f"IT Support <support@{bad_domain(r)}>", r.choice(["Security update", "Protected document", "Important attachment"]), " ".join(pick(r, s, 2, 3)) + f" Attachment: {fname}"


def fam_phish_job_scam(r):
    s = ["We reviewed your resume and would like to offer you a work from home position paying $4,000 a week.",
         "To start, we will send you a cheque, deposit it and transfer the processing fee by Western Union.",
         "Send your bank account number, date of birth and a copy of your ID to receive the contract.",
         "Keep this confidential until you are onboarded."]
    sender = f"HR Recruiter <hr.{r.choice(COMPANIES)}{r.randint(1, 99)}@{r.choice(FREE)}>"
    return sender, r.choice(["Job offer", "Work from home opportunity", "Your application was accepted"]), " ".join(pick(r, s, 3, 4))


def fam_phish_subscription(r):
    b = r.choice(["Netflix", "Microsoft", "Apple", "Amazon", "PayPal"])
    s = [f"Your {b} subscription payment was declined and your membership will be cancelled.",
         "Update your payment method and billing information now to keep your account active.",
         f"A charge of ${r.randint(199, 499)}.00 was made to your card. If this was not you, call support or sign in now.",
         "Verify your credit card number and security code to stop the charge."]
    return _brand_sender(r, b), r.choice(["Payment declined", "Subscription renewal charge", "Billing problem"]), " ".join(pick(r, s, 2, 4)) + " " + url_for(bad_domain(r, b), r)


PHISH_FAMILIES = {
    "credentials": fam_phish_credentials, "bank_kyc": fam_phish_bank_kyc, "delivery": fam_phish_delivery,
    "invoice": fam_phish_invoice, "bec": fam_phish_bec, "gov_refund": fam_phish_gov_refund,
    "shared_doc": fam_phish_shared_doc, "malware": fam_phish_malware, "job_scam": fam_phish_job_scam,
    "subscription": fam_phish_subscription,
}


def make_email_rows(per_family_safe=420, per_family_spam=300, per_family_phish=320, seed=7):
    r = random.Random(seed)
    rows = []
    for label, fams, n in (("safe", SAFE_FAMILIES, per_family_safe), ("spam", SPAM_FAMILIES, per_family_spam),
                           ("phishing", PHISH_FAMILIES, per_family_phish)):
        for fam, fn in fams.items():
            for _ in range(n):
                sender, subject, body = fn(r)
                subject, body = add_noise(subject, r), add_noise(body, r)
                if r.random() < 0.12:                      # some messages are pasted without a sender
                    sender = ""
                urls = extract_urls_from_text(body)
                urls += extract_bare_domains(body, urls)
                if fam.startswith("mkt_"):               # HTML marketing mail hides many tracking links
                    urls += [f"https://track.example/{i}" for i in range(r.randint(5, 18))]
                rows.append({"text": build_model_text(sender, subject, body, urls), "label": label,
                             "family": f"{label}/{fam}"})
    r.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# PDF families (real PDFs generated in memory, features from the static analyzer)
# ---------------------------------------------------------------------------
SAFE_PDF_TOPICS = [
    ["Quarterly Report", "Revenue increased by {n} percent compared with the previous quarter.",
     "The marketing team launched two campaigns and the support team reduced response times.",
     "Next quarter we plan to hire three engineers and open a small regional office."],
    ["Meeting Minutes", "Attendees reviewed the project timeline and agreed on the next milestones.",
     "Action items were assigned to the design and engineering leads.", "The next meeting is on {day}."],
    ["User Guide", "Step one: install the application using the provided installer.",
     "Step two: open the settings page and choose your preferred language.", "For more help see the documentation pages."],
    ["Invoice {n}", "Consulting services for the month of October.", "Amount due: {n}.00 within 30 days.",
     "Bank account: 12345678, IBAN GB00TEST00001234, SWIFT TESTGB2L. Thank you for your business."],
    ["Lecture Notes", "Today we covered graph traversal, breadth first search and depth first search.",
     "Remember that complexity depends on the number of vertices and edges.", "Homework is due next week."],
    ["Travel Itinerary", "Flight departs at {t} from terminal two.", "Hotel check-in is after 2pm, breakfast is included.",
     "Please carry a printed copy of the booking confirmation."],
    ["Recipe Book", "Mix flour, sugar and butter until smooth.", "Bake for twenty five minutes at medium heat.", "Serve warm with cream."],
    ["Company Policy", "All employees must complete safety training each year.",
     "Visitors must sign in at reception and wear a badge.", "Questions can be sent to the human resources team."],
]
BENIGN_LINKS = ["https://www.example.com/docs", "https://github.com/example/project", "https://www.wikipedia.org/wiki/Graph",
                "https://docs.python.org/3/", "https://www.google.com/maps"]


def _safe_text(r):
    topic = list(r.choice(SAFE_PDF_TOPICS))
    lines = [t.format(n=r.randint(3, 40), day=r.choice(DAYS), t=r.choice(TIMES)) for t in topic]
    return lines


def _phish_text(r):
    b = r.choice(BRANDS)
    pool = [f"{b} Security Notice", "URGENT: your account has been suspended.", "Verify your account password immediately to avoid closure.",
            "Confirm your identity and enter your one-time password (OTP).", "Update your billing information and card number now.",
            "Send the outstanding payment by wire transfer to the new bank account today.",
            "Failure to respond within 24 hours will result in permanent suspension.", "Buy gift cards and send the codes to the address below.",
            "Sign in to restore access to your account."]
    return pick(r, pool, 3, 6), b


def make_pdf_rows(n_safe=360, n_susp=240, n_mal=300, seed=11):
    """Build PDFs in memory and extract the REAL features with the project's analyzer."""
    from pdf_analyzer import extract_pdf_findings
    from pdf_model import PDF_FEATURE_NAMES, findings_to_features
    r = random.Random(seed)
    rows = []
    tmp = tempfile.mkdtemp(prefix="pg_train_")

    def emit(data, label, fam):
        path = os.path.join(tmp, "t.pdf")
        with open(path, "wb") as fh:
            fh.write(data)
        findings = extract_pdf_findings(path)
        if findings["analysis_issues"] and label == "safe":
            return                                       # only fully-analysed PDFs are used for training
        feats = findings_to_features(findings)
        feats["label"], feats["family"] = label, fam
        rows.append(feats)

    comp = lambda: r.choice([None, "flate", "a85flate"])
    js = b"<< /S /JavaScript /JS (var testOnly = 1;) >>"
    for _ in range(n_safe):
        k = r.choice(["plain", "links", "form", "long"])
        lines = _safe_text(r)
        if k == "long":
            lines = lines * r.randint(2, 4)
        links = pick(r, BENIGN_LINKS, 1, 3) if k == "links" else []
        if k == "form":
            emit(F.widget_form_pdf(r.sample(["First name", "Last name", "Comments", "City", "Department", "Date"], 3)), "safe", "safe/form")
        else:
            emit(F.text_pdf(lines, comp(), links=links), "safe", f"safe/{k}")
    for _ in range(n_susp):
        k = r.choice(["js_only", "lookalike_link", "ip_link", "password_form", "embedded_file", "shortener", "submit_form", "tld_link"])
        lines = _safe_text(r)
        if k == "js_only":
            emit(F.text_pdf(lines, comp(), catalog_extra=b"/Names << /JavaScript 5 0 R >>",
                            extra_objects=[b"<< /Names [(calc) 6 0 R] >>", js]), "suspicious", "susp/js_only")
        elif k == "lookalike_link":
            emit(F.text_pdf(lines, comp(), links=[f"http://{typo_brand(r.choice(BRANDS), r)}.{r.choice(['com', 'net'])}/verify"]), "suspicious", "susp/lookalike")
        elif k == "ip_link":
            emit(F.text_pdf(lines + [f"Download: http://203.0.113.{r.randint(2, 250)}/files/{r.randint(1, 99)}"], comp()), "suspicious", "susp/ip")
        elif k == "password_form":
            emit(F.widget_form_pdf(["Full name", r.choice(["Password", "PIN", "Card number", "OTP"])]), "suspicious", "susp/password_form")
        elif k == "embedded_file":
            emit(F.text_pdf(lines, comp(), catalog_extra=b"/Names << /EmbeddedFiles 5 0 R >>",
                            extra_objects=[b"<< /Names [(notes.txt) 6 0 R] >>",
                                           b"<< /Type /Filespec /F (notes.txt) /EF << /F 7 0 R >> >>",
                                           b"<< /Type /EmbeddedFile /Length 0 >>\nstream\n\nendstream"]), "suspicious", "susp/embedded")
        elif k == "shortener":
            emit(F.text_pdf(lines + ["Confirm your account here: " + r.choice(["https://bit.ly/3xAbC", "https://tinyurl.com/y8abc", "https://cutt.ly/Qwe1"])], comp()), "suspicious", "susp/shortener")
        elif k == "submit_form":
            emit(F.text_pdf(lines, comp(), extra_objects=[b"<< /S /SubmitForm /F (http://203.0.113.9/collect) >>"]),
                 "suspicious", "susp/submit")
        else:
            emit(F.text_pdf(lines, comp(), links=[f"http://{r.choice(WORDS)}-{r.choice(COMPANIES)}.{r.choice(BAD_TLDS)}/login"]), "suspicious", "susp/tld")
    for _ in range(n_mal):
        k = r.choice(["openaction_js", "obfuscated", "objstm", "phish_text", "phish_lookalike", "launch", "exe_embedded"])
        lines = _safe_text(r)
        if k == "openaction_js":
            emit(F.text_pdf(lines, comp(), catalog_extra=b"/OpenAction 5 0 R", extra_objects=[js]), "malicious", "mal/openaction_js")
        elif k == "obfuscated":
            emit(F.obfuscated_js_pdf(), "malicious", "mal/obfuscated")
        elif k == "objstm":
            emit(F.objstm_javascript_pdf(), "malicious", "mal/objstm")
        elif k == "phish_text":
            text, _b = _phish_text(r)
            emit(F.text_pdf(text, comp()), "malicious", "mal/phish_text")
        elif k == "phish_lookalike":
            text, b = _phish_text(r)
            emit(F.text_pdf(text, comp(), links=[f"http://{typo_brand(b, r)}-{r.choice(WORDS)}.com/login"]), "malicious", "mal/phish_lookalike")
        elif k == "launch":
            emit(F.text_pdf(lines, comp(), catalog_extra=b"/OpenAction 5 0 R", extra_objects=[b"<< /S /Launch /F (calc.exe) >>"]), "malicious", "mal/launch")
        else:
            emit(F.text_pdf(lines, comp(), catalog_extra=b"/Names << /EmbeddedFiles 5 0 R >> /OpenAction 8 0 R",
                            extra_objects=[b"<< /Names [(update.exe) 6 0 R] >>",
                                           b"<< /Type /Filespec /F (update.exe) /EF << /F 7 0 R >> >>",
                                           b"<< /Type /EmbeddedFile /Length 0 >>\nstream\n\nendstream", js]), "malicious", "mal/exe_embedded")
    r.shuffle(rows)
    return rows, PDF_FEATURE_NAMES


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["email", "pdf"])
    args = ap.parse_args()
    if args.only in (None, "email"):
        rows = make_email_rows()
        write_csv(os.path.join(DATA_DIR, "email_dataset.csv"), rows, ["text", "label", "family"])
        from collections import Counter
        print("email rows:", len(rows), dict(Counter(x["label"] for x in rows)))
    if args.only in (None, "pdf"):
        rows, names = make_pdf_rows()
        write_csv(os.path.join(DATA_DIR, "pdf_features.csv"), rows, list(names) + ["label", "family"])
        from collections import Counter
        print("pdf rows:", len(rows), dict(Counter(x["label"] for x in rows)))


if __name__ == "__main__":
    main()
