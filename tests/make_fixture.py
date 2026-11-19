"""Write a small fake archive export so the pipeline can be tested offline."""
import csv
import io
import random
import string
import zipfile

PRODUCTS = {
    "Credit reporting or other personal consumer reports": ["Incorrect information on your report", "Improper use of your report"],
    "Debt collection": ["Attempts to collect debt not owed", "Communication tactics"],
    "Mortgage": ["Trouble during payment process"],
    "Checking or savings account": ["Managing an account"],
}
HEADER = ["Date received", "Product", "Sub-product", "Issue", "Sub-issue", "Consumer complaint narrative",
          "Company public response", "Company", "State", "ZIP code", "Tags", "Consumer consent provided?",
          "Submitted via", "Date sent to company", "Company response to consumer", "Timely response?",
          "Consumer disputed?", "Complaint ID"]


def write(path, n=400, seed=1):
    rnd = random.Random(seed)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    for i in range(n):
        p = rnd.choice(list(PRODUCTS))
        issue = rnd.choice(PRODUCTS[p])
        words = " ".join("".join(rnd.choices(string.ascii_lowercase, k=6)) for _ in range(8))
        narr = f"I have a problem with my {p.lower()}: {issue.lower()}. " * 4 + words
        if i % 50 == 0:
            narr = ""                                   # no narrative
        if i % 40 == 1:
            narr = "Template credit repair letter. " * 10  # duplicate template
        resp = rnd.choice(["Closed with explanation", "Closed with monetary relief", "In progress"])
        tags = rnd.choice(["", "", "Servicemember", "Older American"])
        w.writerow(["01/15/25", p, "", issue, "", narr, "", "Acme Bank", "CA", "90000", tags, "Consent provided",
                    "Web", "01/16/25", resp, rnd.choice(["Yes", "Yes", "No"]), "N/A", str(10000 + i)])
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("complaints.csv", buf.getvalue())
