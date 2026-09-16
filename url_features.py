from urllib.parse import urlparse


def extract_url_features(url):

    parsed = urlparse(url)

    domain = parsed.netloc

    if "@" in domain:
        domain = domain.split("@")[-1]

    domain = domain.split(":")[0]

    url_length = len(url)

    domain_length = len(domain)

    is_domain_ip = 0

    parts = domain.split(".")

    if len(parts) == 4:

        try:

            if all(
                0 <= int(part) <= 255
                for part in parts
            ):
                is_domain_ip = 1

        except ValueError:
            is_domain_ip = 0

    no_of_subdomain = max(
        0,
        len(parts) - 2
    )

    no_of_letters = sum(
        c.isalpha()
        for c in url
    )

    no_of_digits = sum(
        c.isdigit()
        for c in url
    )

    no_of_equals = url.count("=")

    no_of_qmark = url.count("?")

    no_of_ampersand = url.count("&")

    special_characters = set(
        "!#$%&'()*+,/:;<=>?@[\\]^_`{|}~"
    )

    no_of_other_special = sum(
        c in special_characters
        for c in url
    )

    is_https = 1 if parsed.scheme.lower() == "https" else 0

    features = {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "NoOfSubDomain": no_of_subdomain,
        "NoOfLettersInURL": no_of_letters,
        "NoOfDegitsInURL": no_of_digits,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special,
        "IsHTTPS": is_https
    }

    return features


if __name__ == "__main__":

    url = input("Enter website URL: ")

    features = extract_url_features(url)

    print("\n---------- URL FEATURES ----------")

    for name, value in features.items():

        print(
            f"{name}: {value}"
        )

    print("----------------------------------")