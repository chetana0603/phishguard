"""Local smoke test for the frozen PhishGuard predictor."""

from phishguard.inference import PhishGuardPredictor


def main() -> None:
    """Run a few inference examples."""
    predictor = PhishGuardPredictor()

    urls = [
        "https://example.com",
        "https://example.com/",
        "http://www.example.com",
    ]

    for result in predictor.predict_many(urls):
        print(result.to_dict())


if __name__ == "__main__":
    main()
