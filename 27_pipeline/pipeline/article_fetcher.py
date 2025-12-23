from playwright.sync_api import sync_playwright

def fetch_article_text(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_timeout(4000)

        text = page.inner_text("body")
        browser.close()

        return text.strip()