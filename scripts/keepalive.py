from playwright.sync_api import sync_playwright
import time

STREAMLIT_URL = "https://spotifyreviewscrapper-02.streamlit.app/"

def wake_app():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print(f"Visiting {STREAMLIT_URL}...")
        page.goto(STREAMLIT_URL, timeout=60000)
        time.sleep(5)

        # Check if sleeping and click wake button
        try:
            wake_btn = page.locator("text=Yes, get this app back up!")
            if wake_btn.is_visible(timeout=5000):
                print("App was sleeping — waking it up...")
                wake_btn.click()
                time.sleep(30)
                print("App woken successfully.")
            else:
                print("App already awake.")
        except Exception:
            print("App already awake (no sleep screen detected).")

        # Stay on page briefly so visit registers
        time.sleep(10)
        browser.close()

if __name__ == "__main__":
    wake_app()
