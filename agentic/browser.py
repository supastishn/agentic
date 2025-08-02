from playwright.sync_api import sync_playwright
import time

class BrowserManager:
    """Manages a single Playwright browser instance for the agent's session."""
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.closed = True

    def start(self) -> str:
        """Starts a headless browser instance."""
        if not self.closed:
            return "Browser is already running."
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()
            self.closed = False
            return "Browser started successfully."
        except Exception as e:
            return f"Error starting browser: {e}. Have you run 'playwright install'?"

    def close(self) -> str:
        """Closes the headless browser instance."""
        if self.closed:
            return "Browser is already closed."
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.closed = True
            return "Browser closed successfully."
        except Exception as e:
            return f"Error closing browser: {e}"

    def navigate(self, url: str) -> str:
        """Navigates the browser to a specific URL."""
        if self.closed or not self.page:
            return "Error: Browser is not running. Please start it first."
        try:
            self.page.goto(url, wait_until="domcontentloaded")
            time.sleep(2) # A simple wait for async operations to settle
            return f"Successfully navigated to {url}."
        except Exception as e:
            return f"Error navigating to {url}: {e}"

    def get_content(self) -> str:
        """Returns the full HTML content of the current browser page."""
        if self.closed or not self.page:
            return "Error: Browser is not running. Please start it first."
        try:
            return self.page.content()
        except Exception as e:
            return f"Error getting page content: {e}"

    def click(self, selector: str) -> str:
        """Clicks on an element on the current page using a CSS selector."""
        if self.closed or not self.page:
            return "Error: Browser is not running. Please start it first."
        try:
            self.page.click(selector, timeout=5000)
            time.sleep(2) # A simple wait for potential page changes
            return f"Clicked element with selector '{selector}'."
        except Exception as e:
            return f"Error clicking element '{selector}': {e}"

    def type_text(self, selector: str, text: str) -> str:
        """Types text into an element on the current page using a CSS selector."""
        if self.closed or not self.page:
            return "Error: Browser is not running. Please start it first."
        try:
            self.page.type(selector, text, timeout=5000)
            return f"Typed text into element with selector '{selector}'."
        except Exception as e:
            return f"Error typing into element '{selector}': {e}"

# Global instance for the agent session to share a single browser
browser_manager = BrowserManager()
